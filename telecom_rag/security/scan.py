"""Pre-upload PII/secret scanner (Issue #15).

Runs a regex catalog against extracted PDF text to detect known
secret formats and strong PII patterns. Used by
:func:`app.pages.upload._ingest_upload` to block uploads before any
storage happens.

Design choices
--------------

1. **Pure-data, no I/O.** Every function takes text (or a list of
   pages from :func:`telecom_rag.ingestion.docling_parser.parse_pdf`)
   and returns a list of :class:`Finding`. No S3, no Chroma, no
   Streamlit, no logger. Caller decides what to do with findings.

2. **Three severity tiers:**
   - ``high`` — confirmed secret format (``AKIA…``, ``sk-…``,
     ``ghp_…``, etc.). ALWAYS blocks. Source: ``docs/SECURITY.md``
     gitleaks allowlist.
   - ``medium`` — strong PII (``SSN``, Luhn-valid credit card).
     ALWAYS blocks. Source: customer PII heuristics.
   - ``low`` — heuristic (``email``, ``phone``, internal IP). Logs
     only, does NOT block. Vendor manuals legitimately contain
     support contacts.

3. **Redacted output.** :attr:`Finding.redacted` masks the
   sensitive portion so the value can be rendered in a Streamlit
   error message or log line without leaking the secret itself.
   Regression test ``test_finding_redacted_does_not_leak_full_secret``
   pins this.

4. **Regex catalog, not a detection library.** Presidio / detect-secrets
   are heavy (50 MB / 5 s import). The regex catalog runs in
   microseconds and covers the unambiguous formats. Named-entity
   detection (names in prose, addresses) is deferred — see the
   plan's "Out of scope" section.

Catalog source
--------------

All ``high`` patterns are derived from the gitleaks default ruleset
documented in ``docs/SECURITY.md`` (§ "What gitleaks Catches"). The
SSN and credit card patterns are well-known regexes from NIST SP
800-63B and the Luhn algorithm respectively.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from telecom_rag.security.policy import (
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    block_for_tier,
)


# ---------------------------------------------------------------------------
# Severity tier (re-exported from policy.py for backward compatibility)
# ---------------------------------------------------------------------------
#
# The tier constants live in :mod:`telecom_rag.security.policy`
# (Issue #18 AC2 — centralization). They are re-exported here so
# existing imports (``from telecom_rag.security.scan import
# SEVERITY_HIGH``) keep working without modification. New code
# should import from :mod:`policy` directly to make the dependency
# direction explicit.
#
# The single-line re-export below is the entire back-compat surface.
SEVERITY_HIGH = SEVERITY_HIGH      # noqa: F811 — re-export
SEVERITY_MEDIUM = SEVERITY_MEDIUM  # noqa: F811 — re-export
SEVERITY_LOW = SEVERITY_LOW        # noqa: F811 — re-export


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """A single detected instance of a secret or PII pattern.

    Attributes
    ----------
    pattern_id:
        Stable identifier for the pattern (e.g. ``"aws_access_key"``).
        Use this in tests and log lines so renaming a pattern_id in
        a future refactor is a deliberate breaking change.
    severity:
        One of :data:`SEVERITY_HIGH`, :data:`SEVERITY_MEDIUM`,
        :data:`SEVERITY_LOW`. The upload page's block decision is
        driven by this field.
    match:
        The actual matched substring from the text. NEVER rendered
        in a user-facing surface — use :attr:`redacted` for that.
    redacted:
        A safe-to-render representation of the match with the
        sensitive middle portion masked. E.g. an AWS-style access
        key id like ``AKIA****MPLE`` is shown with the inner
        characters masked.
    page_number:
        1-indexed page number where the match was found. ``None``
        when scanning raw text via :func:`scan_text` (no page
        context).
    """

    pattern_id: str
    severity: str
    match: str
    redacted: str
    page_number: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly representation. Excludes the raw ``match``
        field (only ``redacted`` is safe to serialize to logs or
        ship to LangSmith)."""
        return {
            "pattern_id": self.pattern_id,
            "severity": self.severity,
            "redacted": self.redacted,
            "page_number": self.page_number,
        }


# ---------------------------------------------------------------------------
# Redaction helpers
# ---------------------------------------------------------------------------


def _redact_keep_first_last(s: str, *, keep: int = 4) -> str:
    """Mask the middle of ``s``, keeping the first and last ``keep``
    characters. For a 16-character AWS access key with ``keep=4``,
    produces ``AKIA****MPLE`` (4 + 4 stars + 4).

    ``keep=0`` is a special case: the entire value is masked
    (``"*" * len(s)``). Needed for SSN redaction where we want
    ``***-**-****``-style full masking without leaking the
    leading or trailing digit groups.

    For short strings where ``len(s) <= 2 * keep``, returns
    ``"*" * len(s)`` so the entire value is masked.
    """
    if keep <= 0:
        return "*" * len(s)
    if len(s) <= 2 * keep:
        return "*" * len(s)
    return f"{s[:keep]}{'*' * (len(s) - 2 * keep)}{s[-keep:]}"


def _redact_keep_prefix(s: str, *, prefix: int = 4) -> str:
    """Keep the first ``prefix`` characters, mask the rest. Useful
    for tokens where the prefix is a public-format identifier
    (``AKIA``, ``sk-``, ``ghp_``) and the rest is the secret half.

    For a 16-character AWS key with ``prefix=4``: ``AKIA************``.
    """
    if len(s) <= prefix:
        return "*" * len(s)
    return f"{s[:prefix]}{'*' * (len(s) - prefix)}"


# ---------------------------------------------------------------------------
# Pattern catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pattern:
    """Internal catalog entry. ``redact`` is a callable that takes
    the raw match and returns a redacted string. ``severity`` is
    one of the SEVERITY_* constants."""

    pattern_id: str
    severity: str
    regex: re.Pattern[str]
    redact: Any  # Callable[[str], str]


def _aws_access_key() -> _Pattern:
    """AWS access key ID — 16 uppercase alphanumerics after ``AKIA``.

    gitleaks default rule. The prefix is fixed (``AKIA``) so this
    has very low false-positive risk.
    """
    return _Pattern(
        pattern_id="aws_access_key",
        severity=SEVERITY_HIGH,
        regex=re.compile(r"AKIA[0-9A-Z]{16}"),
        redact=lambda m: _redact_keep_prefix(m, prefix=4),
    )


def _has_class_diversity(s: str) -> bool:
    """True iff ``s`` has at least one lowercase, one uppercase, AND
    one digit. Used as a post-filter for the AWS-secret-key pattern
    to reduce false positives on hashes/IDs that happen to be
    exactly 40 chars in the base64 alphabet."""
    has_lower = any(c.islower() for c in s)
    has_upper = any(c.isupper() for c in s)
    has_digit = any(c.isdigit() for c in s)
    return has_lower and has_upper and has_digit


def _aws_secret_key() -> _Pattern:
    """AWS secret access key — 40-char base64.

    gitleaks default rule. Higher false-positive risk than access
    key (any 40-char base64 string matches), so we add a
    class-diversity post-filter: the match must contain at least
    one lowercase, one uppercase, and one digit character. Random
    base64 blobs fail one of those classes; an actual AWS secret
    passes all three.

    Note: this filter assumes AWS secrets are alphanumeric+symbol
    with case-mixed content. A pure-symbol 40-char secret would
    be missed. AWS docs explicitly show case-mixed examples; the
    filter matches real-world AWS secret shapes.
    """
    return _Pattern(
        pattern_id="aws_secret_key",
        severity=SEVERITY_HIGH,
        regex=re.compile(
            r"(?<![A-Za-z0-9/+=])"
            r"[A-Za-z0-9/+=]{40}"
            r"(?![A-Za-z0-9/+=])"
        ),
        redact=lambda m: _redact_keep_first_last(m, keep=4),
    )


def _openai_key() -> _Pattern:
    """OpenAI project key (``sk-…``, ``sk-proj-…``, ``sk-svcacct-…``).

    Excludes ``sk-ant-…`` (Anthropic has its own pattern).
    """
    return _Pattern(
        pattern_id="openai_key",
        severity=SEVERITY_HIGH,
        regex=re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_-]{20,}"),
        redact=lambda m: _redact_keep_prefix(m, prefix=7),
    )


def _anthropic_key() -> _Pattern:
    """Anthropic API key (``sk-ant-…``)."""
    return _Pattern(
        pattern_id="anthropic_key",
        severity=SEVERITY_HIGH,
        regex=re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
        redact=lambda m: _redact_keep_prefix(m, prefix=7),
    )


def _google_api_key() -> _Pattern:
    """Google API key (``AIza…``, 39 chars total)."""
    return _Pattern(
        pattern_id="google_api_key",
        severity=SEVERITY_HIGH,
        regex=re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
        redact=lambda m: _redact_keep_prefix(m, prefix=4),
    )


def _github_pat() -> _Pattern:
    """GitHub personal access token (``ghp_…``)."""
    return _Pattern(
        pattern_id="github_pat",
        severity=SEVERITY_HIGH,
        regex=re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
        redact=lambda m: _redact_keep_prefix(m, prefix=4),
    )


def _langsmith_key() -> _Pattern:
    """LangSmith API key (``lsv2_pt_…`` or ``lsv2_sk_…``)."""
    return _Pattern(
        pattern_id="langsmith_key",
        severity=SEVERITY_HIGH,
        regex=re.compile(r"\blsv2_(?:pt|sk)_[A-Za-z0-9]{32,}\b"),
        redact=lambda m: _redact_keep_prefix(m, prefix=8),
    )


def _slack_token() -> _Pattern:
    """Slack token (``xox[baprs]-…``)."""
    return _Pattern(
        pattern_id="slack_token",
        severity=SEVERITY_HIGH,
        regex=re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
        redact=lambda m: _redact_keep_prefix(m, prefix=5),
    )


def _stripe_live_key() -> _Pattern:
    """Stripe live secret key (``sk_live_…``)."""
    return _Pattern(
        pattern_id="stripe_live_key",
        severity=SEVERITY_HIGH,
        regex=re.compile(r"\bsk_live_[A-Za-z0-9]{24,}\b"),
        redact=lambda m: _redact_keep_prefix(m, prefix=8),
    )


def _ssn() -> _Pattern:
    """US Social Security Number (``NNN-NN-NNNN``).

    False-positive risk: any 9-digit dash-separated string. Vendor
    manuals may contain part numbers formatted as ``123-45-6789``,
    but those are rare in telecom PDFs (telecom uses alphanumeric
    part numbers like ``NTK-5542A``). The risk is acceptable for
    v1; a future v2 could narrow to "SSN-shaped + nearby keyword".
    """
    return _Pattern(
        pattern_id="ssn",
        severity=SEVERITY_MEDIUM,
        regex=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        redact=lambda m: _redact_keep_first_last(m, keep=0) or "***-**-****",
    )


def _credit_card() -> _Pattern:
    """Credit-card-shaped 13-19 digit number, Luhn-valid.

    Luhn check reduces false-positives on phone numbers, IDs, etc.
    Implemented as a post-filter inside :func:`_scan_with_pattern`.
    """
    # Capture the candidate digits; Luhn check happens in the scan
    # loop. The regex itself is permissive so the Luhn filter can
    # reject non-card-shaped matches.
    return _Pattern(
        pattern_id="credit_card",
        severity=SEVERITY_MEDIUM,
        # Word-boundary 13-19 digits, allowing optional spaces/dashes
        # between groups (e.g. "4111 1111 1111 1111" or
        # "4111-1111-1111-1111"). The capture group ``(\d[\d -]*)``
        # is consumed by the Luhn check.
        regex=re.compile(r"\b(\d[\d -]{11,22}\d)\b"),
        redact=lambda m: _redact_keep_first_last(re.sub(r"[ -]", "", m), keep=0),
    )


def _email() -> _Pattern:
    """Email address (RFC 5322 simplified).

    Logs only (low severity). Vendor manuals legitimately contain
    support emails.
    """
    return _Pattern(
        pattern_id="email",
        severity=SEVERITY_LOW,
        regex=re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
        redact=lambda m: _redact_email(m),
    )


def _redact_email(s: str) -> str:
    """Redact an email address keeping only the domain. E.g.
    ``support@vendor.com`` → ``****@vendor.com``."""
    if "@" not in s:
        return "*" * len(s)
    _, _, domain = s.partition("@")
    return f"****@{domain}"


def _us_phone() -> _Pattern:
    """US phone number (``NNN-NNN-NNNN`` or ``(NNN) NNN-NNNN``).

    Logs only (low severity). Vendor manuals contain support lines.
    """
    return _Pattern(
        pattern_id="us_phone",
        severity=SEVERITY_LOW,
        regex=re.compile(r"\b(?:\(\d{3}\)\s*)?\d{3}-\d{3}-\d{4}\b"),
        redact=lambda m: "***-***-****",
    )


# The order matters only for stable error messages when a single
# text triggers multiple patterns — first match wins per pattern.
# Patterns are independent; finding aggregation does not dedupe.
PATTERN_CATALOG: List[_Pattern] = [
    _aws_access_key(),
    _aws_secret_key(),
    _openai_key(),
    _anthropic_key(),
    _google_api_key(),
    _github_pat(),
    _langsmith_key(),
    _slack_token(),
    _stripe_live_key(),
    _ssn(),
    _credit_card(),
    _email(),
    _us_phone(),
]


# ---------------------------------------------------------------------------
# Luhn check for credit-card filter
# ---------------------------------------------------------------------------


def _luhn_valid(digits: str) -> bool:
    """Return True iff ``digits`` (a string of decimal digits) passes
    the Luhn checksum.

    Standard credit-card validation: starting from the rightmost
    digit, double every second digit; if doubling exceeds 9, subtract
    9. Sum all digits. Valid iff sum % 10 == 0.
    """
    digits_only = re.sub(r"\D", "", digits)
    if not digits_only or not (13 <= len(digits_only) <= 19):
        return False
    total = 0
    for i, ch in enumerate(reversed(digits_only)):
        n = int(ch)
        if i % 2 == 1:  # double every second digit from right
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Scan functions
# ---------------------------------------------------------------------------


def _scan_with_pattern(
    text: str,
    pat: _Pattern,
    *,
    page_number: int | None,
) -> List[Finding]:
    """Run a single pattern against ``text`` and return the
    findings.

    For the credit-card pattern, Luhn-validates each match and
    drops invalid ones (reduces false-positives on phone numbers,
    order IDs, etc.).

    For the aws_secret_key pattern, applies a class-diversity
    filter (at least one lowercase, one uppercase, one digit) to
    reduce false-positives on random base64-shaped blobs.
    """
    findings: List[Finding] = []
    for m in pat.regex.finditer(text):
        raw = m.group(0) if pat.pattern_id != "credit_card" else m.group(1)
        if pat.pattern_id == "credit_card" and not _luhn_valid(raw):
            continue
        if pat.pattern_id == "aws_secret_key" and not _has_class_diversity(raw):
            continue
        findings.append(
            Finding(
                pattern_id=pat.pattern_id,
                severity=pat.severity,
                match=raw,
                redacted=pat.redact(raw),
                page_number=page_number,
            )
        )
    return findings


def scan_text(text: str) -> List[Finding]:
    """Scan a single string for all known patterns. Returns an empty
    list when ``text`` is clean.

    ``page_number`` on the returned findings is always ``None`` —
    use :func:`scan_pages` when the text came from a parsed PDF and
    you want per-page attribution.
    """
    if not text:
        return []
    findings: List[Finding] = []
    for pat in PATTERN_CATALOG:
        findings.extend(_scan_with_pattern(text, pat, page_number=None))
    return findings


def scan_pages(pages: List[Dict[str, Any]]) -> List[Finding]:
    """Scan a list of page records (the output of
    :func:`telecom_rag.ingestion.docling_parser.parse_pdf`) and
    return aggregated findings with per-page attribution.

    Each page record is expected to have at least a ``"text"`` key
    and a ``"page_number"`` key. Missing keys are tolerated — the
    page is skipped silently rather than raising, so a malformed
    page record from a future refactor doesn't crash the whole scan.
    """
    findings: List[Finding] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        text = page.get("text", "") or ""
        page_number = page.get("page_number")
        if not text:
            continue
        for pat in PATTERN_CATALOG:
            findings.extend(
                _scan_with_pattern(text, pat, page_number=page_number)
            )
    return findings


def has_blocking_finding(findings: List[Finding]) -> bool:
    """Return True iff any finding's severity tier is blocking.

    Delegates to :func:`telecom_rag.security.policy.block_for_tier`
    — the canonical source of the "does this tier block?" decision
    (Issue #18 AC2). The previous inlined check
    (``severity in (SEVERITY_HIGH, SEVERITY_MEDIUM)``) had two
    hard-coded tier strings; the helper makes the contract
    single-source-of-truth and means a future tier change lands
    in one place, not two.

    The upload page uses this as the gate: ``if has_blocking_finding(
    scan_pages(...)): block_upload(...)``.

    ``low`` findings (email, phone) are NOT blocking.
    """
    return any(block_for_tier(f.severity) for f in findings)


def summarize_blocking_patterns(findings: List[Finding]) -> str:
    """Return a short, user-facing string summarizing the blocking
    pattern types found.

    Example output::

        "aws_access_key, openai_key (2 patterns)"

    Used by the upload page's per-file ``error`` cell.

    Uses :func:`block_for_tier` for the blocking-decision check
    (same single-source-of-truth rationale as
    :func:`has_blocking_finding`).
    """
    blocking = [f.pattern_id for f in findings if block_for_tier(f.severity)]
    # Dedupe while preserving first-seen order.
    seen: List[str] = []
    for pid in blocking:
        if pid not in seen:
            seen.append(pid)
    if not seen:
        return ""
    return f"{', '.join(seen)} ({len(seen)} pattern{'s' if len(seen) != 1 else ''})"