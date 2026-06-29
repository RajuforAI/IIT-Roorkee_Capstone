"""Cost / quota telemetry for LLM calls (Issue #16).

Pure-data module: no provider SDK imports, no network calls, no
external dependencies beyond the stdlib. The whole module can be
imported in any test environment without touching the LLM providers.

Public surface:
- :class:`TokenUsage` / :class:`EmbeddingUsage` — typed token-count carriers
- :class:`PriceEntry` — per-model USD pricing (input + output per 1K tokens)
- :data:`DEFAULT_PRICING` — bundled catalog snapshot (USD per 1K tokens)
- :func:`price_for` — catalog lookup with `TELECOM_RAG_LLM_PRICER` pluggable seam
- :func:`compute_cost` — pure function, no side effects
- :func:`load_pricing_overrides` — reads `TELECOM_RAG_LLM_PRICING_JSON` env var
- :class:`CostRecord` / :class:`CostLedger` — in-process ledger with thread lock
- :func:`record` — single entry point: updates ledger + emits JSON log line
- :func:`get_ledger` / :func:`reset_ledger` — singleton accessor for tests
- :func:`setup_cost_tracking` — idempotent no-op marker hook (future use)

Currency: USD only. No FX conversion. The catalog is the source of truth.

Why float (not Decimal) for cost:
    Float64 has ~15 significant decimal digits. At our resolution
    (prices quoted to 1e-5 USD per 1K tokens, totals expected <$100/day
    in normal operation, <$10K/day even under runaway conditions),
    float accumulation error is bounded to ~1e-10 USD. Decimal would
    force a custom JSON encoder (stdlib ``json`` doesn't encode
    ``Decimal``) and would complicate the ``record()`` log payload.

The `TELECOM_RAG_LLM_PRICER` env var is a one-line drop-in for a future
LiteLLM swap. When production sets ``TELECOM_RAG_LLM_PRICER=module.attr``,
:func:`price_for` delegates to that callable instead of the bundled
catalog. No further code changes needed.
"""
from __future__ import annotations

import importlib
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from telecom_rag.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent enum (Issue #19)
# ---------------------------------------------------------------------------


class Agent(str, Enum):
    """Canonical agent names for per-agent cost attribution.

    Each member's ``.value`` is its lowercase name — the same string
    that flows into the ``event=cost_record`` JSON log payload so a
    CloudWatch Logs Insights ``stats ... by agent`` query works
    without a separate name-mapping.

    Why a string-valued Enum (not plain strings):
        A typo'd string ``record(..., agent="validate")`` would
        silently create an orphan dashboard dimension that no call
        site ever produces again. The Enum makes that a TypeError at
        the call site (caught by tests in CI). See Issue #19
        "Why enum (not string)" rationale.

    Why six values:
        ROUTER + RETRIEVAL + VALIDATION + SUMMARIZATION match the
        four agents in the LangGraph state machine
        (:mod:`telecom_rag.agents`). EMBEDDING covers the
        Chroma-ingest embedding calls. UNKNOWN is the sentinel
        reserved for a future caller that legitimately cannot
        attribute to one of the named agents (none today — every
        call site must declare its agent, so UNKNOWN is rarely
        used in practice; it exists for symmetry and to make
        ``Optional[Agent]`` paths explicit).
    """

    ROUTER = "router"
    RETRIEVAL = "retrieval"
    SUMMARIZATION = "summarization"
    VALIDATION = "validation"
    EMBEDDING = "embedding"
    UNKNOWN = "unknown"


# Backward-compat alias (Issue #18 pattern: ``block_for_tier(tier)``
# pinned a function, so we follow the same shape here).
_agent_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Typed token counts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for a chat completion.

    ``prompt_tokens`` = input tokens, ``completion_tokens`` = output
    tokens, ``total_tokens`` = their sum. The provider SDK always
    reports all three (OpenAI ``resp.usage`` and Gemini
    ``resp.usage_metadata`` both populate this shape).
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class EmbeddingUsage:
    """Token counts for an embedding call (input-only)."""

    input_tokens: int


# Type alias: a usage record is one of the two dataclasses above.
Usage = Union[TokenUsage, EmbeddingUsage]


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriceEntry:
    """Per-model USD pricing (per 1K tokens).

    Embeddings are symmetric: ``input_per_1k_usd == output_per_1k_usd``.
    Chat models usually have asymmetric pricing (output is more
    expensive than input).
    """

    input_per_1k_usd: float
    output_per_1k_usd: float


# Bundled catalog — source of truth for development / test.
# Production should override via TELECOM_RAG_LLM_PRICING_JSON.
# Updated 2026-06-27 (verify against vendor pricing pages monthly).
DEFAULT_PRICING: Dict[str, PriceEntry] = {
    # OpenAI
    "openai/gpt-4o-mini": PriceEntry(input_per_1k_usd=0.00015, output_per_1k_usd=0.00060),
    "openai/gpt-4o": PriceEntry(input_per_1k_usd=0.00250, output_per_1k_usd=0.01000),
    "openai/text-embedding-3-small": PriceEntry(
        input_per_1k_usd=0.00002, output_per_1k_usd=0.00002
    ),
    # Gemini
    "gemini/gemini-2.5-flash": PriceEntry(input_per_1k_usd=0.00030, output_per_1k_usd=0.00250),
    "gemini/gemini-2.0-flash": PriceEntry(input_per_1k_usd=0.00010, output_per_1k_usd=0.00040),
    "gemini/text-embedding-004": PriceEntry(
        input_per_1k_usd=0.00001, output_per_1k_usd=0.00001
    ),
}


def _catalog_key(model: str, provider: str) -> str:
    """Combine model + provider into the catalog key shape ``provider/model``."""
    return f"{provider}/{model}"


def load_pricing_overrides() -> Dict[str, PriceEntry]:
    """Read ``TELECOM_RAG_LLM_PRICING_JSON`` env var and parse as JSON.

    Returns an empty dict if the env var is unset. Raises
    :class:`ValueError` if the env var is set but malformed JSON.

    Expected shape: ``{"provider/model": {"input_per_1k": 0.5,
    "output_per_1k": 1.5}, ...}``. Both fields are in USD per 1K
    tokens.
    """
    # Issue #34 — read via ``Settings().llm_pricing_json`` (canonical
    # single-source-of-truth). The pre-Issue #34 name was
    # ``TELECOM_RAG_LLM_PRICING_JSON``; renamed to
    # ``TELECOM_RAG_LLM_PRICING_JSON`` to match the project env-prefix
    # convention. Tests that previously used ``setenv("TELECOM_RAG_LLM_...")``
    # now use ``setenv("TELECOM_RAG_LLM_...")`` (see test_cost.py).
    raw = (Settings().llm_pricing_json or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"TELECOM_RAG_LLM_PRICING_JSON is not valid JSON: {exc.msg} "
            f"(at line {exc.lineno}, col {exc.colno})"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"TELECOM_RAG_LLM_PRICING_JSON must be a JSON object, got {type(data).__name__}"
        )
    out: Dict[str, PriceEntry] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            raise ValueError(
                f"TELECOM_RAG_LLM_PRICING_JSON['{key}'] must be an object, "
                f"got {type(value).__name__}"
            )
        try:
            out[key] = PriceEntry(
                input_per_1k_usd=float(value["input_per_1k"]),
                output_per_1k_usd=float(value["output_per_1k"]),
            )
        except KeyError as exc:
            raise ValueError(
                f"TELECOM_RAG_LLM_PRICING_JSON['{key}'] missing required key {exc.args[0]!r}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"TELECOM_RAG_LLM_PRICING_JSON['{key}'] has non-numeric price: {exc}"
            ) from exc
    return out


# Pluggable pricer seam: TELECOM_RAG_LLM_PRICER=module.attr swaps the
# default catalog for a user-supplied callable. Resolved lazily on
# first call so a misconfigured env var surfaces only when cost
# tracking is exercised (not at import time).
_PricerFn = Callable[[str, str], Optional[PriceEntry]]
_pricer_cache: Optional[_PricerFn] = None
_pricer_resolved: bool = False
_pricer_lock = threading.Lock()


def _resolve_pricer() -> _PricerFn:
    """Resolve the pricer callable once and cache it.

    Reads ``TELECOM_RAG_LLM_PRICER`` env var. When unset, returns the
    bundled catalog lookup. When set, imports ``module.attr`` and
    uses that callable. The callable signature is
    ``(model: str, provider: str) -> PriceEntry | None``.
    """
    global _pricer_cache, _pricer_resolved
    if _pricer_resolved:
        return _pricer_cache  # type: ignore[return-value]

    # Issue #34 — see ``load_pricing_overrides`` for the rename note.
    raw = (Settings().llm_pricer or "").strip()
    if not raw:
        _pricer_cache = _default_pricer
    else:
        module_path, sep, attr = raw.rpartition(".")
        if not sep or not module_path or not attr:
            raise ValueError(
                f"TELECOM_RAG_LLM_PRICER must be 'module.attr' (e.g. "
                f"'litellm.cost_calculator'), got {raw!r}"
            )
        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:
            raise ValueError(
                f"TELECOM_RAG_LLM_PRICER module {module_path!r} is not importable: {exc}"
            ) from exc
        pricer = getattr(mod, attr, None)
        if pricer is None:
            raise ValueError(
                f"TELECOM_RAG_LLM_PRICER target {raw!r} not found on module {module_path!r}"
            )
        if not callable(pricer):
            raise ValueError(
                f"TELECOM_RAG_LLM_PRICER target {raw!r} on module {module_path!r} "
                f"is not callable (got {type(pricer).__name__})"
            )
        _pricer_cache = pricer

    _pricer_resolved = True
    return _pricer_cache


def _reset_pricer_cache() -> None:
    """Reset the cached pricer callable. Test-only."""
    global _pricer_cache, _pricer_resolved
    with _pricer_lock:
        _pricer_cache = None
        _pricer_resolved = False


def _default_pricer(model: str, provider: str) -> Optional[PriceEntry]:
    """Bundled catalog lookup: returns ``PriceEntry`` or ``None``."""
    overrides = load_pricing_overrides()
    key = _catalog_key(model, provider)
    if key in overrides:
        return overrides[key]
    return DEFAULT_PRICING.get(key)


def price_for(model: str, provider: str) -> Optional[PriceEntry]:
    """Look up pricing for a (model, provider) pair.

    Returns ``None`` when the model is not in the catalog (caller
    decides whether to log / fall back to a default price). The
    bundled catalog is augmented by ``TELECOM_RAG_LLM_PRICING_JSON``
    overrides. When ``TELECOM_RAG_LLM_PRICER`` is set, the entire
    catalog lookup is delegated to the user-supplied callable.
    """
    pricer = _resolve_pricer()
    return pricer(model, provider)


def compute_cost(usage: Usage, price: PriceEntry) -> float:
    """Compute USD cost for a single call.

    Pure function: no side effects, no environment reads, no I/O.
    Caller supplies both the usage and the price — that contract
    makes the function trivially testable and reusable from any
    context (including the future LiteLLM pricer).
    """
    if isinstance(usage, TokenUsage):
        if usage.prompt_tokens < 0 or usage.completion_tokens < 0:
            raise ValueError(
                f"negative token count: prompt={usage.prompt_tokens} "
                f"completion={usage.completion_tokens}"
            )
        return (
            usage.prompt_tokens / 1000.0 * price.input_per_1k_usd
            + usage.completion_tokens / 1000.0 * price.output_per_1k_usd
        )
    if isinstance(usage, EmbeddingUsage):
        if usage.input_tokens < 0:
            raise ValueError(f"Negative embedding token count: {usage.input_tokens}")
        # EmbeddingUsage is input-only — both sides of the PriceEntry
        # are equal so this multiplies correctly either way.
        return usage.input_tokens / 1000.0 * price.input_per_1k_usd
    raise TypeError(f"Unsupported usage type: {type(usage).__name__}")


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostRecord:
    """A single cost record emitted by :func:`record`."""

    timestamp: datetime
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    # Issue #19: per-agent attribution. Required at construction —
    # a missed attribution surfaces as a ``TypeError`` at the call
    # site, never as a silent "unknown" dashboard cell.
    agent: "Agent"


class CostLedger:
    """Process-global cost ledger keyed by ``(provider, model, date_utc)``.

    Thread-safe: a single ``threading.Lock`` guards every read and
    write. The lock is held briefly (microseconds) — just long
    enough to update a dict or read a snapshot.

    Singleton: use :func:`get_ledger` to obtain the process-global
    instance. Tests call :func:`reset_ledger` between tests for
    isolation.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Keyed by (provider, model). Value is a list of CostRecord
        # for that pair. Records are append-only — we never mutate
        # or remove.
        self._records: Dict[tuple[str, str], List[CostRecord]] = {}

    def add(self, record: CostRecord) -> None:
        with self._lock:
            self._records.setdefault((record.provider, record.model), []).append(record)

    def records(self) -> List[CostRecord]:
        """Return a flat list of all records (snapshot under lock)."""
        with self._lock:
            out: List[CostRecord] = []
            for recs in self._records.values():
                out.extend(recs)
            return out

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Per-(provider/model) aggregation for the admin page / CLI.

        Returns a dict keyed by ``provider/model``. Each value is::

            {"calls": int, "total_tokens": int, "cost_usd": float}

        ``total_tokens`` is the sum of every record's ``total_tokens``
        (prompt + completion combined). For pure-embedding calls,
        every token is an input token; we still report the same number
        under ``total_tokens`` for consistency.

        Issue #19: an additional ``"by_agent"`` key holds the same
        shape keyed by :class:`Agent`. Both projections are produced
        in a single pass over the records so the snapshot stays
        O(n).
        """
        with self._lock:
            by_pm: Dict[str, Dict[str, Any]] = {}
            by_agent: Dict["Agent", Dict[str, Any]] = {}
            for (provider, model), recs in self._records.items():
                key = f"{provider}/{model}"
                calls = len(recs)
                total_tokens = sum(r.total_tokens for r in recs)
                cost_usd = sum(r.cost_usd for r in recs)
                by_pm[key] = {
                    "calls": calls,
                    "total_tokens": total_tokens,
                    "cost_usd": cost_usd,
                }
                for r in recs:
                    agg = by_agent.setdefault(
                        r.agent,
                        {"calls": 0, "total_tokens": 0, "cost_usd": 0.0},
                    )
                    agg["calls"] += 1
                    agg["total_tokens"] += r.total_tokens
                    agg["cost_usd"] += r.cost_usd
            out = dict(by_pm)
            out["by_agent"] = by_agent
            return out

    def daily_total_usd(self, today: Optional[datetime] = None) -> float:
        """Sum of today's cost records in USD.

        ``today`` defaults to ``datetime.now(timezone.utc).date()``.
        Comparison is on the UTC date so a midnight-UTC rollover is
        the only sensible boundary (matches the JSON log timestamps
        which are also UTC).
        """
        if today is None:
            today = datetime.now(timezone.utc)
        today_date = today.date()
        with self._lock:
            return sum(
                r.cost_usd
                for recs in self._records.values()
                for r in recs
                if r.timestamp.date() == today_date
            )

    def daily_total_usd_by_agent(
        self, agent: "Agent", today: Optional[datetime] = None
    ) -> float:
        """Sum of today's cost for a single agent (Issue #19 AC5).

        Filters ``daily_total_usd`` by agent so the per-agent
        CloudWatch widget, admin-page section, and quota warning
        can read independently of other agents.
        """
        if today is None:
            today = datetime.now(timezone.utc)
        today_date = today.date()
        with self._lock:
            return sum(
                r.cost_usd
                for recs in self._records.values()
                for r in recs
                if r.agent == agent and r.timestamp.date() == today_date
            )

    def total_calls(self) -> int:
        with self._lock:
            return sum(len(recs) for recs in self._records.values())

    def total_tokens(self) -> int:
        with self._lock:
            return sum(r.total_tokens for recs in self._records.values() for r in recs)


# ---------------------------------------------------------------------------
# Singleton + reset (for tests)
# ---------------------------------------------------------------------------


_ledger_instance: Optional[CostLedger] = None
_ledger_lock = threading.Lock()


def get_ledger() -> CostLedger:
    """Return the process-global :class:`CostLedger` singleton."""
    global _ledger_instance
    if _ledger_instance is None:
        with _ledger_lock:
            if _ledger_instance is None:
                _ledger_instance = CostLedger()
    return _ledger_instance


def reset_ledger() -> None:
    """Reset the singleton ledger. Test-only — do not call in production."""
    global _ledger_instance
    with _ledger_lock:
        _ledger_instance = None


# ---------------------------------------------------------------------------
# Warning dedup
# ---------------------------------------------------------------------------


_quota_warning_fired_date: Optional[Any] = None
_quota_warning_lock = threading.Lock()


def daily_quota_warning_fired() -> bool:
    """Return whether the daily quota warning has already fired for today."""
    with _quota_warning_lock:
        return _quota_warning_fired_date is not None


def reset_quota_warning() -> None:
    """Reset the dedup state. Test-only."""
    global _quota_warning_fired_date
    with _quota_warning_lock:
        _quota_warning_fired_date = None


# ---------------------------------------------------------------------------
# Per-agent quota warning dedup (Issue #19)
# ---------------------------------------------------------------------------


_by_agent_quota_warning_fired: Dict["Agent", Any] = {}
_by_agent_quota_warning_lock = threading.Lock()


def _reset_by_agent_quota_warning() -> None:
    """Reset the per-agent quota-warning dedup state. Test-only."""
    global _by_agent_quota_warning_fired
    with _by_agent_quota_warning_lock:
        _by_agent_quota_warning_fired = {}


def _maybe_fire_by_agent_quota_warning(agent: "Agent", today_date: Any) -> None:
    """Fire a per-agent daily quota WARNING at most once per UTC day per agent.

    Reads ``TELECOM_RAG_COST_QUOTA_USD_DAILY_BY_AGENT`` env var (default
    $10). When the agent's cumulative daily cost exceeds the
    threshold AND no warning has fired today for this agent, emit a
    structured WARNING log line with ``event=cost_quota_warning_by_agent``.

    Dedup is keyed by agent — the warning fires independently per
    agent so a runaway VALIDATION loop never suppresses a RETRIEVAL
    warning.
    """
    global _by_agent_quota_warning_fired
    with _by_agent_quota_warning_lock:
        if _by_agent_quota_warning_fired.get(agent) == today_date:
            return

    # Issue #34 — see ``load_pricing_overrides`` for the rename note.
    raw_quota = Settings().cost_quota_usd_daily_by_agent
    if raw_quota is None:
        raw_quota = ""  # treat None same as unset for the float-parse path
    raw_quota = str(raw_quota).strip()
    try:
        quota_usd = float(raw_quota) if raw_quota else 10.0
    except ValueError:
        logger.warning(
            "Invalid TELECOM_RAG_COST_QUOTA_USD_DAILY_BY_AGENT=%r — using default $10",
            raw_quota,
        )
        quota_usd = 10.0

    agent_total = get_ledger().daily_total_usd_by_agent(agent)
    if agent_total > quota_usd:
        with _by_agent_quota_warning_lock:
            if _by_agent_quota_warning_fired.get(agent) != today_date:
                _by_agent_quota_warning_fired[agent] = today_date
                logger.warning(
                    "Daily LLM cost quota exceeded for agent=%s: $%.4f > $%.4f",
                    agent.value,
                    agent_total,
                    quota_usd,
                    extra={
                        "event": "cost_quota_warning_by_agent",
                        "agent": agent.value,
                        "daily_total_usd_by_agent": agent_total,
                        "quota_usd": quota_usd,
                    },
                )


def _maybe_fire_quota_warning(today_date: Any) -> None:
    """Fire the daily quota WARNING log at most once per UTC day.

    Reads ``TELECOM_RAG_COST_QUOTA_USD_DAILY`` env var (default $10). When
    cumulative daily cost exceeds the threshold AND the warning
    hasn't already fired today, emit a structured WARNING log line
    with ``event=cost_quota_warning``.
    """
    global _quota_warning_fired_date
    with _quota_warning_lock:
        if _quota_warning_fired_date == today_date:
            return

    # Issue #34 — see ``load_pricing_overrides`` for the rename note.
    raw_quota = Settings().cost_quota_usd_daily
    if raw_quota is None:
        raw_quota = ""
    raw_quota = str(raw_quota).strip()
    try:
        quota_usd = float(raw_quota) if raw_quota else 10.0
    except ValueError:
        logger.warning(
            "Invalid TELECOM_RAG_COST_QUOTA_USD_DAILY=%r — using default $10",
            raw_quota,
        )
        quota_usd = 10.0

    daily_total = get_ledger().daily_total_usd()
    if daily_total > quota_usd:
        with _quota_warning_lock:
            if _quota_warning_fired_date != today_date:
                _quota_warning_fired_date = today_date
                logger.warning(
                    "Daily LLM cost quota exceeded: $%.4f > $%.4f",
                    daily_total,
                    quota_usd,
                    extra={
                        "event": "cost_quota_warning",
                        "daily_total_usd": daily_total,
                        "quota_usd": quota_usd,
                    },
                )


# ---------------------------------------------------------------------------
# Missing-model warning dedup
# ---------------------------------------------------------------------------


_missing_model_warned: set[str] = set()
_missing_model_lock = threading.Lock()


def _warn_missing_model_once(model: str, provider: str) -> None:
    """Emit a WARNING log line for a missing catalog entry, at most once per model."""
    key = f"{provider}/{model}"
    with _missing_model_lock:
        if key in _missing_model_warned:
            return
        _missing_model_warned.add(key)
    logger.warning(
        "No pricing entry for model %r (provider=%s); cost will be $0",
        model,
        provider,
        extra={
            "event": "cost_pricing_missing",
            "model": model,
            "provider": provider,
        },
    )


def reset_pricing_warnings() -> None:
    """Reset the missing-model dedup set. Test-only."""
    global _missing_model_warned
    with _missing_model_lock:
        _missing_model_warned = set()


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------


def record(
    usage: Usage,
    model: str,
    provider: str,
    *,
    latency_ms: float,
    agent: "Agent",
) -> CostRecord:
    """Record a single LLM call: update ledger + emit JSON log line.

    The single entry point for cost tracking. Called from
    :func:`telecom_rag.llm.chat_with_fallback` and
    :func:`telecom_rag.llm.embed_with_fallback` on every successful
    dispatch.

    Issue #19: ``agent`` is a REQUIRED keyword-only argument.
    Missed attribution surfaces as a ``TypeError`` at the call
    site — never as a silent "unknown" dashboard cell.

    Always emits a structured INFO log record with
    ``event=cost_record`` (carrying the agent). Fires the global
    daily quota WARNING when the cumulative daily cost exceeds
    ``TELECOM_RAG_COST_QUOTA_USD_DAILY`` (at most once per UTC day)
    AND the per-agent daily quota WARNING when the agent's daily
    cost exceeds ``TELECOM_RAG_COST_QUOTA_USD_DAILY_BY_AGENT`` (at most
    once per UTC day per agent).

    When the model is not in the catalog, the call is still recorded
    with ``cost_usd=0.0`` and a WARNING fires once per missing model.
    """
    # Issue #19: validate ``agent`` is an :class:`Agent` enum member.
    # A plain string (typo'd dimension name) would silently orphan a
    # dashboard cell; the Enum makes it a loud TypeError.
    if not isinstance(agent, Agent):
        raise TypeError(
            f"record() agent must be a telecom_rag.observability.cost.Agent "
            f"member, got {type(agent).__name__}: {agent!r}"
        )

    # Look up the price (may be None for unknown models).
    price = price_for(model, provider)
    if price is None:
        _warn_missing_model_once(model, provider)
        cost_usd = 0.0
    else:
        cost_usd = compute_cost(usage, price)

    # Normalize usage counts onto the CostRecord shape.
    if isinstance(usage, TokenUsage):
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
    else:
        prompt_tokens = usage.input_tokens
        completion_tokens = 0
        total_tokens = usage.input_tokens

    now_utc = datetime.now(timezone.utc)
    rec = CostRecord(
        timestamp=now_utc,
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        agent=agent,
    )

    get_ledger().add(rec)

    logger.info(
        "LLM call: %s/%s agent=%s tokens=%d cost=$%.6f latency=%.1fms",
        provider,
        model,
        agent.value,
        total_tokens,
        cost_usd,
        latency_ms,
        extra={
            "event": "cost_record",
            "provider": provider,
            "model": model,
            "agent": agent.value,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
        },
    )

    today_date = now_utc.date()
    _maybe_fire_quota_warning(today_date)
    _maybe_fire_by_agent_quota_warning(agent, today_date)

    return rec


# ---------------------------------------------------------------------------
# setup_cost_tracking
# ---------------------------------------------------------------------------


_setup_done: bool = False
_setup_lock = threading.Lock()


def setup_cost_tracking() -> None:
    """Idempotent setup hook for cost tracking.

    Currently a no-op marker — the cost module is process-global
    state and needs no explicit setup beyond ``import``. Provided
    as a stable hook so future cross-cutting setup (e.g. CloudWatch
    metric registration, LangSmith cost events) has a single,
    idempotent entry point that mirrors :func:`setup_json_logging`.

    Idempotency is enforced by a module-level flag so two callers
    in the same process see the no-op behavior.
    """
    global _setup_done
    with _setup_lock:
        if _setup_done:
            return
        _setup_done = True


__all__ = [
    # Types
    "TokenUsage",
    "EmbeddingUsage",
    "Usage",
    "PriceEntry",
    "CostRecord",
    "CostLedger",
    # Agent enum (Issue #19)
    "Agent",
    # Catalog
    "DEFAULT_PRICING",
    "load_pricing_overrides",
    "price_for",
    "compute_cost",
    # Record + ledger
    "record",
    "get_ledger",
    "reset_ledger",
    # Quota warning
    "daily_quota_warning_fired",
    "reset_quota_warning",
    # Per-agent quota warning (Issue #19)
    "_reset_by_agent_quota_warning",
    # Setup
    "setup_cost_tracking",
    # Test helpers
    "reset_pricing_warnings",
    "_reset_pricer_cache",
]


# ---------------------------------------------------------------------------
# CLI entry point (``python -m telecom_rag.observability.cost report``)
# ---------------------------------------------------------------------------
#
# Defined at module bottom (not via the parent ``__main__.py``) so the
# invocation ``python -m telecom_rag.observability.cost report`` resolves
# the submodule's ``__main__`` block — the parent package's
# ``__main__.py`` would only fire for ``python -m telecom_rag.observability``.
#
# The actual implementation lives in ``telecom_rag.observability.__main__``
# so the same render logic can be unit-tested without spawning a
# subprocess. We import + delegate here so ``python -m`` finds it.


if __name__ == "__main__":  # pragma: no cover -- exercised via the CLI test
    from telecom_rag.observability.__main__ import main as _cli_main

    raise SystemExit(_cli_main())
