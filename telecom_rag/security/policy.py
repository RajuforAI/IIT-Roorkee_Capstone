"""Scanner policy module (Issue #18).

This module is the canonical source of the pre-upload scanner's
3-tier posture:

- ``high`` severity → blocks (confirmed secret formats)
- ``medium`` severity → blocks (strong PII; symmetric with high)
- ``low`` severity → logs only (heuristic; vendor manuals
  legitimately contain support contacts)

The constants are re-exported by :mod:`telecom_rag.security.scan`
for backward compatibility with the Issue #15 import surface
(``from telecom_rag.security.scan import SEVERITY_HIGH`` still
works). The helper :func:`block_for_tier` is the single source
of truth for the "does this tier block?" decision; the scanner's
:func:`has_blocking_finding` delegates to it.

Design choices
--------------

1. **No I/O, no imports from scan.py.** The dependency direction
   is one-way: ``scan.py`` imports from ``policy.py``, never the
   reverse. This avoids the circular-import trap if a future
   issue adds a caller that imports both modules.

2. **Unknown tier fails open.** :func:`block_for_tier` returns
   ``False`` for any tier not in the known set AND logs a
   WARNING. A future pattern added with a typo'd ``severity=``
   would otherwise be silently skipped by ``has_blocking_finding``,
   creating a coverage gap. The WARNING surfaces the regression
   in CloudWatch at upload time.

3. **Three tier constants, not an Enum.** The tiers are also
   used as plain strings in ``Finding.severity`` for JSON
   serialization (LangSmith, CloudWatch). An Enum would require
   ``.value`` coercion at every serialization site; a plain
   string is simpler and matches the existing
   ``Finding.to_dict()`` contract.

Bypass / override policy
------------------------

No override path exists. Per the user-ratified policy (2026-06-28):
a file with any ``high``- or ``medium``-severity finding is
hard-blocked with no UI acknowledgement, no admin allowlist, and
no ``I_ACKNOWLEDGE_PII`` checkbox. If a legitimate need arises,
it gets its own issue that re-debates the policy — see
``docs/security/policy.md``.

Why "fail open" for unknown tiers (not fail closed)
---------------------------------------------------

A NEW pattern added with a typo'd ``severity=`` is most likely
a regression introduced by a contributor who intended the
pattern to detect something. Failing closed (treating unknown
as "block") would hard-block every upload, breaking the upload
page entirely. Failing open (treating unknown as "log only")
preserves the page's functionality and surfaces the regression
via the WARNING log line. The catalog integrity test
``test_every_pattern_has_known_tier`` (Issue #18 AC6) catches
the typo at PR time, so the fail-open path is a defense-in-depth
guard, not the primary defense.
"""

from __future__ import annotations

import logging

# Module-level logger. Tests assert on this name so a future
# refactor that renames it is a deliberate breaking change.
_LOGGER = logging.getLogger("telecom_rag.security.policy")


# ---------------------------------------------------------------------------
# Tier constants
# ---------------------------------------------------------------------------


SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"


# Set of known tiers. Used by :func:`block_for_tier` to decide
# "is this tier recognized?". Exposed as a module constant so
# the catalog integrity test (Issue #18 AC6) can import the same
# canonical set rather than re-listing the tier strings.
KNOWN_TIERS: frozenset[str] = frozenset(
    {SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW}
)


# ---------------------------------------------------------------------------
# block_for_tier
# ---------------------------------------------------------------------------


def block_for_tier(tier: str) -> bool:
    """Return ``True`` iff a finding with ``tier`` severity should
    block the upload.

    The ratified 3-tier posture (Issue #18):

    - ``high``   → ``True``  (blocks)
    - ``medium`` → ``True``  (blocks)
    - ``low``    → ``False`` (logs only)
    - unknown   → ``False`` (fail-open) + WARNING log

    The fail-open path for unknown tiers is defense-in-depth.
    The catalog integrity test (``tests/test_pattern_catalog.py``)
    is the primary guard against a typo'd ``severity=`` landing
    in production; this function is the second line.
    """
    if tier == SEVERITY_HIGH:
        return True
    if tier == SEVERITY_MEDIUM:
        return True
    if tier == SEVERITY_LOW:
        return False
    _LOGGER.warning(
        "unknown scanner tier %r; treating as non-blocking "
        "(fail-open). This indicates a typo'd severity= in a "
        "pattern entry or a new tier added without updating "
        "policy.py.",
        tier,
    )
    return False