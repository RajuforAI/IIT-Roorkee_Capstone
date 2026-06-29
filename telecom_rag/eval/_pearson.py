"""Pearson correlation helper for the RAGAS harness (Issue #9, AC8).

Why a vendored copy
-------------------

We could have used :func:`numpy.corrcoef` or :func:`scipy.stats.pearsonr`
but both require a third-party import and both return ``nan`` on
degenerate inputs (constant X, constant Y, fewer than 2 points).
The contract spec for this issue explicitly requires ``None`` on
degenerate inputs (so the JSON report serializes as ``null`` rather
than ``NaN``, which is illegal JSON), and a ``ValueError`` on
mismatched-length inputs (so an off-by-one in the harness is visible
at test time rather than producing silent garbage correlations).

This is ~30 lines of stdlib math; vendoring it keeps the harness
zero-dep beyond ``numpy`` (which RAGAS already pulls in).

Boundaries (per AC6)
--------------------

- perfect positive / negative: ±1.0
- constant X or Y, or fewer than 2 points: ``None``
- empty inputs: ``None``
- mismatched lengths: ``ValueError``
"""

from __future__ import annotations

from typing import Optional, Sequence


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Return the Pearson correlation coefficient of ``xs`` and ``ys``.

    Returns ``None`` (not NaN, not 0.0) when:

    - ``xs`` and ``ys`` are empty.
    - ``len(xs) != len(ys)`` -- raises :class:`ValueError` instead.
    - ``len(xs) < 2`` (a single point has no defined correlation).
    - ``xs`` is constant (zero variance in X).
    - ``ys`` is constant (zero variance in Y).

    Raises :class:`ValueError` when the lengths differ.

    Otherwise returns the standard Pearson r in ``[-1.0, 1.0]``.
    """
    if len(xs) != len(ys):
        raise ValueError(
            f"pearson: length mismatch xs={len(xs)} ys={len(ys)}"
        )

    n = len(xs)
    if n < 2:
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    # Use a single pass with Kahan-style accumulation? No — for the
    # small N we care about (a single evaluation run has tens, not
    # millions, of rows) plain summation is fine. The variance /
    # covariance terms are robust to FP drift at this scale.
    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        dy = y - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    # Degenerate: any zero variance means the correlation is
    # undefined. Return None so the harness can serialize it as JSON
    # ``null`` (instead of NaN, which is illegal in strict JSON).
    if var_x == 0.0 or var_y == 0.0:
        return None

    return cov / (var_x * var_y) ** 0.5
