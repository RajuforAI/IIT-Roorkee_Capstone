"""CLI entry point for the RAGAS evaluation harness (Issue #9, AC4).

Usage::

    python -m telecom_rag.eval.ragas_eval --dataset <golden.jsonl> --output <report.json> \\
        [--collection <name>] [--top-k N] [--help]

Behavior
--------

1. Parses arguments via :mod:`argparse` (so ``--help`` exits 0 and
   prints the usage line).
2. Constructs a :class:`TelecomRAGEvaluator` with the supplied
   ``--collection`` and ``--top-k``.
3. Calls :meth:`TelecomRAGEvaluator.run` to produce the report.
4. Writes the report as pretty-printed JSON to ``--output``.
5. Prints the AC8 one-line summary to stdout, including the
   ``validation_agent_overlap`` value with the explicit ``0.50``
   threshold and interpretation.

Failure modes (exits non-zero)
------------------------------

- ``--dataset`` missing: argparse error (exit 2).
- ``--output`` missing: argparse error (exit 2).
- Dataset file missing or malformed: re-raised; the harness prints
  the traceback and exits 1.
- Empty dataset: :class:`DatasetSchemaError`; exits 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from telecom_rag.eval.evaluator import (
    METRIC_NAMES,
    VALIDATION_OVERLAP_THRESHOLD,
    EvaluationReport,
    TelecomRAGEvaluator,
)


def _format_overlap(overlap: Optional[float]) -> str:
    """Format the overlap value for the summary line.

    ``None`` becomes the string ``"null"`` (per AC8 spec). A finite
    float is formatted to two decimal places.
    """
    if overlap is None:
        return "null"
    return f"{overlap:.2f}"


def _format_metric(value: float) -> str:
    """Format a single aggregated metric to two decimal places."""
    return f"{float(value):.2f}"


def _build_summary(report: EvaluationReport) -> str:
    """Build the AC8 one-line summary.

    Format::

        RAGAS report: context_precision=X.XX context_recall=X.XX \
            faithfulness=X.XX answer_relevancy=X.XX | \
            validation_agent_overlap=X.XX \
            (threshold=0.50; <0.50 means validation_agent is NOT a useful proxy for RAGAS faithfulness — retune or retire)

    The pipe separates the four RAGAS aggregates from the
    validation_agent overlap; the parenthetical always includes the
    explicit 0.50 threshold + interpretation per AC8.
    """
    metrics = report.metrics
    parts = []
    for name in METRIC_NAMES:
        parts.append(f"{name}={_format_metric(metrics.get(name, 0.0))}")
    head = " ".join(parts)
    overlap_str = _format_overlap(report.validation_agent_overlap)
    tail = (
        f"validation_agent_overlap={overlap_str} "
        f"(threshold={VALIDATION_OVERLAP_THRESHOLD:.2f}; "
        f"<{VALIDATION_OVERLAP_THRESHOLD:.2f} means validation_agent "
        f"is NOT a useful proxy for RAGAS faithfulness — retune or retire)"
    )
    return f"RAGAS report: {head} | {tail}"


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser.

    ``--dataset`` and ``--output`` are required positional flags
    (kept as flags rather than positionals so the help text is
    readable; argparse accepts both styles).
    """
    parser = argparse.ArgumentParser(
        prog="python -m telecom_rag.eval.ragas_eval",
        description=(
            "Run the RAGAS-based evaluation harness (Issue #9). "
            "Reads a golden Q&A JSONL, runs each query through the "
            "retrieval + LLM stack, computes the four RAGAS "
            "metrics, correlates with the per-query "
            "ValidationAgent.grade_faithfulness score, and writes "
            "a JSON report."
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Path to the golden Q&A JSONL file (10+ records).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path where the JSON report will be written.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="telecom_docs",
        help=(
            "Chroma collection name to query (default: "
            "'telecom_docs')."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K chunks to retrieve per query (default: 5).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns the process exit code.

    Exit codes (Issue #10 AC2):
        0 — success, gate passed (overlap >= threshold OR overlap is
            null OR fewer than 2 non-refused pairs).
        1 — gate failed (overlap < threshold AND >= 2 non-refused pairs).
        1 — harness raised (any unexpected exception).
        1 — dataset file missing.
        2 — argparse error (bad CLI flags).

    The summary line ALWAYS prints before the gate decision so the
    operator can see the conductor's signal in CI logs even on a
    failed gate.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(
            f"ERROR: dataset file does not exist: {dataset_path}",
            file=sys.stderr,
        )
        return 1

    evaluator = TelecomRAGEvaluator(
        dataset_path=dataset_path,
        collection_name=args.collection,
        top_k=args.top_k,
    )

    try:
        report = evaluator.run()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: harness failed: {exc}", file=sys.stderr)
        return 1

    # Write the JSON report. ``model_dump(mode="json")`` ensures
    # Optional[float] -> null and datetimes -> ISO strings (we use
    # ISO strings already; the mode="json" flag is for forward compat
    # if a datetime sneaks in).
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # The AC8 one-line summary is the conductor's key signal; print
    # to stdout so it's easy to grep / pipe. ALWAYS print, regardless
    # of the gate decision (the operator needs to see WHY the gate
    # fired).
    summary = _build_summary(report)
    print(summary)

    # Issue #10 AC2: threshold-aware exit. The gate fires only when
    # the validation_agent overlap is a finite number below the AC8
    # threshold AND we have enough non-refused pairs to trust the
    # correlation (>= 2; otherwise per AC8 the conductor triages by
    # hand and overlap may be null).
    num_non_refused = sum(
        1 for q in report.per_query if not q.refused
    )
    overlap = report.validation_agent_overlap
    gate_failed = (
        overlap is not None
        and overlap < VALIDATION_OVERLAP_THRESHOLD
        and num_non_refused >= 2
    )
    if gate_failed:
        print(
            f"ERROR: RAGAS gate failed — validation_agent_overlap="
            f"{overlap:.2f} < threshold={VALIDATION_OVERLAP_THRESHOLD:.2f} "
            f"(num_non_refused={num_non_refused}). The validation_agent "
            f"is NOT a useful proxy for RAGAS faithfulness. "
            f"See Issue #10 AC2.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
