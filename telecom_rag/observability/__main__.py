"""CLI: python -m telecom_rag.observability.cost report

Renders the in-process cost ledger as a human-readable table.
Exit code 0 when the ledger has records; exit code 2 when empty.

Usage:
    python -m telecom_rag.observability.cost report
    python -m telecom_rag.observability.cost report --help

The CLI reads the same singleton ledger the admin page renders.
For durable per-call history, query the JSON log stream (CloudWatch
Logs Insights; ``event=cost_record``).
"""
from __future__ import annotations

import argparse
import sys

from telecom_rag.observability.cost import get_ledger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m telecom_rag.observability.cost",
        description="Cost / quota telemetry report (Issue #16).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("report", help="Print today's cost / quota report.")
    return parser


def _render_report() -> int:
    """Print a human-readable cost report. Returns process exit code.

    Issue #19: also renders a per-agent breakdown so the CLI
    surfaces the new ``by_agent`` dimension alongside the existing
    per-provider/model rollup.
    """
    ledger = get_ledger()
    total_calls = ledger.total_calls()
    if total_calls == 0:
        print("Cost ledger is empty — no LLM calls recorded in this process.", file=sys.stderr)
        return 2

    snap = ledger.snapshot()
    daily_total = ledger.daily_total_usd()

    print("Cost / quota report (in-process ledger)")
    print("=" * 60)
    print(f"Today's total cost:   ${daily_total:.4f}")
    print(f"Total calls:          {total_calls}")
    print(f"Total tokens:         {ledger.total_tokens():,}")
    print()

    # Per-provider / model breakdown. Issue #19: skip the ``by_agent``
    # sibling key — it's rendered separately below.
    rows = [
        (key, info)
        for key, info in snap.items()
        if key != "by_agent"
    ]
    rows.sort(key=lambda kv: kv[1].get("cost_usd", 0.0), reverse=True)
    print("Per-provider / model breakdown (sorted by cost, descending):")
    print("-" * 60)
    print(f"{'provider/model':<40} {'calls':>8} {'tokens':>12} {'cost_usd':>12}")
    print("-" * 60)
    for key, info in rows:
        print(
            f"{key:<40} {info['calls']:>8} {info['total_tokens']:>12,} "
            f"${info['cost_usd']:>10.6f}"
        )
    print("-" * 60)

    # Issue #19: per-agent breakdown.
    by_agent = snap.get("by_agent", {})
    if by_agent:
        print()
        print("Per-agent breakdown (sorted by cost, descending):")
        print("-" * 60)
        print(f"{'agent':<25} {'calls':>8} {'tokens':>12} {'cost_usd':>12}")
        print("-" * 60)
        agent_rows = sorted(
            by_agent.items(),
            key=lambda kv: kv[1].get("cost_usd", 0.0),
            reverse=True,
        )
        for agent_key, info in agent_rows:
            print(
                f"{agent_key:<25} {info['calls']:>8} {info['total_tokens']:>12,} "
                f"${info['cost_usd']:>10.6f}"
            )
        print("-" * 60)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "report":
        return _render_report()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())