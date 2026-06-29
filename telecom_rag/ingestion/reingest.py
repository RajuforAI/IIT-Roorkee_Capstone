"""CLI entry point for Issue #20: ``python -m telecom_rag.ingestion.reingest``.

By default (no ``--apply`` flag), the CLI computes a diff against the
current Chroma collection and prints a human-readable summary, then
exits 0 without writing. Pass ``--apply`` to execute the diff
(delete stale chunks, re-embed new/changed files).

Exit codes:
    0  — success (dry-run or apply with no per-file failures)
    1  — apply mode, one or more files failed to ingest (the batch
         continues; only the failing files are recorded in the diff
         and the in-process state)
    2  — configuration error (e.g. missing --source-dir)

The CLI is intentionally thin: it parses args, calls
:func:`telecom_rag.ingestion.pipeline.reingest_directory`, and prints
the diff. All the structured-event emission lives in the pipeline
module so other call sites (admin page, future scheduled jobs) share
the same code path.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, Sequence


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m telecom_rag.ingestion.reingest",
        description=(
            "Idempotent re-ingest with diff preview. "
            "Default mode is dry-run (no writes); pass --apply to execute."
        ),
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Apply the diff: delete stale chunks, embed new/changed "
            "files. Without this flag the CLI exits 0 after printing "
            "the diff and writes nothing."
        ),
    )
    p.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing the PDFs to re-ingest.",
    )
    p.add_argument(
        "--persist-dir",
        required=True,
        help="Chroma persist directory (same path used by ingest_directory).",
    )
    p.add_argument(
        "--collection-name",
        default="telecom_docs",
        help="Chroma collection name (default: telecom_docs).",
    )
    p.add_argument(
        "--allowlist",
        default=None,
        help=(
            "Optional path to a text file with one basename per line; "
            "files not in the list are skipped (mirrors ingest_directory's "
            "file_allowlist kwarg)."
        ),
    )
    return p


def _parse_allowlist(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _print_summary(diff: dict, apply: bool) -> None:
    totals = diff.get("totals", {})
    print("=" * 60)
    if apply:
        print("RE-INGEST — APPLY MODE")
    else:
        print("RE-INGEST — DRY-RUN (no writes)")
    print("=" * 60)
    print(f"NEW:       {totals.get('new', 0):>3} files "
          f"(~{totals.get('new_chunks_est', 0)} chunks est.)")
    print(f"CHANGED:   {totals.get('changed', 0):>3} files")
    print(f"STALE:     {totals.get('stale', 0):>3} files "
          f"(would delete ~{totals.get('stale_chunks', 0)} chunks)")
    print(f"UNCHANGED: {totals.get('unchanged', 0):>3} files (skipped)")
    print()

    def _row(label: str, entries: List[dict]) -> None:
        for e in entries:
            line = f"  {label}: {e['file']}"
            extra = []
            if "est_chunks" in e:
                extra.append(f"~{e['est_chunks']} chunks")
            if "old_chunks" in e:
                extra.append(f"old={e['old_chunks']}")
            if "chunks" in e:
                extra.append(f"{e['chunks']} chunks")
            if extra:
                line += "  (" + ", ".join(extra) + ")"
            print(line)

    _row("NEW", diff.get("new", []))
    _row("CHANGED", diff.get("changed", []))
    _row("STALE", diff.get("stale", []))
    _row("UNCHANGED", diff.get("unchanged", []))

    if not apply:
        print()
        print("Re-run with --apply to execute this diff.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # Surface structured events on stdout at INFO level so operators
    # running this in CI logs can grep them.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Late import so the CLI doesn't pull the pipeline module on --help.
    from telecom_rag.ingestion.pipeline import reingest_directory

    allowlist = _parse_allowlist(args.allowlist)

    try:
        diff = reingest_directory(
            dir_path=args.source_dir,
            persist_dir=args.persist_dir,
            apply=args.apply,
            collection_name=args.collection_name,
            file_allowlist=allowlist,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    _print_summary(diff, apply=args.apply)

    if args.apply:
        # Exit 1 if any per-file failure was recorded. The diff dict's
        # ``totals.failed`` counter is incremented when the diff loop
        # couldn't parse a file (empty/corrupt PDF) or when a per-file
        # ingest/delete failed during the apply path.
        failures = diff.get("totals", {}).get("failed", 0)
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
