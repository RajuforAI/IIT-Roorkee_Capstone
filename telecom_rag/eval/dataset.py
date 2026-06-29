"""Golden Q&A dataset loader (Issue #9, AC2).

Public surface
--------------

- :class:`GoldenQARecord` — dataclass with the five required fields
  (``query``, ``expected_answer``, ``expected_source_files``,
  ``expected_relevant_chunk_ids``, ``doc_category``).
- :class:`DatasetSchemaError` — raised on any malformed row.
- :func:`load_dataset` — read a JSONL file from disk and return a
  list of :class:`GoldenQARecord` instances.

Schema enforcement
------------------

Each JSONL row MUST contain exactly these five keys, with these
exact types (no coercion — wrong types are errors, not silent
defaults)::

    {
        "query":                       str,
        "expected_answer":             str,
        "expected_source_files":       List[str],
        "expected_relevant_chunk_ids": List[int],
        "doc_category":                str,
    }

Extra keys are tolerated (forward-compat) but the five above are
required. The file MUST contain at least one record; an empty file
raises :class:`DatasetSchemaError` (the harness cannot run a report
against zero queries).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union


# Required schema for every row. Order is the canonical display order
# in error messages; membership is the contract.
_REQUIRED_KEYS: tuple[str, ...] = (
    "query",
    "expected_answer",
    "expected_source_files",
    "expected_relevant_chunk_ids",
    "doc_category",
)


class DatasetSchemaError(ValueError):
    """Raised when a JSONL row fails schema validation or the file is empty."""


@dataclass(frozen=True)
class GoldenQARecord:
    """One row from ``tests/fixtures/golden_qa.jsonl``.

    Frozen so callers can't mutate records after loading (the harness
    only reads them).
    """

    query: str
    expected_answer: str
    expected_source_files: List[str]
    expected_relevant_chunk_ids: List[int]
    doc_category: str


def _validate_row(row: Dict[str, Any], *, line_no: int) -> GoldenQARecord:
    """Return a :class:`GoldenQARecord` for ``row`` or raise.

    Raises :class:`DatasetSchemaError` on missing keys, wrong types,
    or empty required string fields. The line number is included in
    every error so a bad row in a 1000-line JSONL is findable.
    """
    if not isinstance(row, dict):
        raise DatasetSchemaError(
            f"line {line_no}: expected a JSON object, got {type(row).__name__}"
        )

    missing = [k for k in _REQUIRED_KEYS if k not in row]
    if missing:
        raise DatasetSchemaError(
            f"line {line_no}: missing required keys: {missing!r}; "
            f"required keys are {list(_REQUIRED_KEYS)!r}"
        )

    # Type checks — no coercion. ``isinstance(x, bool)`` is a subclass
    # trap for ints; reject booleans on the int-typed fields.
    query = row["query"]
    if not isinstance(query, str):
        raise DatasetSchemaError(
            f"line {line_no}: 'query' must be str, got {type(query).__name__}"
        )

    expected_answer = row["expected_answer"]
    if not isinstance(expected_answer, str):
        raise DatasetSchemaError(
            f"line {line_no}: 'expected_answer' must be str, "
            f"got {type(expected_answer).__name__}"
        )

    expected_source_files = row["expected_source_files"]
    if (
        not isinstance(expected_source_files, list)
        or not all(isinstance(s, str) for s in expected_source_files)
    ):
        raise DatasetSchemaError(
            f"line {line_no}: 'expected_source_files' must be List[str], "
            f"got {type(expected_source_files).__name__}"
        )

    expected_relevant_chunk_ids = row["expected_relevant_chunk_ids"]
    if not isinstance(expected_relevant_chunk_ids, list) or not all(
        isinstance(i, int) and not isinstance(i, bool)
        for i in expected_relevant_chunk_ids
    ):
        raise DatasetSchemaError(
            f"line {line_no}: 'expected_relevant_chunk_ids' must be List[int], "
            f"got {type(expected_relevant_chunk_ids).__name__}"
        )

    doc_category = row["doc_category"]
    if not isinstance(doc_category, str):
        raise DatasetSchemaError(
            f"line {line_no}: 'doc_category' must be str, "
            f"got {type(doc_category).__name__}"
        )

    return GoldenQARecord(
        query=query,
        expected_answer=expected_answer,
        expected_source_files=list(expected_source_files),
        expected_relevant_chunk_ids=list(expected_relevant_chunk_ids),
        doc_category=doc_category,
    )


def load_dataset(path: Union[str, Path]) -> List[GoldenQARecord]:
    """Load a JSONL golden Q&A dataset from ``path``.

    Each non-empty line is parsed as one JSON object and validated
    against the schema. Raises :class:`DatasetSchemaError` on any
    schema violation or if the file contains zero records.

    The empty-file check is intentional: a JSONL with no records is
    almost always an authoring mistake, and the harness cannot
    produce a meaningful report from zero queries.
    """
    p = Path(path)
    if not p.exists():
        raise DatasetSchemaError(f"dataset file does not exist: {p}")

    records: List[GoldenQARecord] = []
    with p.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                # Tolerate trailing newlines / blank lines.
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetSchemaError(
                    f"line {line_no}: invalid JSON: {exc}"
                ) from exc
            records.append(_validate_row(row, line_no=line_no))

    if not records:
        raise DatasetSchemaError(
            f"dataset file {p} contains zero records; "
            f"the harness requires at least one golden Q&A row"
        )

    return records
