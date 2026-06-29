"""Read the JSONL upload audit log for the ``/my_uploads`` page.

The upload page emits two structured events via
:func:`telecom_rag.observability.logging.setup_json_logging`:

* ``event=upload_indexed`` — emitted once per successfully indexed
  file, with ``user``, ``file_name``, ``chunks``, ``file_sha256``,
  ``batch_id``, and ``timestamp``.
* ``event=upload_blocked`` — emitted once per file the pre-upload
  PII / secret scanner rejected, with ``user``, ``file_name``,
  ``blocked_patterns``, ``file_sha256``, and ``timestamp``.

This module reads the JSONL file (default
``./logs/telecom_rag.jsonl``; overridable via
``TELECOM_RAG_LOG_FILE``) and surfaces the records as a list of
plain dicts the ``/my_uploads`` page can render in a DataFrame.

Why a plain JSONL reader, not a SQLite mirror
---------------------------------------------

A SQLite mirror would give us indexed queries, but the audit log
is an append-only stream — the page only ever reads the most
recent N records and filters in Python. A SQLite mirror would
require a write-path on every log emission (added latency to the
hot ingest path) and a separate schema. JSONL is the canonical
"CloudWatch Logs Insights" shape, and ``/my_uploads`` mirrors that
shape directly. The CloudWatch log group is the durable source of
truth for SOC review; this file is the local convenience cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from telecom_rag.config import Settings

__all__ = [
    "default_log_path",
    "read_upload_events",
]


def default_log_path() -> Path:
    """Return the absolute path to the JSONL upload log file.

    Reads ``Settings().log_file`` (the canonical single-source-of-
    truth knob, prefixed Issue #34). The ``env_prefix="TELECOM_RAG_"``
    loader resolves the value from shell env OR the repo's ``.env``
    file — both flows, no silent miss.

    Relative paths resolve against the process's CWD, matching
    pydantic-settings semantics for the rest of the config. The
    fallback ``./logs/telecom_rag.jsonl`` is set as a class default
    on ``Settings.log_file`` (see ``telecom_rag/config.py``).

    Note: we instantiate a FRESH ``Settings()`` per call rather than
    using the module-level singleton, so tests using
    ``monkeypatch.setenv("TELECOM_RAG_LOG_FILE", tmp_path)`` are
    honored (the module singleton is captured at import time and is
    immune to test-side setenv). The instantiation is microsecond-
    cheap.
    """
    raw = Settings().log_file
    return Path(raw).resolve()


def read_upload_events(
    *,
    user: Optional[str] = None,
    event_type: Optional[str] = None,
    log_path: Optional[Path] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """Read ``upload_indexed`` / ``upload_blocked`` records from the JSONL log.

    Parameters
    ----------
    user:
        When set, only records whose ``user`` field matches are
        returned. Case-sensitive exact match (the auth gate always
        stores usernames in their canonical lowercase form, so a
        case-sensitive compare is correct).
    event_type:
        When set, only records with that ``event`` value are
        returned (``"upload_indexed"`` or ``"upload_blocked"``).
    log_path:
        Override the default log file path (for tests with a
        tmp_path fixture).
    limit:
        Maximum number of records to return. Reads the LAST
        ``limit`` records from the file (most recent uploads are
        what the user cares about on the page). Default 500 — a
        generous bound for a single Streamlit session.

    Returns
    -------
    list[dict]
        Each dict matches the JSON shape emitted by the upload
        page's ``event=upload_*`` log lines. Sorted by ``timestamp``
        DESCENDING (newest first).
    """
    target = log_path or default_log_path()
    if not target.exists():
        return []

    try:
        with target.open("r", encoding="utf-8") as fh:
            # Read all lines, keep only those that parse AND match
            # the filter. ``tail = lines[-limit:]`` keeps the most
            # recent ``limit`` records before filtering (more
            # efficient than filtering the entire history when
            # the file grows large).
            lines = fh.readlines()
    except OSError:
        # File was rotated / deleted between the exists() check
        # and the open() call. Treat as empty.
        return []

    tail = lines[-limit * 2 :]  # over-read to absorb filtered-out lines
    records: List[Dict[str, Any]] = []
    for raw in tail:
        try:
            obj = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            # A malformed line (truncated write, partial flush) —
            # skip rather than raise. The audit log is an
            # append-only stream; a bad line shouldn't crash the
            # page.
            continue
        event = obj.get("event")
        if event not in ("upload_indexed", "upload_blocked"):
            continue
        if event_type and event != event_type:
            continue
        if user and obj.get("user") != user:
            continue
        records.append(obj)

    # Sort newest-first. The timestamp is ISO-8601 UTC, which sorts
    # lexically the same way it sorts chronologically.
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return records[:limit]