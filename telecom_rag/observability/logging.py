"""Structured JSON logging (Issue #13 AC3 / AC4 + Issue #22).

Exposes :func:`setup_json_logging` which configures the root logger
with a JSON-line formatter so every record emits as one
newline-delimited JSON object. This is the format CloudWatch Logs
Insights and ``awslogs``/``vector`` ship natively — no regex
parsing on the receiving end.

Two handlers are installed:

1. **StreamHandler** — emits to stdout. The dev / container path.
   Container logs are picked up by the orchestrator (ECS / docker
   compose) and shipped to CloudWatch via the standard out pipe.
2. **FileHandler** — emits to ``./logs/telecom_rag.jsonl`` (path
   override via ``TELECOM_RAG_LOG_FILE`` env var). Issue #22 reads
   this file from the ``/my_uploads`` page to render a per-user
   upload history. The directory is created if missing.

The JSON record always includes:
- ``timestamp`` — ISO-8601 UTC, parseable by
  :meth:`datetime.datetime.fromisoformat`
- ``level`` — record level name (``INFO``, ``WARNING``, ...)
- ``logger`` — the logger name (e.g. ``telecom_rag.llm``)
- ``message`` — the formatted log message

Any ``extra={...}`` fields passed at the log call site are merged
into the record as top-level keys. This is the seam that lets call
sites attach structured context (e.g. ``agent_node``, ``thread_id``,
``latency_ms``) without string-formatting.

The module is dependency-free (stdlib only) so it can be imported
at app startup without pulling in any heavyweight dependencies.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from telecom_rag.config import Settings

# Standard ``LogRecord`` attributes that we do NOT want to forward
# as JSON fields when they happen to be missing. Anything else set
# on the record (typically via ``extra=...``) IS forwarded.
_STD_LOGRECORD_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "asctime",
        "message",
        "taskName",
    }
)

# Default location for the file handler. Relative paths resolve
# against the process's CWD, matching pydantic-settings semantics
# for the rest of the config. Overridable via env var for
# deployments that mount /var/log at a different path.
_DEFAULT_LOG_FILE = "./logs/telecom_rag.jsonl"


class JsonFormatter(logging.Formatter):
    """Logging formatter that emits each record as one JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        # Use timezone-aware UTC ISO-8601 (CloudWatch Logs Insights
        # expects ISO-8601 and prefers UTC for cross-region queries).
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload: Dict[str, Any] = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Forward any extra=... fields as top-level keys.
        for key, value in record.__dict__.items():
            if key in _STD_LOGRECORD_ATTRS or key in payload:
                continue
            # JSON cannot encode arbitrary objects — coerce via repr()
            # as a last resort so the log line is still valid JSON.
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        # Attach exception info as a structured field (not as text).
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def setup_json_logging(level: str = "INFO") -> None:
    """Configure the root logger with a JSON-line formatter.

    Idempotent: calling this twice does not add a duplicate handler.
    Replaces any existing handlers on the root logger so the test
    runner (which itself uses plain logging) cannot leak records
    into the JSON output stream.

    Parameters
    ----------
    level:
        Root-logger level (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    root = logging.getLogger()
    formatter = JsonFormatter()

    # Remove any handlers installed by a prior call (or by the test
    # runner) so the JSON formatter is the only one writing.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)

    # Issue #22: also write to a JSONL file so the ``/my_uploads``
    # page can read the upload history without a CloudWatch query.
    # The file path is ``./logs/telecom_rag.jsonl`` by default; the
    # ``TELECOM_RAG_LOG_FILE`` env var overrides (a test can point
    # it at a tmp_path fixture so concurrent test runs don't race).
    # The field is read via ``Settings()`` (Issue #34 single-source-
    # of-truth contract); a fresh instance per call so test-side
    # ``monkeypatch.setenv`` is honored.
    log_file = Settings().log_file
    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        root.addHandler(file_handler)
    except OSError:
        # Filesystem is read-only (a CI sandbox or a misconfigured
        # container) — fall back to stdout-only. The /my_uploads
        # page will surface an empty history with a friendly
        # banner; production must mount a writable volume at
        # /logs.
        pass

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
