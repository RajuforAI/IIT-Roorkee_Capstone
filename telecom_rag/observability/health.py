"""Health-check endpoint (Issue #13 AC7 / AC8).

Exposes a tiny HTTP handler that returns JSON describing the
health of three subsystems:

* ``chroma`` — can we reach the Chroma collection on disk?
* ``postgres_checkpoint`` — is the production checkpointer
  reachable (or ``"skipped"`` when ``TELECOM_RAG_CHECKPOINT_DSN``
  is unset, the dev/test default)?
* ``langsmith_configured`` — is the LangSmith tracing env wired
  consistently (tracing on + key set, or tracing off + no key)?

The handler returns HTTP 200 when every check passes (or returns
``"skipped"``, which is the documented dev-mode state) and HTTP
503 when any check fails with an ``"error: ..."`` string.

Mounting
--------
This module ships a :func:`build_app` factory that returns a small
Starlette app with a single ``/_healthz`` route. The app can be
mounted two ways:

1. **As a Starlette sidecar** — run alongside the Streamlit
   server on a different port (e.g. 8502). AWS ALBs hit the sidecar.
2. **Embedded in Streamlit** via ``st.experimental_set_query_params``
   on a dedicated page (less common; the sidecar is the
   recommended path for production).

The tests drive the Starlette app via
``fastapi.testclient.TestClient`` (httpx under the hood).
"""

from __future__ import annotations

from typing import Any, Dict

import chromadb

try:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
except ImportError:  # pragma: no cover
    Starlette = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]
    Response = None  # type: ignore[assignment]

from telecom_rag.config import settings


# ---------------------------------------------------------------------------
# Individual checks — pure functions, easy to stub in tests
# ---------------------------------------------------------------------------


def check_chroma() -> str:
    """Check Chroma collection reachability.

    Returns ``"ok"`` on success, ``"error: <msg>"`` on failure.

    Uses the canonical ``chromadb.PersistentClient(path=...)``
    constructor — the same shape as
    :func:`telecom_rag.ingestion.embedder.get_or_create_collection` —
    so the healthz probe reads from the exact persist directory the
    ingest path writes to. The ``chromadb`` import is at module top
    so tests can monkeypatch the constructor via
    ``monkeypatch.setattr(health_module, "chromadb", ...)``.
    """
    try:
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        # A lightweight reachability probe — list collections.
        client.list_collections()
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def check_postgres_checkpoint() -> str:
    """Check the production Postgres checkpointer.

    Returns:
    - ``"skipped"`` when ``TELECOM_RAG_CHECKPOINT_DSN`` is unset
      (the dev/test default — SqliteSaver is in use).
    - ``"ok"`` when the Postgres saver can be instantiated.
    - ``"error: <msg>"`` on failure.
    """
    dsn = getattr(settings, "checkpoint_dsn", None)
    if not dsn:
        # Dev / test mode — SqliteSaver is in use, not an error.
        return "skipped"
    try:
        from telecom_rag.memory.checkpointer import (
            get_postgres_checkpointer,
        )

        cp = get_postgres_checkpointer(dsn, auto_setup=False)
        try:
            cp.close()
        except Exception:  # pragma: no cover — close is best-effort
            pass
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def check_langsmith_configured() -> bool:
    """Return ``True`` when LangSmith tracing is consistently
    configured.

    "Consistently" means either (tracing off AND no key) or
    (tracing on AND key set). The inverse — tracing on but key
    missing — returns ``False`` and is the silent-failure case
    Issue #13 fixes (see :mod:`telecom_rag.observability.langsmith`
    for the full surface).
    """
    tracing_on = bool(settings.langchain_tracing_v2)
    key_set = bool(settings.langchain_api_key)
    if tracing_on and not key_set:
        return False
    # tracing_off + key_set is a "key stashed but unused" state — not
    # an error, just informational. Counts as configured.
    return True


# ---------------------------------------------------------------------------
# Starlette handler + app factory
# ---------------------------------------------------------------------------


HEALTH_PATH = "/_healthz"


def _build_response(checks: Dict[str, Any]) -> Response:
    """Compose the JSON response + status code from the check dict.

    Status is 200 when every check is ``"ok"`` / ``"skipped"`` /
    ``True`` and 503 otherwise.
    """
    chroma_ok = checks["chroma"] == "ok"
    pg_state = checks["postgres_checkpoint"]
    pg_ok = pg_state in ("ok", "skipped")
    ls_ok = bool(checks["langsmith_configured"])

    all_ok = chroma_ok and pg_ok and ls_ok
    body = {
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
    }
    return JSONResponse(
        body, status_code=200 if all_ok else 503
    )


async def healthz_handler(request: Request) -> Response:  # noqa: ARG001
    """Starlette route handler for ``/_healthz``."""
    checks: Dict[str, Any] = {
        "chroma": check_chroma(),
        "postgres_checkpoint": check_postgres_checkpoint(),
        "langsmith_configured": check_langsmith_configured(),
    }
    return _build_response(checks)


def build_app() -> Any:
    """Build a Starlette app with the ``/_healthz`` route mounted.

    Returns the app object so tests can drive it with
    ``fastapi.testclient.TestClient`` and operators can serve it
    with ``uvicorn telecom_rag.observability.health:app``.
    """
    if Starlette is None:  # pragma: no cover
        raise RuntimeError(
            "starlette is not installed; install it via "
            "`pip install starlette` to enable the /_healthz endpoint."
        )
    app = Starlette()
    app.add_route(HEALTH_PATH, healthz_handler, methods=["GET"])
    return app


# Module-level app for `uvicorn telecom_rag.observability.health:app`.
app = build_app() if Starlette is not None else None  # type: ignore[misc]
