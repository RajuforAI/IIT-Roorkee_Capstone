"""LangGraph checkpointer factories (Issue #7 + Issue #11).

Public surface
--------------

- :func:`get_checkpointer` — returns a :class:`SqliteSaver` configured
  against the given SQLite connection string. Caller owns the
  connection lifetime (call ``.close()`` when done).
- :func:`get_postgres_checkpointer` — returns a :class:`PostgresSaver`
  for production multi-worker / multi-process deployments. Same
  lifetime story as the SQLite factory: caller calls ``.close()``
  on app shutdown.

Why a plain function (not a context manager)
--------------------------------------------

In LangGraph 1.x, ``SqliteSaver.from_conn_string(...)`` and
``PostgresSaver.from_conn_string(...)`` are both ``@contextmanager``
that close their underlying connections on exit. The chat page builds
the graph **once** per Streamlit session (cached via
``@st.cache_resource``) and the graph needs the connection to stay
open for the whole session — not just the block in which the graph
is built. A context-manager factory would tie connection lifetime to
a ``with`` block that exits before the graph is used, causing
``sqlite3.ProgrammingError: Cannot operate on a closed database.``
(or ``psycopg.InterfaceError: connection already closed`` for the
Postgres path) on every later ``graph.invoke(...)`` call.

Returning the saver directly and exposing ``.close()`` lets the chat
page build the graph once and close the connection on app shutdown,
while tests can still scope lifetime with a ``try/finally`` block.
Production file-backed checkpoints survive Streamlit reruns because
the connection (and hence the SQLite file lock) stays open for the
whole process.

Usage — production (chat page, SQLite)::

    from telecom_rag.memory.checkpointer import get_checkpointer

    cp = get_checkpointer("./checkpoints.db")
    try:
        graph = build_graph(collection=vs, checkpointer=cp)
        # ... use the graph for the lifetime of the Streamlit session
    finally:
        cp.close()  # on app shutdown

Usage — production (chat page, Postgres)::

    from telecom_rag.config import Settings
    from telecom_rag.memory.checkpointer import get_postgres_checkpointer

    cp = get_postgres_checkpointer(Settings().checkpoint_dsn)
    try:
        graph = build_graph(collection=vs, checkpointer=cp)
        # ... multi-worker safe: every worker reads/writes the same DSN
    finally:
        cp.close()  # on app shutdown

Usage — test::

    def test_x():
        cp = get_checkpointer(":memory:")
        try:
            graph = build_graph(collection=stub_vs, checkpointer=cp)
            # ... assertions against the in-memory graph state
        finally:
            cp.close()
"""

from __future__ import annotations

import sqlite3
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

# ``langgraph-checkpoint-postgres`` is an optional dep — only the
# Postgres factory imports it.  We import the symbol lazily inside
# ``get_postgres_checkpointer`` so a Sqlite-only deployment does not
# need the Postgres package installed.

__all__ = ["get_checkpointer", "get_postgres_checkpointer"]


def get_checkpointer(conn_string: str) -> SqliteSaver:
    """Return a :class:`SqliteSaver` bound to ``conn_string``.

    Parameters
    ----------
    conn_string:
        A SQLite connection string. Use ``":memory:"`` for tests;
        use a file path (e.g. ``"./checkpoints.db"``) for the chat
        page so state persists across Streamlit reruns.

    Returns
    -------
    SqliteSaver
        A checkpointer ready to pass to ``StateGraph.compile(...)``.
        The returned saver holds an open SQLite connection; call
        ``.close()`` when done to release the file lock (Windows)
        and free the connection.
    """
    # We open the connection directly (rather than using
    # SqliteSaver.from_conn_string, which is a @contextmanager that
    # closes the connection on exit).  The connection's
    # ``check_same_thread=False`` matches from_conn_string's default
    # so LangGraph's threading model is preserved; the chat page
    # runs in a single Streamlit worker thread per session, so
    # sharing across threads is theoretical here.
    conn = sqlite3.connect(
        conn_string,
        check_same_thread=False,
    )
    saver = SqliteSaver(conn)

    # SqliteSaver does not expose a public ``close()`` (its parent
    # ``BaseCheckpointSaver`` has no close method either — cleanup
    # was originally via ``SqliteSaver.from_conn_string``'s context
    # manager).  Attach one here so callers can release the SQLite
    # connection (and the Windows file lock on a file-backed DB)
    # via the natural ``saver.close()`` API.  Idempotent: a second
    # ``close()`` call is a no-op.
    closed_flag = {"closed": False}

    def _close() -> None:
        if closed_flag["closed"]:
            return
        closed_flag["closed"] = True
        try:
            conn.close()
        except Exception:
            pass

    saver.close = _close  # type: ignore[attr-defined]
    return saver


# ``PostgresSaver`` is imported lazily so the Sqlite-only path stays
# dependency-free.  The tests patch this symbol on the module object
# via ``monkeypatch.setattr(checkpointer_module, "PostgresSaver",
# stub_class)`` — that's why the symbol lives at module scope
# (rather than inside the function) and is referenced via the
# module namespace below.
PostgresSaver: Any = None  # set on first factory call (or test stub)

# ``psycopg`` is imported lazily inside ``get_postgres_checkpointer``;
# the tests patch it on the module via ``monkeypatch.setattr`` so the
# stub ``connect`` function is used.  We declare the symbol at module
# scope as ``None`` so ``monkeypatch.setattr`` has a real attribute
# to patch (it errors on missing attributes).
psycopg: Any = None  # set on first factory call (or test stub)


def get_postgres_checkpointer(
    conn_string: str,
    *,
    auto_setup: bool = True,
) -> Any:
    """Return a :class:`PostgresSaver` bound to ``conn_string``.

    Parameters
    ----------
    conn_string:
        A Postgres connection string (e.g.
        ``"postgresql://user:pass@host:5432/db"``).
    auto_setup:
        When ``True`` (the default), call ``.setup()`` on the
        returned saver to create the four langgraph tables
        (``checkpoints``, ``checkpoint_blobs``, ``checkpoint_writes``,
        ``checkpoint_migrations``) before returning.  Set
        ``auto_setup=False`` if you manage the schema externally
        (e.g. via a migration step in CI).

    Returns
    -------
    PostgresSaver
        A checkpointer ready to pass to ``StateGraph.compile(...)``.
        The returned saver holds an open psycopg connection; call
        ``.close()`` when done to release it.  The attached
        ``close()`` is idempotent — a second call is a no-op.

    Notes
    -----
    ``PostgresSaver.from_conn_string`` is a ``@contextmanager`` that
    closes the connection on exit.  Like the SQLite factory, we open
    the connection directly so the chat page can hold the saver for
    the lifetime of the Streamlit session.  The psycopg connection
    is opened with ``autocommit=True`` because ``setup()`` issues
    DDL (CREATE TABLE IF NOT EXISTS) and the langgraph docs
    require autocommit for the schema bootstrap to persist.
    """
    # Lazily import so Sqlite-only deployments don't need
    # langgraph-checkpoint-postgres installed.
    global PostgresSaver, psycopg
    if PostgresSaver is None:
        from langgraph.checkpoint.postgres import PostgresSaver as _PgSaver
        PostgresSaver = _PgSaver

    # Lazy import for the same reason — psycopg is a transitive
    # dependency of langgraph-checkpoint-postgres but we keep the
    # factory's import surface narrow.
    if psycopg is None:
        import psycopg as _psycopg
        psycopg = _psycopg

    conn = psycopg.connect(conn_string, autocommit=True)
    saver = PostgresSaver(conn)

    if auto_setup:
        # setup() creates the four tables langgraph needs.  Idempotent
        # (CREATE TABLE IF NOT EXISTS) so a redundant call is safe; we
        # gate it on the auto_setup flag rather than tracking per-saver
        # state because the cost is negligible and the langgraph docs
        # recommend calling setup() explicitly on first use.
        saver.setup()

    # PostgresSaver does not expose a public ``close()`` (its parent
    # ``BaseCheckpointSaver`` has no close method).  Attach an
    # idempotent one mirroring the SQLite factory's lifetime story.
    # The psycopg connection's ``.close()`` is itself idempotent
    # (a second call is a no-op on a closed connection), but we
    # still gate via the closed_flag so the wrapper is consistent
    # across both factories.
    closed_flag = {"closed": False}

    def _close() -> None:
        if closed_flag["closed"]:
            return
        closed_flag["closed"] = True
        try:
            conn.close()
        except Exception:
            pass

    saver.close = _close  # type: ignore[attr-defined]
    return saver