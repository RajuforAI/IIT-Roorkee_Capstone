"""SQLite connection + schema for the auth DB (Issue #22, AC1).

The auth DB is a separate SQLite file from
``telecom_rag/memory/checkpointer.py``'s ``checkpoints.db``. The two
serve different lifetimes (auth credentials survive the chat session;
checkpoints survive the Streamlit process), and a corruption in one
should not cascade into the other.

Configuration
-------------

The DB path is read from the ``TELECOM_RAG_AUTH_DB`` env var via
``Settings().auth_db`` (the canonical single-source-of-truth knob,
prefixed Issue #34), falling back to ``./auth.db`` (the repo root).
Tests override the env var with a ``tmp_path`` fixture so two tests
cannot collide on the file lock.

Schema
------

A single ``users`` table:

    id              INTEGER PK AUTOINCREMENT
    username        TEXT UNIQUE NOT NULL
    password_hash   TEXT NOT NULL            -- bcrypt cost 12
    email           TEXT                     -- optional
    display_name    TEXT                     -- optional
    role            TEXT NOT NULL CHECK(role IN ('admin','user'))
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    last_login_at   TEXT NULL                -- ISO-8601 UTC

WAL mode
--------

We open the connection with ``PRAGMA journal_mode=WAL`` so concurrent
readers (Streamlit worker threads, login widget, admin panel) don't
serialize behind a single writer. SQLite WAL allows N readers + 1
writer concurrently — fits our use case (admin CRUD is rare; login is
read-heavy). Mirrors the connection pattern at
``telecom_rag/memory/checkpointer.py:110-113``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

__all__ = ["auth_db_path", "get_connection", "init_schema"]


# bcrypt cost factor — pinned at 12 to match the pre-Issue #22 inline
# value at ``app/main.py:103``. Bumping the cost factor invalidates all
# existing hashes (users would need to re-set their passwords), which
# is a deliberate operational decision, not a code change.
_BCRYPT_ROUNDS = 12


def auth_db_path() -> Path:
    """Return the absolute path to the auth SQLite file.

    Reads ``Settings().auth_db`` — the canonical single-source-of-
    truth knob (prefixed Issue #34). The ``env_prefix="TELECOM_RAG_"``
    loader resolves the value from shell env OR the repo's ``.env``
    file — both flows, no silent miss.

    Relative paths resolve against the process's CWD, matching
    pydantic-settings semantics for the rest of the config. The
    fallback ``./auth.db`` is set as a class default on
    ``Settings.auth_db`` (see ``telecom_rag/config.py``); the value
    here is whatever ``Settings()`` reports, with a hard guard
    against ``None`` for safety.

    Note: we instantiate a FRESH ``Settings()`` per call rather than
    using the module-level singleton, so that tests using
    ``monkeypatch.setenv("TELECOM_RAG_AUTH_DB", tmp_path)`` are
    honored (the module singleton is captured at import time and is
    immune to test-side setenv). The instantiation is microsecond-
    cheap.
    """
    from telecom_rag.config import Settings

    raw = Settings().auth_db
    if raw:
        return Path(raw).resolve()
    return Path("./auth.db").resolve()


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a SQLite connection to the auth DB.

    Sets ``check_same_thread=False`` so a connection opened on the
    Streamlit worker thread can be used from a callback thread (e.g.,
    the auth widget's session-state read). Mirrors
    ``telecom_rag/memory/checkpointer.py:107-113``.

    Sets ``row_factory=sqlite3.Row`` so callers can access columns by
    name (``row["username"]``) — the CRUD layer wraps this in dicts
    for stability across sqlite3 versions.

    Enables WAL journal mode so concurrent readers do not block on a
    single writer (admin CRUD vs. login widget). The PRAGMA is a
    no-op for ``:memory:`` connections and returns silently when the
    filesystem doesn't support WAL (e.g., some FUSE mounts).
    """
    target = Path(db_path) if db_path is not None else auth_db_path()
    # ``:memory:`` is a sqlite special-case that bypasses the file
    # system. Skip the parent-dir create for it (there is no parent).
    if str(target) != ":memory:":
        target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL is the recommended journal mode for any workload with
    # concurrent readers + occasional writers. Best-effort — fall
    # back to the default journal mode if WAL is unavailable.
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError:
        pass
    return conn


# Schema DDL — single source of truth for AC1.
_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    email           TEXT,
    display_name    TEXT,
    role            TEXT NOT NULL CHECK (role IN ('admin','user')),
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login_at   TEXT
)
"""


def init_schema(db_path: Optional[Path] = None) -> None:
    """Create the ``users`` table if it does not already exist.

    Idempotent: calling ``init_schema()`` repeatedly is safe and
    carries no data-loss risk (``CREATE TABLE IF NOT EXISTS`` is a
    no-op when the table exists). ``bootstrap.ensure_bootstrap_admin``
    relies on this idempotency to avoid seeding the bootstrap admin on
    every app start.
    """
    conn = get_connection(db_path)
    try:
        conn.execute(_SCHEMA_DDL)
        conn.commit()
    finally:
        conn.close()
