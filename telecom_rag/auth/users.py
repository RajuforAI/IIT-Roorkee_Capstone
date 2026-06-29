"""User CRUD for the auth DB (Issue #22, AC3).

Public surface
--------------

- :func:`list_users` — return every row, sorted by username.
- :func:`get_user` — return a single row by username, or ``None``.
- :func:`create_user` — insert a new row; rejects duplicate
  usernames and unknown role values.
- :func:`delete_user` — remove a row by username; idempotent
  (deleting a missing username does not raise).
- :func:`update_last_login` — stamp ``last_login_at`` with the
  current UTC ISO-8601 timestamp.

All functions use **parameterized queries** so SQL-injection payloads
in user input are stored as literal strings (pinned by
``test_auth_users_db.py::test_create_user_rejects_sql_injection_payload``).

Connection lifetime
-------------------

Each function opens and closes its own connection. SQLite's
``check_same_thread=False`` (set in :func:`db.get_connection`) lets
Streamlit's worker-thread + callback-thread pattern coexist; the
cost of opening per call is negligible (~1ms on a warm filesystem)
and avoids the lifetime-management complexity of a long-lived
connection pool.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from telecom_rag.auth.db import get_connection, init_schema
from telecom_rag.auth.passwords import hash_password

__all__ = [
    "create_user",
    "delete_user",
    "get_user",
    "list_users",
    "update_last_login",
]

_VALID_ROLES = ("admin", "user")

# Username format: lowercase letters, digits, underscore; 3-32 chars.
# Tight enough to fit a sane admin panel UX (no spaces, no unicode)
# and to make log/audit queries predictable.
_USERNAME_RE = re.compile(r"^[a-z0-9_]{3,32}$")


def _row_to_dict(row: Any) -> Dict[str, Any]:
    """Convert a :class:`sqlite3.Row` to a plain dict.

    Returning a ``dict`` (rather than a ``sqlite3.Row``) decouples the
    public API from the underlying DB library — a future migration to
    Postgres or a different driver would otherwise break every caller.
    """
    return {key: row[key] for key in row.keys()}


def _validate_role(role: Any) -> str:
    """Return ``role`` if it's a known role string; raise ``ValueError`` otherwise."""
    if not isinstance(role, str) or role not in _VALID_ROLES:
        raise ValueError(
            f"role must be one of {_VALID_ROLES!r}, got {role!r}"
        )
    return role


def _validate_username(username: Any) -> str:
    """Return ``username`` if it matches the canonical username pattern; raise otherwise."""
    if not isinstance(username, str) or not _USERNAME_RE.match(username):
        raise ValueError(
            f"username must match {_USERNAME_RE.pattern!r}, got {username!r}"
        )
    return username


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def list_users(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return every user row, sorted by username ascending.

    An empty list is returned for an empty table — never ``None``.
    """
    init_schema(db_path)
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id, username, password_hash, email, display_name, "
            "role, created_at, last_login_at "
            "FROM users ORDER BY username ASC"
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_dict(r) for r in rows]


def get_user(username: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return the row for ``username``, or ``None`` if not found.

    Never raises for a missing username — the gate calls this on
    every login attempt and must be able to compare against ``None``
    without special-casing.
    """
    init_schema(db_path)
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT id, username, password_hash, email, display_name, "
            "role, created_at, last_login_at "
            "FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    finally:
        conn.close()
    return _row_to_dict(row) if row is not None else None


def create_user(
    username: str,
    password: str,
    role: str,
    email: Optional[str] = None,
    display_name: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Insert a new user row.

    Validates ``username`` and ``role`` BEFORE touching the database
    so the admin panel can render a clear form error rather than
    surfacing an opaque ``sqlite3.IntegrityError``. The bcrypt hash
    is computed at call time (cost 12 ≈ 250ms) so a streamlit rerun
    after the form submit doesn't re-hash the same plaintext.

    Raises
    ------
    ValueError
        ``username`` violates the canonical pattern, ``role`` is not
        in ``('admin','user')``, or the username already exists.
    TypeError
        ``password`` is not a ``str``.
    """
    _validate_username(username)
    _validate_role(role)
    if not isinstance(password, str) or not password:
        raise TypeError("password must be a non-empty str")

    init_schema(db_path)
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, email, "
            "display_name, role) VALUES (?, ?, ?, ?, ?)",
            (
                username,
                hash_password(password),
                email,
                display_name,
                role,
            ),
        )
        conn.commit()
    except Exception as exc:
        # Translate SQLite UNIQUE-constraint failure into a clean
        # ValueError so the admin panel can show "username taken"
        # rather than an opaque database error.
        msg = str(exc).lower()
        if "unique" in msg and "username" in msg:
            raise ValueError(f"username {username!r} already exists") from exc
        raise
    finally:
        conn.close()


def delete_user(username: str, db_path: Optional[Path] = None) -> None:
    """Remove the row for ``username``.

    Idempotent: deleting a missing username is a no-op (no error).
    The admin panel calls this from a per-row delete button where
    double-clicks and stale UI state are common — a raised error on
    the second call would surface a confusing trace to the operator.
    """
    init_schema(db_path)
    conn = get_connection(db_path)
    try:
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
    finally:
        conn.close()


def update_last_login(username: str, db_path: Optional[Path] = None) -> None:
    """Stamp ``last_login_at`` with the current UTC ISO-8601 timestamp.

    Idempotent on missing username (silently no-ops) — same lifetime
    story as :func:`delete_user`. The timestamp is rendered in the
    admin panel's user table and parsed by
    ``datetime.fromisoformat`` in the test layer, so the format
    MUST be a valid ISO-8601 string.
    """
    init_schema(db_path)
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE users SET last_login_at = ? WHERE username = ?",
            (ts, username),
        )
        conn.commit()
    finally:
        conn.close()
