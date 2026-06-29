"""Bootstrap admin seeding for the auth DB (Issue #22, AC7).

First-run path: when the ``users`` table is empty, no admin row
exists, and the operator cannot log in. The bootstrap:

1. Reads ``TELECOM_RAG_BOOTSTRAP_ADMIN_PASSWORD`` from the environment.
2. If the var is set (and non-empty), creates the ``admin`` row with
   role=admin and that bcrypt-hashed password.
3. If the var is unset or empty, returns a banner string for the auth
   widget to render — the operator sees clear next-step instructions
   instead of a hard 500.
4. If the table is non-empty, the bootstrap is a no-op (idempotent —
   no admin overwrite on app restart).

Why a banner, not an exception
------------------------------

The pre-Issue #22 auth gate silently used the hardcoded
``admin / telecom2024``. The bootstrap is the migration seam — a
fresh checkout has no admin row, and forcing the operator to set an
env var BEFORE the app can render a login widget would be hostile.
A yellow banner that explains the next step is the right UX.

Performance note
----------------

``bootstrap.ensure_bootstrap_admin`` runs at Streamlit app-import
time (i.e., on every cold start AND on every Streamlit rerun). The
DB lookup is fast (~1ms warm), but the bcrypt call inside
``create_user`` at cost factor 12 takes ~250ms. The auth gate
caches the bootstrap via :func:`cache_bootstrap_banner` (Streamlit's
``@st.cache_resource``), so the bcrypt cost is paid exactly ONCE
per process — subsequent reruns see the cached empty-string banner
and skip the DB write. The cache is invalidated only when an admin
explicitly deletes every user, which is operationally rare.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

from telecom_rag.auth.db import init_schema
from telecom_rag.auth.users import create_user, list_users

__all__ = ["ensure_bootstrap_admin", "cache_bootstrap_banner"]

# Default display name for the bootstrap admin row — readable in the
# admin panel's user table until the operator edits it.
_BOOTSTRAP_DISPLAY_NAME = "Administrator"
_BOOTSTRAP_EMAIL = "admin@telecom.local"

# Banner string surfaced when no env var is set. Concise — the auth
# widget renders it once per app start; verbose error pages are an
# anti-pattern.
_SETUP_BANNER = (
    "**First-run setup required.** No users exist yet. Set "
    "`TELECOM_RAG_BOOTSTRAP_ADMIN_PASSWORD` in `.env` to a non-empty "
    "value and refresh to seed the initial admin user."
)


def ensure_bootstrap_admin(db_path: Optional[Path] = None) -> str:
    """Seed the ``admin`` row on first run, or no-op if already seeded.

    Returns
    -------
    str
        Empty string when bootstrap was performed (or skipped because
        the DB is non-empty). A non-empty banner string when the
        table is empty AND the env var is unset — the auth widget
        renders this so the operator sees next-step instructions.
    """
    init_schema(db_path)

    # Gate on table emptiness: if any user exists (admin or not), the
    # bootstrap is skipped. This pins the invariant that bootstrap is
    # strictly a first-run convenience — once the operator has created
    # any user, subsequent boots never auto-create an admin.
    existing = list_users(db_path)
    if existing:
        return ""

    # Consume the field via a FRESH ``Settings()`` (not the module
    # singleton) — the ``Settings`` loader is the single source of
    # truth for env-var resolution, with ``env_prefix="TELECOM_RAG_"``
    # gating both the shell env and the .env file. Reading directly
    # via ``os.environ.get`` only sees shell-exported vars and
    # silently misses .env content — the pre-Issue #34 silent-
    # bootstrap bug surfaced exactly that gap. A fresh ``Settings()``
    # per call is microsecond-cheap and lets tests exercise
    # ``monkeypatch.setenv`` correctly (the module singleton is
    # captured at import time and is immune to test-side setenv).
    # See ``tests/test_config_env_prefix.py`` for the regression
    # guard.
    from telecom_rag.config import Settings

    raw = Settings().bootstrap_admin_password
    password = raw.strip() if isinstance(raw, str) else ""
    if not password:
        # Empty DB + no env var: return the banner. The auth widget
        # surfaces this BEFORE the streamlit-authenticator widget mounts,
        # so the operator sees the setup instructions even if the env
        # var path is the only thing missing.
        return _SETUP_BANNER

    # Empty DB + env var set: create the admin row.
    create_user(
        username="admin",
        password=password,
        role="admin",
        email=_BOOTSTRAP_EMAIL,
        display_name=_BOOTSTRAP_DISPLAY_NAME,
        db_path=db_path,
    )
    return ""


@st.cache_resource
def cache_bootstrap_banner() -> str:
    """Cache the bootstrap result for the lifetime of the Streamlit process.

    ``ensure_bootstrap_admin()`` is pure-Python + sqlite (no Streamlit
    calls inside), but the bcrypt cost factor 12 hash inside
    ``create_user`` takes ~250ms — and ``app/main.py`` calls the
    bootstrap on every rerun. Caching here pins the cost to one
    invocation per process. The cache is invalidated only on full
    app restart, which is the right operational story for the rare
    "operator deleted every user, need to re-bootstrap" case.
    """
    return ensure_bootstrap_admin()
