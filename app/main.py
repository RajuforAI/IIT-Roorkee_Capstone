"""Streamlit entry point (Issue #7 AC9 + Issue #22).

Configures the page, renders the auth gate, and surfaces the multi-
page navigation once the user is authenticated. The chat read-path
lives in ``app/pages/chat.py``.

Auth gate (Issue #22)
---------------------

Multi-user authentication backed by SQLite (``./auth.db``) and
bcrypt. The first-run admin is seeded from the
``TELECOM_RAG_BOOTSTRAP_ADMIN_PASSWORD`` env var; the operator can
then create additional users (admin or user role) via the
``User Management`` expander on the ``/admin`` page. The
``Authenticate`` widget returns ``(name, authenticator_status,
username)`` on a successful login; we stash
``st.session_state.authenticated = True`` and
``st.session_state.role`` (from the DB row) so the inner pages can
gate themselves — ``/admin`` is admin-only, ``/my_uploads`` is
visible to both roles.

Why the credentials dict is cached, not recomputed per rerun
------------------------------------------------------------

``streamlit-authenticator`` requires the stored password to be hashed
(bcrypt) before it compares against the user's input. Hashing on every
rerun would be wasteful (bcrypt cost factor 12 ≈ 250ms per password).
We cache the hashed dict in ``@st.cache_resource`` keyed on
``os.path.getmtime(auth_db_path())`` so:

1. The hash is computed once per process — NOT per Streamlit rerun.
2. When an admin creates / deletes a user via the ``/admin`` User
   Management expander, the cache invalidates automatically (the DB
   file's mtime changes), and the next rerun rebuilds the dict.
   No explicit ``st.cache_resource.clear()`` plumbing required.

We still re-query the DB on every rerun for the per-user ``role``
field (the credentials dict only carries ``email``/``name``/
``password`` per the streamlit-authenticator contract), so a freshly
promoted admin sees the ``/admin`` sidebar link on the next click.
"""

from __future__ import annotations

import logging

import streamlit as st
import streamlit_authenticator as stauth

from telecom_rag.auth.bootstrap import cache_bootstrap_banner
from telecom_rag.auth.users import get_user, list_users
from telecom_rag.config import settings
from telecom_rag.observability import check_tracing_configured
from telecom_rag.observability.logging import setup_json_logging

st.set_page_config(
    page_title="TeleGenie AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---- Observability wiring (Issue #13) -------------------------------------
# Install JSON logging on the root logger so every record emitted by
# the app + libraries becomes one newline-delimited JSON object.
# CloudWatch Logs Insights and ``awslogs``/``vector`` consume this
# format natively — no regex parsing on the receiving end.
#
# The LangSmith self-check runs on every page rerun so a misconfigured
# env (tracing on but key missing) is visible to the operator until
# they fix it. The check returns an empty list when the env is
# consistent; non-empty results are rendered as warnings.

setup_json_logging(settings.log_level)

_langsmith_messages = check_tracing_configured()
for _msg in _langsmith_messages:
    # We can't easily distinguish warnings from info at this seam
    # without changing the helper's return shape; the helper's docs
    # already say "warning" in the message text for the bad cases.
    if "silently dropped" in _msg or "not be sent" in _msg:
        st.warning(_msg)
    else:
        st.info(_msg)

logger = logging.getLogger(__name__)


# ---- Auth credentials (Issue #22) ---------------------------------------
# Replaces the pre-#22 hardcoded ``_RAW_PASSWORD = "telecom2024"`` with
# a SQLite-backed user store. The bootstrap is called on import so
# the first-render path always has at least one row in the table
# (when ``TELECOM_RAG_BOOTSTRAP_ADMIN_PASSWORD`` is set) or surfaces
# the setup banner (when it isn't). The bootstrap is cached via
# :func:`cache_bootstrap_banner` so the bcrypt cost-factor-12 hash
# inside ``create_user`` is paid once per process, NOT per Streamlit
# rerun. Subsequent reruns return the cached empty-string banner
# without touching the DB.
#
# ``_hashed_credentials`` is also re-evaluated per rerun, but it
# only does a SELECT (no bcrypt) — the hashes are pre-computed at
# user-create time and stored on disk. ``list_users()`` is a cheap
# indexed query (~1ms warm). Admin CRUDs (create / delete) surface
# in the next rerun's dict without any explicit cache plumbing.

_BOOTSTRAP_BANNER = cache_bootstrap_banner()


def _hashed_credentials() -> dict:
    """Build the credentials dict from the SQLite ``users`` table.

    Returns the dict in the shape ``streamlit-authenticator``'s
    ``Authenticate`` widget expects:

        {"usernames": {<username>: {"email": ..., "name": ..., "password": <bcrypt hash>}}}

    Returns an empty ``{"usernames": {}}`` when the table is empty
    (post-bootstrap-banner state). The widget mount below gates on
    ``bool(_hashed_credentials()["usernames"])`` and renders the
    setup banner instead of an auth widget in that case — passing
    an empty dict to ``Authenticate(...)`` would raise an internal
    IndexError on the first login attempt.
    """
    rows = list_users()
    usernames: dict = {}
    for row in rows:
        usernames[row["username"]] = {
            "email": row.get("email") or "",
            "name": row.get("display_name") or row["username"],
            "password": row["password_hash"],
        }
    return {"usernames": usernames}


# ---- Auth widget ---------------------------------------------------------

# Sign cookies with the SECRET_KEY env var when set, fall back to a
# stable dev string when running under AppTest (the test harness
# doesn't load ``.env``). The fallback is logged so a misconfigured
# production deploy shows up in CloudWatch — see the warning emitted
# below.
_cookie_key = settings.secret_key or "dev-only-cookie-key-DO-NOT-USE-IN-PROD"
if not settings.secret_key:
    logger.warning(
        "SECRET_KEY is not set; using a development fallback cookie "
        "signing key. Set TELECOM_RAG_SECRET_KEY (or the SECRET_KEY "
        "env var) in .env for any non-local deployment."
    )

_credentials = _hashed_credentials()
_has_users = bool(_credentials.get("usernames"))

if not _has_users:
    # No users in the DB and bootstrap did not seed one (env var
    # unset). Render the setup banner and stop — mounting the auth
    # widget with an empty credentials dict raises inside
    # streamlit-authenticator.
    st.warning(_BOOTSTRAP_BANNER or "No users exist. Contact an admin.")
    st.stop()

authenticator = stauth.Authenticate(
    _credentials,
    "telecom_auth_cookie",  # cookie name
    _cookie_key,
    cookie_expiry_days=1,
)

name, authentication_status, username = authenticator.login(location="main")

if authentication_status:
    # Stamp last_login_at on the FIRST successful login in this
    # session only — every subsequent Streamlit rerun would otherwise
    # write the timestamp on every widget interaction. We guard via
    # ``st.session_state._last_login_stamped`` so the DB write fires
    # exactly once per browser session, not once per render.
    if not st.session_state.get("_last_login_stamped"):
        try:
            from telecom_rag.auth.users import update_last_login
            update_last_login(username)
        except Exception as exc:  # noqa: BLE001
            # A DB write failure (e.g., a transient SQLite lock) must
            # NOT block the login flow — the user can still use the
            # app, the timestamp is just stale. Log and move on.
            logger.warning("update_last_login failed: %s", exc)
        st.session_state._last_login_stamped = True

    # The role field is re-queried from the DB on every rerun (the
    # credentials dict doesn't carry it) so a freshly-promoted admin
    # sees the /admin sidebar link without a full page reload.
    user_row = get_user(username)
    role = user_row["role"] if user_row else "user"
    display_name = (
        user_row.get("display_name") if user_row else None
    ) or name

    authenticator.logout("Logout", "sidebar")
    st.session_state.authenticated = True
    st.session_state.username = username
    st.session_state.role = role
    st.session_state.display_name = display_name

    st.title("Welcome to TeleGenie AI")
    st.sidebar.success(
        f"Logged in as {display_name} ({role}). Select a page above."
    )
    st.markdown(
        "Your intelligent telecom knowledge companion.\n\n"
        "Ask questions in natural language and receive accurate, cited "
        "answers from your organization's trusted technical documentation.\n\n"
        "Start a conversation from the **Chat** page or expand your knowledge "
        "base through **Upload**."
    )
    # Admin-only sidebar hint. The actual ``/admin`` page is also
    # gated on ``st.session_state.role == "admin"`` (see
    # ``app/pages/admin.py``) so a direct URL hit by a non-admin
    # user still hits the role-gate and renders "Admin access
    # required" rather than the observability widgets.
    if role == "admin":
        st.sidebar.info(
            "You have admin access — the **Admin** page is available "
            "in the sidebar for user management, collection stats, "
            "and the cost/quota ledger."
        )
elif authentication_status is False:
    st.error("Username/password is incorrect")
elif authentication_status is None:
    st.warning("Please enter your username and password")
