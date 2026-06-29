"""Admin page (read-only observability surface for Issue #6 AC3 + Issue #22).

Role gate (Issue #22 AC5)
-------------------------

The page is admin-only. Any non-admin role hits an early-return
``st.error("Admin access required")`` + ``st.stop()`` BEFORE the
observability widgets render — the per-role streamlit-authenticator
session determines access via ``st.session_state.role`` (set in
``app/main.py`` after a successful login).

Why a hard ``st.stop()``, not a redirect
----------------------------------------

``st.stop()`` halts the page render at the role check. The user
sees only the "Admin access required" message, never the provider
status / collection stats / cost data. A redirect to ``app/main.py``
would (a) lose the user's current scroll position and (b) require a
second round-trip; ``st.stop()`` is the right shape.

User management expander (Issue #22 AC6)
----------------------------------------

A new ``User Management`` expander at the top of the page lists
all rows in the ``users`` table and exposes a create-user form
plus per-row delete buttons. The expander is the FIRST thing
admins see on this page — it's the highest-frequency operation
(adding/removing testers) and sits above the read-only
observability sections.

Three ``st.expander`` sections (existing — Issue #6 + #16 + #19 + #20):

1. **Provider status** — render ``telecom_rag.llm.provider_status()`` so
   ops staff can confirm which LLM providers are wired up, in what
   priority order, and which models are configured for chat / embedding.
2. **Collection stats** — render
   :func:`telecom_rag.tools.retrieval.collection_stats` as a small
   metrics panel: total chunks, distinct source PDFs, and the most
   recently indexed basenames.
3. **Cost & quota** (Issue #16) — render the in-process
   ``telecom_rag.observability.cost`` ledger snapshot: today's USD
   total, per-provider/model breakdown, and the daily quota
   threshold. The ledger is in-process state; numbers reflect the
   current Streamlit session only. Per-call JSON log records
   (routed to CloudWatch via Issue #13) are the durable source of
   truth and survive Streamlit restarts.

Per the 2026-06-24 scope-down decision, the third "Feedback log"
section originally scoped for #6 is deferred to Issue #7 — the
validation agent's grading hook owns the write-path. No placeholder
section is added.

All expanders default to ``expanded=False`` so the page-load stays
under the README §7.1 admin SLA (< 2s) — the underlying provider +
collection + cost reads are deferred until the user clicks.
"""

from __future__ import annotations

import re
from typing import Any, Dict

import pandas as pd
import streamlit as st

# Route through the source module attribute (rather than ``from telecom_rag
# import llm; llm.provider_status()``) so tests that monkeypatch
# ``telecom_rag.llm.provider_status`` see the patched function. Same
# pattern for the retrieval and cost modules.
from telecom_rag import llm as llm_module
from telecom_rag.auth.users import (
    create_user,
    delete_user,
    list_users,
)
from telecom_rag.ingestion import pipeline as pipeline_module
from telecom_rag.observability import cost as cost_module
from telecom_rag.tools.retrieval import collection_stats, get_vectorstore


# ---------------------------------------------------------------------------
# Role gate (Issue #22 AC5) — fail closed before any expensive import runs
# ---------------------------------------------------------------------------

_role = st.session_state.get("role")
if _role != "admin":
    st.error("Admin access required")
    st.stop()


# ---------------------------------------------------------------------------
# User Management expander (Issue #22 AC6)
# ---------------------------------------------------------------------------

# Form-validators mirror the canonical user-creation rules from
# ``telecom_rag.auth.users``. Keeping them here as local constants
# lets the form render clear error messages BEFORE the DB call,
# instead of surfacing a raw ``ValueError`` from the CRUD layer.
_USERNAME_RE = re.compile(r"^[a-z0-9_]{3,32}$")
_MIN_PASSWORD_LEN = 8
_VALID_ROLES = ("admin", "user")


def _render_user_management() -> None:
    """Render the User Management expander (Issue #22 AC6).

    Three sub-sections, all inside one expander:

    1. The user table (username, role, email, display name, created
       at, last login at) rendered as a DataFrame so the admin can
       see who's currently in the system at a glance.
    2. A per-row delete button (with ``st.session_state`` confirmation
       — no ``st.confirm_dialog`` API exists at the time of writing,
       so the admin must click "Confirm delete" in a second step).
    3. A create-user form below the table.
    """
    st.markdown(
        "Create and delete users. Admin role grants access to "
        "this page; user role grants access to Chat, Upload, and "
        "the My Uploads page only."
    )

    users = list_users()
    if not users:
        st.info(
            "No users in the auth DB yet. Use the form below to "
            "create the first user."
        )
    else:
        # Render the user table — columns chosen for admin clarity
        # (no password_hash column, obviously). The dataframe index
        # is the username so the per-row delete button (rendered
        # below) is keyed deterministically.
        rows_for_df = [
            {
                "username": u["username"],
                "role": u["role"],
                "email": u.get("email") or "",
                "display_name": u.get("display_name") or "",
                "created_at": u.get("created_at") or "",
                "last_login_at": u.get("last_login_at") or "never",
            }
            for u in users
        ]
        df = pd.DataFrame(rows_for_df)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Per-row delete. We render one button per row and confirm
        # via a two-step pattern (click "Delete X" → click "Confirm
        # delete X" in the next render). The session-state key is
        # ``_pending_delete_username`` so the admin can confirm OR
        # cancel by clicking elsewhere (the cancel button below).
        st.markdown("---")
        st.markdown("**Delete a user**")
        for u in users:
            cols = st.columns([3, 1, 1])
            cols[0].markdown(
                f"`{u['username']}` ({u['role']}) — "
                f"{u.get('display_name') or u['username']}"
            )
            pending = st.session_state.get("_pending_delete_username")
            if pending == u["username"]:
                cols[1].button(
                    "Cancel",
                    key=f"cancel_delete_{u['username']}",
                    on_click=_clear_pending_delete,
                )
                if cols[2].button(
                    "Confirm delete",
                    key=f"confirm_delete_{u['username']}",
                    type="primary",
                ):
                    try:
                        delete_user(u["username"])
                        st.success(
                            f"Deleted user {u['username']!r}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not delete: {exc}")
                    _clear_pending_delete()
            else:
                if cols[2].button(
                    "Delete",
                    key=f"delete_{u['username']}",
                ):
                    st.session_state._pending_delete_username = (
                        u["username"]
                    )

    # Create-user form. The form validator surfaces clear errors
    # BEFORE the DB call so the admin sees "password too short"
    # rather than a raw ``ValueError`` from the CRUD layer.
    st.markdown("---")
    st.markdown("**Create a new user**")
    with st.form("create_user_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        username = c1.text_input(
            "Username",
            help="3-32 chars, lowercase letters, digits, underscore.",
        )
        role = c2.selectbox("Role", _VALID_ROLES)
        c3, c4 = st.columns(2)
        password = c3.text_input(
            "Password",
            type="password",
            help=f"Minimum {_MIN_PASSWORD_LEN} characters.",
        )
        confirm = c4.text_input(
            "Confirm password",
            type="password",
            help="Re-enter the password to confirm.",
        )
        c5, c6 = st.columns(2)
        email = c5.text_input("Email (optional)", value="")
        display_name = c6.text_input(
            "Display name (optional)",
            value="",
            help="Shown in the sidebar welcome message.",
        )
        submitted = st.form_submit_button("Create user")

    if submitted:
        _handle_create_user(
            username=username,
            password=password,
            confirm=confirm,
            role=role,
            email=email,
            display_name=display_name,
        )


def _handle_create_user(
    username: str,
    password: str,
    confirm: str,
    role: str,
    email: str,
    display_name: str,
) -> None:
    """Validate the create-user form and call ``create_user``.

    Validates locally so the admin sees a per-field error message
    instead of a generic ``ValueError`` from the CRUD layer. On
    success, surfaces a green confirmation; on failure, the error
    string is rendered as ``st.error``.

    Username normalization (regression fix)
    --------------------------------------
    ``username`` is normalized with ``.strip().lower()`` BEFORE the
    regex match AND BEFORE the ``create_user(...)`` call. The
    ``_USERNAME_RE`` in :mod:`telecom_rag.auth.users` is deliberately
    strict (lowercase-only, no whitespace) for log/audit query
    predictability — we do NOT loosen that regex. Instead the form
    layer is the right boundary for "input normalization": admins
    who type ``Aiswarya`` or `` aiswarya `` get the same canonical
    stored username ``aiswarya``. An all-whitespace input still
    rejects (becomes empty after strip → fails the regex).

    ``email`` and ``display_name`` are stripped of leading/trailing
    whitespace (case preserved) so common admin typos like a stray
    newline don't leak into the user table UI.
    """
    username = (username or "").strip().lower()
    email = (email or "").strip() or None
    display_name = (display_name or "").strip() or None

    if not _USERNAME_RE.match(username):
        st.error(
            "Username must be 3-32 chars of lowercase letters, "
            "digits, or underscore."
        )
        return
    if not password or len(password) < _MIN_PASSWORD_LEN:
        st.error(
            f"Password must be at least {_MIN_PASSWORD_LEN} "
            f"characters."
        )
        return
    if password != confirm:
        st.error("Passwords do not match.")
        return
    if role not in _VALID_ROLES:
        st.error(f"Role must be one of {_VALID_ROLES!r}.")
        return

    try:
        create_user(
            username=username,
            password=password,
            role=role,
            email=email,
            display_name=display_name,
        )
        st.success(
            f"Created user {username!r} ({role}). They can now log in."
        )
    except ValueError as exc:
        # E.g. duplicate username — surface the CRUD-layer error
        # directly. The admin sees the same message we'd surface
        # to a programmatic caller, which is the right UX.
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not create user: {exc}")


def _clear_pending_delete() -> None:
    """Reset the two-step delete confirmation state."""
    st.session_state._pending_delete_username = None


@st.cache_resource
def _get_vs():
    """Load the persisted Chroma vectorstore once per Streamlit session.

    ``@st.cache_resource`` (not ``@st.cache_data``) is required because
    Chroma / LangChain wrapper objects are not pickleable. Mirrors the
    pattern in :mod:`app.pages.chat`.
    """
    return get_vectorstore()


def _render_provider_status(status: Dict[str, Any]) -> None:
    """Render ``provider_status()`` as a small key/value panel.

    Layout: priority list as a single bullet row, then a definition-list
    row per provider showing ``configured / in_priority / chat_model /
    embedding_model``. Using ``st.markdown`` (not ``st.json``) because
    the panel reads more naturally as a status card than as a raw JSON
    blob in the admin context.
    """
    priority = status.get("priority", []) or []
    if priority:
        priority_md = ", ".join(f"`{p}`" for p in priority)
    else:
        priority_md = "(none configured)"

    st.markdown(f"**Priority order:** {priority_md}")

    for provider_name in ("openai", "gemini"):
        info = status.get(provider_name, {}) or {}
        configured = info.get("configured", False)
        in_priority = info.get("in_priority", False)
        chat_model = info.get("chat_model") or "(not set)"
        embedding_model = info.get("embedding_model") or "(not set)"

        indicator = "READY" if (configured and in_priority) else (
            "CONFIGURED" if configured else "NOT CONFIGURED"
        )
        st.markdown(
            f"- **{provider_name}** — `{indicator}`  \n"
            f"  - chat model: `{chat_model}`  \n"
            f"  - embedding model: `{embedding_model}`"
        )


def _render_collection_stats(stats: Dict[str, Any]) -> None:
    """Render ``collection_stats()`` as a metric strip + bullet list.

    Uses :func:`st.metric` for the scalar counts so ops staff can read
    them at a glance, then a plain bulleted list for ``recent_sources``.

    Issue #20: when a re-scan has completed in this Streamlit session,
    renders a "Last re-scan" sub-section under the existing collection
    stats, showing the most recent ``reingest_directory(apply=True)``
    summary's totals + duration.
    """
    total_chunks = int(stats.get("total_chunks", 0))
    distinct_sources = int(stats.get("distinct_sources", 0))
    recent_sources = stats.get("recent_sources", []) or []

    col1, col2 = st.columns(2)
    col1.metric("Total chunks", total_chunks)
    col2.metric("Distinct sources", distinct_sources)

    if recent_sources:
        st.markdown("**Recent sources (insertion order, max 10):**")
        for src in recent_sources:
            st.markdown(f"- `{src}`")
    else:
        st.markdown("_No indexed sources yet — upload a PDF to populate._")

    # Issue #20 AC9: "Last re-scan" sub-section. Reads from the
    # in-process module-level state (``pipeline_module.LAST_RESCAN``).
    # In-process only — for durable history, query CloudWatch with
    # ``event=ingest_completed`` (Issue #13 surface).
    last = pipeline_module.get_last_rescan()
    if last is not None:
        st.markdown("---")
        st.markdown("**Last re-scan (Issue #20):**")
        totals = last.get("totals", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("New", totals.get("new", 0))
        c2.metric("Changed", totals.get("changed", 0))
        c3.metric("Stale", totals.get("stale", 0))
        st.markdown(
            f"- batch_id: `{last.get('batch_id', '?')}`  \n"
            f"- duration: **{last.get('duration_ms', 0)} ms**  \n"
            f"- unchanged: {totals.get('unchanged', 0)} files"
        )


def _render_cost_quota() -> None:
    """Render the cost / quota ledger snapshot (Issue #16 + Issue #19).

    The page shows today's USD total (from the in-process ledger), a
    per-(provider/model) breakdown, and — for Issue #19 — a
    per-agent breakdown so operators can see which agent is
    driving cost spikes. When the ledger is empty, a friendly
    placeholder is shown — no metric values, no error.

    Caveat (documented in the section header): the in-process ledger
    reflects the current Streamlit session only. For durable per-call
    history, query the JSON log stream (CloudWatch Logs Insights on the
    ``telecom_rag.observability.cost`` logger; ``event=cost_record``).
    """
    st.markdown(
        "_In-process session ledger. Per-call JSON log records "
        "(`event=cost_record`) are the durable source of truth._"
    )

    ledger = cost_module.get_ledger()
    snap = ledger.snapshot()
    daily_total = ledger.daily_total_usd()
    total_calls = ledger.total_calls()
    total_tokens = ledger.total_tokens()

    if total_calls == 0:
        st.markdown("_No LLM calls recorded in this session yet._")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Today's cost", f"${daily_total:.4f}")
    col2.metric("Calls (session)", total_calls)
    col3.metric("Tokens (session)", f"{total_tokens:,}")

    st.markdown("**Per-provider / model breakdown:**")
    # Sort by cost desc so the most expensive model surfaces first.
    # Issue #19: skip the ``by_agent`` sibling key — rendered below.
    rows = [
        (key, info)
        for key, info in snap.items()
        if key != "by_agent"
    ]
    rows.sort(key=lambda kv: kv[1].get("cost_usd", 0.0), reverse=True)
    for key, info in rows:
        st.markdown(
            f"- `{key}` — {info['calls']} calls, "
            f"{info['total_tokens']:,} tokens, "
            f"${info['cost_usd']:.6f}"
        )

    # Issue #19: per-agent breakdown. The ``by_agent`` key is a
    # sibling of the per-provider rows inside ``snapshot()``; the
    # CloudWatch dashboard widget groups the same data by agent
    # over a 24h window.
    by_agent = snap.get("by_agent", {})
    if by_agent:
        st.markdown("**Per-agent breakdown (Issue #19):**")
        agent_rows = sorted(
            by_agent.items(),
            key=lambda kv: kv[1].get("cost_usd", 0.0),
            reverse=True,
        )
        for agent_key, info in agent_rows:
            st.markdown(
                f"- `{agent_key}` — {info['calls']} calls, "
                f"{info['total_tokens']:,} tokens, "
                f"${info['cost_usd']:.6f}"
            )


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

st.title("Admin")
st.markdown(
    "Read-only observability surface + user management. All four "
    "sections are lazy-loaded inside expanders so the page renders "
    "in under 2s even on a populated corpus."
)

with st.expander("User Management", expanded=False):
    _render_user_management()

with st.expander("Provider status", expanded=False):
    try:
        status = llm_module.provider_status()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read provider status: {exc}")
    else:
        _render_provider_status(status)

with st.expander("Collection stats", expanded=False):
    try:
        vs = _get_vs()
        stats = collection_stats(vs)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read collection stats: {exc}")
    else:
        _render_collection_stats(stats)

with st.expander("Cost & quota", expanded=False):
    try:
        _render_cost_quota()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read cost ledger: {exc}")