"""Per-user upload history page (Issue #22 AC8).

Renders a dataframe of the calling user's recent upload activity
(indexed + blocked) by reading the structured JSONL audit log that
:func:`telecom_rag.observability.logging.setup_json_logging` writes
to ``./logs/telecom_rag.jsonl``. The reader lives in
:mod:`telecom_rag.observability.upload_log`.

Role behavior
-------------

* **User role** — sees only their own uploads (filtered by
  ``st.session_state.username``). No filter UI exposed; the page
  is read-only for this role.
* **Admin role** — sees their own uploads by default, with a
  ``st.multiselect`` over the user list to widen the view to all
  users. The multiselect default is ``["<admin>"]``; selecting
  more widens; selecting none shows an empty table.

Why a separate page from ``/upload``
------------------------------------

``/upload`` is the ingestion surface — users go there to add new
PDFs. ``/my_uploads`` is the read-only history surface — users go
there to check what they (and, for admins, others) have uploaded
recently. Keeping them separate means each page has one job, and
the user-role gate on ``/upload`` (PII scanner, sha256 hash) does
not constrain the read-only audit view.

Empty-state handling
--------------------

If the JSONL file is missing OR the user has zero matching records,
the page renders a friendly banner instead of an empty DataFrame.
CloudWatch is the durable source of truth — local-cache misses
don't block the user from seeing the page render.

Column contract
---------------

The dataframe columns are:
    timestamp, file_name, status ("indexed" or "blocked"),
    chunks (indexed only), error (blocked only), file_sha256,
    batch_id (indexed only), user (visible only for admins).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from telecom_rag.auth.users import list_users
from telecom_rag.observability import upload_log

# Sentinel session-state key for the admin's user filter. Kept
# private (leading underscore) so it doesn't collide with other
# pages' session state.
_ADMIN_FILTER_KEY = "_my_uploads_admin_user_filter"

# Default per-page record cap. The reader already over-reads by 2x
# to absorb filtered-out lines, so this is the post-filter bound.
_PAGE_LIMIT = 500


def _build_display_rows(records: List[Dict[str, Any]], show_user: bool) -> List[Dict[str, Any]]:
    """Flatten audit-log records into the column shape the page renders.

    ``show_user`` is True when the viewer is admin and the multiselect
    has widened the view beyond the admin's own user — non-admins
    always see only their own user, so the column is suppressed.
    """
    rows: List[Dict[str, Any]] = []
    for rec in records:
        event = rec.get("event")
        row = {
            "timestamp": rec.get("timestamp") or "",
            "file_name": rec.get("file_name") or "",
            "status": "indexed" if event == "upload_indexed" else "blocked",
            "chunks": rec.get("chunks") if event == "upload_indexed" else None,
            "error": (
                ", ".join(rec.get("blocked_patterns") or [])
                if event == "upload_blocked"
                else None
            ),
            "file_sha256": rec.get("file_sha256") or "",
            "batch_id": rec.get("batch_id") or "",
        }
        if show_user:
            row["user"] = rec.get("user") or ""
        rows.append(row)
    return rows


def _render_empty_state(username: str) -> None:
    """Render the empty-state banner when no records match the filter."""
    st.info(
        f"No upload activity recorded for `{username}` yet. "
        "Use the **Upload** page to add PDFs — successful ingests "
        "and PII-blocked rejections are both logged here."
    )


# ---------------------------------------------------------------------------
# Role gate (Issue #22 AC5)
# ---------------------------------------------------------------------------
# A user must be authenticated to see their upload history. We don't
# gate this page on ``role == "admin"`` — both roles see the page;
# they just see different scopes. An unauthenticated visit (e.g.
# direct URL hit while logged out) falls through to the early-return
# below rather than crashing inside Streamlit's auth state.

_role = st.session_state.get("role")
_username = st.session_state.get("username")

if not _role or not _username:
    st.error("You must be logged in to view your upload history.")
    st.stop()


# ---------------------------------------------------------------------------
# Admin user-filter (Issue #22 AC8 admin-scope widen)
# ---------------------------------------------------------------------------
# Admins default to "see only my uploads" — the same shape the user
# role gets — and can widen via the multiselect. The default is
# pinned to the admin's own username so a refresh preserves scope.
# ``st.session_state`` keyed on the admin's username keeps the
# filter sticky across reruns.

if _role == "admin":
    all_users = sorted({u["username"] for u in list_users()})
    # Seed the filter on the FIRST mount only. We do NOT pre-set
    # ``st.session_state[_ADMIN_FILTER_KEY]`` BEFORE the widget
    # mounts — Streamlit raises ``StreamlitAPIException`` ("default
    # value is not part of the options") when both a default AND a
    # ``key=`` are provided AND the session-state value disagrees.
    # The two-step pattern below sidesteps that: first render with
    # the canonical default via the ``default=`` kwarg, then on
    # subsequent reruns read the sticky value out of session_state.
    if _ADMIN_FILTER_KEY not in st.session_state:
        # First mount: the widget mounts with the admin's own user
        # as default. The widget's own write to session_state then
        # makes it sticky for future reruns.
        _initial_default = [_username] if _username in all_users else (
            all_users[:1] if all_users else []
        )
    else:
        _initial_default = st.session_state[_ADMIN_FILTER_KEY]

    selected = st.multiselect(
        "Show uploads for",
        options=all_users,
        default=_initial_default,
        key=_ADMIN_FILTER_KEY,
        help="Admins see the union. Drop a username to filter out.",
    )
    if not selected:
        st.info("Select at least one user above to view upload history.")
        st.stop()
    _filter_users = selected
else:
    _filter_users = [_username]


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

st.title("My Uploads")
st.markdown(
    "Per-user upload history — every successful ingest and every "
    "PII-blocked rejection. Reads the local audit-log mirror at "
    "`./logs/telecom_rag.jsonl` (CloudWatch is the durable source of truth)."
)

# Issue #22 / Issue #21: when the per-user filter on the dashboard
# widget "Blocks by user" shows the calling user blocked N files in
# the last 24h, the /my_uploads page is where they go to see WHICH
# files — the union of indexed + blocked lives here.

# Read each user's events and concatenate. The reader already
# sorts newest-first within one file, but the cross-user merge
# doesn't preserve that ordering — re-sort the union here so the
# dataframe reads top-to-bottom newest first regardless of which
# user's events we're looking at.

all_records: List[Dict[str, Any]] = []
for _u in _filter_users:
    all_records.extend(
        upload_log.read_upload_events(user=_u, limit=_PAGE_LIMIT)
    )

# Deduplicate by (timestamp, file_name, event) — a single upload can
# emit both upload_indexed (per-file) and the ingest_completed event
# at the page level; we don't want the page-level one in this view.
seen = set()
deduped: List[Dict[str, Any]] = []
for rec in all_records:
    key = (rec.get("timestamp", ""), rec.get("file_name", ""), rec.get("event", ""))
    if key in seen:
        continue
    seen.add(key)
    deduped.append(rec)

deduped.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

if not deduped:
    _render_empty_state(_username if _role != "admin" else ", ".join(_filter_users))
    st.stop()

_rows = _build_display_rows(deduped, show_user=(_role == "admin" and len(_filter_users) > 1))
df = pd.DataFrame(_rows)

# Reorder columns so the most-actionable fields come first.
_col_order = ["timestamp", "status", "file_name"]
if "user" in df.columns:
    _col_order.append("user")
_col_order += ["chunks", "error", "file_sha256", "batch_id"]
df = df[[c for c in _col_order if c in df.columns]]

st.dataframe(df, use_container_width=True, hide_index=True)

# Footer: provenance + record count so ops can sanity-check the
# local-cache shape vs CloudWatch at a glance.
_log_path = upload_log.default_log_path()
st.caption(
    f"Showing {len(deduped)} most-recent record(s) "
    f"(limit {_PAGE_LIMIT}). Source: `{_log_path}`. "
    "If the file is missing, the page reads from CloudWatch via the "
    "admin observability page — see `docs/PLAYBOOK.md` §3.4."
)