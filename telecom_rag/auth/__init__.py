"""Multi-user authentication for the Streamlit app (Issue #22).

Public surface
--------------

- :mod:`telecom_rag.auth.passwords` — bcrypt wrap (``hash_password``,
  ``verify_password``).
- :mod:`telecom_rag.auth.db` — SQLite connection + schema bootstrap.
- :mod:`telecom_rag.auth.users` — CRUD for the ``users`` table.
- :mod:`telecom_rag.auth.bootstrap` — first-run admin seeding.

The auth DB lives at ``./auth.db`` by default (overridable via the
``TELECOM_RAG_AUTH_DB`` env var). It is intentionally a SEPARATE file
from the LangGraph ``checkpoints.db`` so a corruption in one does not
take down the other.

Why a fresh package, not an extension of ``telecom_rag.security``
----------------------------------------------------------------

``telecom_rag.security`` is the PII / secret SCANNER surface — it
inspects document content for sensitive patterns. The auth module is a
credential STORE + IDENTITY layer. Mixing the two would muddle the
boundary between "what data we hold about users" and "what content we
allow into the corpus." Keeping them apart also keeps
``security/policy.py`` free of bcrypt imports (the bucket-policy code
should remain cheap to import for every upload).
"""

from __future__ import annotations

from telecom_rag.auth.passwords import hash_password, verify_password
from telecom_rag.auth.users import (
    create_user,
    delete_user,
    get_user,
    list_users,
    update_last_login,
)

__all__ = [
    "hash_password",
    "verify_password",
    "create_user",
    "delete_user",
    "get_user",
    "list_users",
    "update_last_login",
]
