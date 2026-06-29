"""LangSmith tracing self-check (Issue #13 AC5 / AC6).

Exposes :func:`check_tracing_configured` which inspects the live
:data:`telecom_rag.config.settings` object and returns a list of
human-readable messages describing the tracing configuration.

Why this exists: LangChain's tracing is env-driven. If an operator
sets ``LANGCHAIN_TRACING_V2=true`` but forgets
``LANGCHAIN_API_KEY``, the SDK silently no-ops and zero traces
land in Smith — you only find out by going to look for them. The
self-check surfaces the misconfiguration at app-startup so the
operator sees a warning on every Streamlit page rerun.

The four cases:

| tracing_v2 | api_key   | returns                              |
|------------|-----------|--------------------------------------|
| True       | None      | [warning: "tracing on but key missing"] |
| True       | set       | []                                   |
| False      | None      | []  (dev default — nothing to warn)  |
| False      | set       | [info: "key set but tracing disabled"]|
"""

from __future__ import annotations

from typing import List

from telecom_rag.config import settings


def check_tracing_configured() -> List[str]:
    """Return a list of human-readable messages describing the
    LangSmith tracing configuration.

    Always returns at least one message in the
    ``tracing_off + key_set`` case (informational, not a warning).
    Returns an empty list in the ``tracing_off + no_key`` case (the
    dev default) and in the ``tracing_on + key_set`` case (correct
    configuration). Returns a single warning in the
    ``tracing_on + no_key`` case.
    """
    tracing_on = bool(settings.langchain_tracing_v2)
    key_set = bool(settings.langchain_api_key)

    if tracing_on and not key_set:
        return [
            "LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is not set. "
            "Traces will be silently dropped. Set LANGCHAIN_API_KEY in "
            ".env (and rotate the key in Smith if this is a new env)."
        ]
    if not tracing_on and key_set:
        return [
            "LANGCHAIN_API_KEY is set but LANGCHAIN_TRACING_V2 is false. "
            "Traces will not be sent to LangSmith. Set "
            "LANGCHAIN_TRACING_V2=true to enable tracing for this env."
        ]
    return []
