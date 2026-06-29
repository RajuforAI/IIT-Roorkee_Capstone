"""Observability helpers (Issue #13).

This package hosts the structured-logging, LangSmith self-check,
and health-check helpers called out by README §16.

The modules are intentionally dependency-light (stdlib only for the
logging helper; boto3 is lazy-imported by health.py) so they can
be imported from ``app/main.py`` at startup without bringing the
heavyweights online.
"""

from telecom_rag.observability.logging import setup_json_logging
from telecom_rag.observability.langsmith import check_tracing_configured

__all__ = [
    "setup_json_logging",
    "check_tracing_configured",
]
