"""Configuration module for the Telecom RAG system.

Loads settings from a local ``.env`` file via :class:`pydantic_settings.BaseSettings`.

Provider fallback: ``llm_provider_priority`` controls the order in which
:class:`telecom_rag.llm` tries providers when an upstream call fails.
The default is ``openai,gemini`` — OpenAI is preferred, Gemini is the
fallback. Either provider may be omitted by clearing its key; the
remaining providers are tried in declared order.
"""

from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# ``.env`` lives at the repo root, two directories up from this
# module (telecom_rag/config.py -> telecom_rag/ -> repo_root/).
# Resolving to an ABSOLUTE path here fixes the CWD-relative .env
# loading bug discovered during Issue #13's observability review
# (2026-06-26): when the app is launched from a subdirectory
# (e.g. ``app/``), pydantic-settings would silently miss the .env
# file because it resolved ``env_file=".env"`` against the CWD,
# not against the module's location. See
# ``tests/test_config_env_loading.py`` for the regression guard.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOTENV_PATH = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables / ``.env``.

    Every credential is ``Optional[str]`` with a default of ``None``.
    Pydantic-settings will still raise if you call ``.get_secret_value()``
    on a None field, but a missing key will no longer prevent the
    module from importing — which matters for the Docker smoke test
    (Issue #3) and for the unit test suite (Issue #4).
    """

    # ``env_file`` is an absolute path. If the repo's ``.env`` is
    # absent (e.g. CI without one), the loader silently no-ops;
    # ``env_file_encoding`` keeps Windows CRLF happy.
    #
    # ``env_prefix="TELECOM_RAG_"`` is the single-source-of-truth
    # contract (Issue #34): every field on this class reads its value
    # from ``<prefix><field_name>`` in the environment and in ``.env``.
    # No production code in ``telecom_rag/`` may read
    # ``os.environ.get("TELECOM_RAG_*")`` directly — the regression
    # guard in ``tests/test_config_env_loading.py`` enforces this.
    # Pre-fix, ``Settings`` had no prefix, so a field named
    # ``bootstrap_admin_password`` looked up ``BOOTSTRAP_ADMIN_PASSWORD``
    # (without prefix), silently missing the documented
    # ``TELECOM_RAG_BOOTSTRAP_ADMIN_PASSWORD`` that operators set in
    # ``.env``. The fix is exactly this one line.
    model_config = SettingsConfigDict(
        env_file=str(_DOTENV_PATH) if _DOTENV_PATH.exists() else None,
        env_file_encoding="utf-8",
        env_prefix="TELECOM_RAG_",
        extra="ignore",
    )

    # LLM / tracing
    openai_api_key: Optional[str] = None
    langchain_api_key: Optional[str] = None
    langchain_tracing_v2: bool = False
    langchain_project: str = "TeleGenie AI"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # AWS
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_default_region: str = "us-east-1"
    aws_s3_bucket: Optional[str] = None

    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    # Cookie signing key for the auth gate. Required at app start
    # (``app/main.py`` raises if unset) — the dev value in ``.env``
    # line 18 is fine for local dev; production must override.
    secret_key: Optional[str] = None

    # RAG / vector store
    chroma_persist_dir: str = "./chroma_db"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ---- Gemini fallback (optional) ---------------------------------------
    # Set GEMINI_API_KEY in .env to enable Gemini as a fallback when the
    # primary OpenAI call fails (quota, auth, network, or any other error).
    gemini_api_key: Optional[str] = None
    # ``gemini-flash-lite-latest`` is the cheap-and-fast tier.  We prefer
    # it over the bigger ``gemini-flash-latest`` because it has a separate
    # daily-quota pool on free-tier Google AI Studio accounts — when
    # ``gemini-3.5-flash`` is exhausted by the day, the lite pool is
    # typically still available.
    gemini_model: str = "gemini-flash-lite-latest"
    # ``gemini-embedding-001`` is the new default embedding model.
    # ``text-embedding-004`` is not present on newer Google AI Studio
    # accounts.
    gemini_embedding_model: str = "gemini-embedding-001"

    # Comma-separated provider order. Unknown providers are ignored.
    # Tokens are lowercased and de-duplicated.
    llm_provider_priority: str = "openai,gemini"

    # ---- Retry knobs -------------------------------------------------------
    llm_max_retries_per_provider: int = 3
    llm_retry_min_seconds: float = 1.0
    llm_retry_max_seconds: float = 10.0

    # ---- Auth DB (Issue #22) ----------------------------------------------
    # File path for the multi-user credential store. Relative paths
    # resolve against the process's CWD, matching pydantic-settings
    # semantics for the rest of the config. The bootstrap admin
    # password is the FIRST-RUN seed value — once any user exists in
    # the table, the bootstrap is a no-op (no admin overwrite on
    # subsequent boots).
    auth_db: str = "./auth.db"
    bootstrap_admin_password: Optional[str] = None

    # ---- Production checkpointer DSN (Issue #11) --------------------------
    # Optional.  Set TELECOM_RAG_CHECKPOINT_DSN in .env to a Postgres
    # connection string to switch the production checkpointer from
    # SqliteSaver to PostgresSaver.  When unset (default), the chat
    # page uses SqliteSaver via ``get_checkpointer(":memory:")`` and
    # tests stay on the in-memory SQLite path.  This env var is
    # intentionally named ``TELECOM_RAG_*`` to follow the project
    # naming convention and to avoid colliding with any third-party
    # ``CHECKPOINT_*`` variable.
    checkpoint_dsn: Optional[str] = None

    # ---- RAGAS-in-CI offline flag (Issue #10, prefixed Issue #34) -------
    # String-flag. Set TELECOM_RAG_EVAL_OFFLINE=1 in .env (or shell) to
    # short-circuit the ragas.evaluate call. Field type is ``str`` (not
    # ``bool``) because pydantic-settings does not coerce "1"/"true"/
    # "yes"/"on" to True by default; the consumer's
    # ``_is_offline_truthy()`` helper centralizes the truthy vocabulary.
    eval_offline: Optional[str] = None

    # ---- JSONL log file (Issue #22) -------------------------------------
    # Path to the structured JSONL log file the upload / my_uploads
    # reader and the auth-gate audit log both write to. Relative paths
    # resolve against the process's CWD, matching the rest of config.
    # The ``/my_uploads`` page reads this file to render the per-user
    # upload history.
    log_file: str = "./logs/telecom_rag.jsonl"

    # ---- Cost-quota telemetry (Issue #16, prefixed Issue #34) ----------
    # ``llm_pricing_json`` — JSON catalog override for LLM pricing. When
    # unset, the bundled catalog is used.
    # ``llm_pricer`` — module path (e.g. ``litellm.cost_calculator``)
    # to a callable that returns per-token cost. When unset, the
    # bundled pricer is used.
    # ``cost_quota_usd_daily`` — daily USD spend cap across all agents.
    # When ``None``, the cost module's $10 fallback applies.
    # ``cost_quota_usd_daily_by_agent`` — per-agent daily USD cap; a
    # WARN log fires when an agent's daily total exceeds this. When
    # ``None``, no per-agent cap is enforced.
    llm_pricing_json: Optional[str] = None
    llm_pricer: Optional[str] = None
    cost_quota_usd_daily: Optional[float] = None
    cost_quota_usd_daily_by_agent: Optional[float] = None

    @property
    def provider_priority(self) -> List[str]:
        """Parsed, validated provider priority list.

        Returns the configured priority order with any provider whose API
        key is missing dropped from the tail. Order of the remaining
        providers is preserved.
        """
        seen: List[str] = []
        for token in self.llm_provider_priority.split(","):
            name = token.strip().lower()
            if name and name not in seen:
                seen.append(name)
        available: List[str] = []
        for name in seen:
            if name == "openai" and not self.openai_api_key:
                continue
            if name == "gemini" and not self.gemini_api_key:
                continue
            available.append(name)
        return available


settings = Settings()
