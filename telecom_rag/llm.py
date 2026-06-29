"""LLM provider layer with automatic fallback.

Every LLM call in this project should go through :func:`chat_with_fallback`
or :func:`embed_with_fallback`.  These helpers try the configured providers
in priority order (default: ``openai``, then ``gemini``) and only raise
:class:`LLMAvailabilityError` if every configured provider fails or no
provider is configured.

Provider availability is decided by API-key presence in :class:`Settings`
— see :attr:`Settings.provider_priority`.  Adding a new provider requires
implementing one ``_try_<provider>_*`` function and listing it in
``llm_provider_priority``.

Heavy provider SDKs (``openai``, ``google-genai``) are imported lazily
inside each provider function so the rest of the package keeps a light
import footprint when only a subset of providers is in use.

Cost / quota tracking (Issue #16):
    Every successful chat / embedding dispatch calls
    :func:`telecom_rag.observability.cost.record` with the token usage
    reported by the provider SDK and the wall-clock latency in
    milliseconds. Provider failures do NOT generate cost records (no
    charge for failed calls). Token usage is exposed on the
    :class:`ChatResult` / :class:`EmbeddingResult` return values so
    callers that want to do their own accounting can inspect it
    without re-reading the ledger.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from telecom_rag.config import settings
from telecom_rag.observability import cost as _cost
from telecom_rag.observability.cost import Agent, EmbeddingUsage, TokenUsage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMAvailabilityError(RuntimeError):
    """Raised when every configured LLM provider fails or none is configured."""


class ProviderCallError(RuntimeError):
    """Wraps a single failed provider call so the caller can log it cleanly."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatResult:
    """A normalized chat-completion result independent of provider.

    ``usage`` carries the token counts reported by the provider SDK
    (Issue #16). May be ``None`` if the SDK omits usage info — the
    cost ledger treats ``None`` as a zero-cost record.
    """

    text: str
    provider: str
    model: str
    usage: Optional[TokenUsage] = None


@dataclass(frozen=True)
class EmbeddingResult:
    """A normalized embedding result independent of provider.

    ``vector`` is a Python ``list[float]``.  Gemini and OpenAI return
    different dimensionality; downstream code must not assume a fixed
    dimension unless it has been pinned at index time.

    ``usage`` carries the token count reported by the provider SDK
    (Issue #16). Embeddings are input-only, so this is a single
    ``input_tokens`` count. May be ``None`` if the SDK omits it.
    """

    vector: List[float]
    provider: str
    model: str
    usage: Optional[EmbeddingUsage] = None


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


def _openai_available() -> bool:
    return bool(settings.openai_api_key)


def _try_openai_chat(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> ChatResult:
    """Single OpenAI chat attempt. Raises :class:`ProviderCallError` on failure."""
    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=settings.openai_api_key)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model or settings.llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    usage: Optional[TokenUsage] = None
    raw_usage = getattr(resp, "usage", None)
    if raw_usage is not None:
        usage = TokenUsage(
            prompt_tokens=int(getattr(raw_usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(raw_usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(raw_usage, "total_tokens", 0) or 0),
        )
    return ChatResult(
        text=text, provider="openai", model=model or settings.llm_model, usage=usage
    )


def _try_openai_embed(texts: Sequence[str], *, model: Optional[str] = None) -> List[EmbeddingResult]:
    """Single OpenAI embedding attempt. Raises :class:`ProviderCallError` on failure."""
    from openai import OpenAI  # lazy import

    if not texts:
        return []
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.embeddings.create(
        model=model or settings.embedding_model,
        input=list(texts),
    )
    used_model = model or settings.embedding_model
    # OpenAI's embedding response includes ``usage.prompt_tokens`` (input only;
    # embeddings have no completion). Surface it as EmbeddingUsage so the cost
    # ledger can charge per input token.
    usage: Optional[EmbeddingUsage] = None
    raw_usage = getattr(resp, "usage", None)
    if raw_usage is not None:
        usage = EmbeddingUsage(
            input_tokens=int(getattr(raw_usage, "prompt_tokens", 0) or 0),
        )
    return [
        EmbeddingResult(
            vector=list(item.embedding),
            provider="openai",
            model=used_model,
            usage=usage,
        )
        for item in resp.data
    ]


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------


def _gemini_available() -> bool:
    return bool(settings.gemini_api_key)


def _try_gemini_chat(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> ChatResult:
    """Single Gemini chat attempt. Raises :class:`ProviderCallError` on failure.

    Important: do NOT use ``resp.text`` on the response — newer Gemini 3.x
    models spend a portion of ``max_output_tokens`` on internal "thinking"
    and return ``content.parts == []`` once the budget is exhausted, which
    makes ``resp.text`` raise.  We read ``parts`` directly and concatenate.
    """
    from google import genai  # lazy import (Issue #8: legacy SDK reached EOL)

    client = genai.Client(api_key=settings.gemini_api_key)
    used_model = model or settings.gemini_model
    # Gemini exposes temperature / max_output_tokens via GenerateContentConfig
    # in the new SDK; system_instruction moves off the model constructor and
    # onto the config object.  ``getattr`` lets the import resolve against
    # a partial ``google.genai`` stub in unit tests where ``types`` is
    # not registered as a submodule attribute.
    genai_types = getattr(genai, "types", None)
    config: Any = None
    if genai_types is not None and hasattr(genai_types, "GenerateContentConfig"):
        config = genai_types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    resp = client.models.generate_content(
        model=used_model,
        contents=prompt,
        config=config,
    )
    # Concatenate text parts from the first candidate without using
    # resp.text (which raises when content.parts is empty).
    text_parts: list[str] = []
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            t = getattr(part, "text", None)
            if t:
                text_parts.append(t)
    text = "".join(text_parts).strip()
    if not text:
        # ``finish_reason`` is the cleanest signal when Gemini returns no text.
        reason = (
            candidates[0].finish_reason
            if candidates and getattr(candidates[0], "finish_reason", None) is not None
            else "UNKNOWN"
        )
        # Map the SDK's int to a human label when we can.
        label_map = {0: "UNSPECIFIED", 1: "STOP", 2: "MAX_TOKENS",
                     3: "SAFETY", 4: "RECITATION", 5: "OTHER", 6: "BLOCKLIST"}
        if isinstance(reason, int) and reason in label_map:
            reason = label_map[reason]
        raise ProviderCallError(f"Gemini returned empty text (finish_reason={reason})")
    # Gemini's GenerateContentResponse exposes ``usage_metadata`` with
    # ``prompt_token_count`` / ``candidates_token_count`` / ``total_token_count``.
    # Surface it as TokenUsage so the cost ledger can charge per token.
    usage: Optional[TokenUsage] = None
    raw_usage = getattr(resp, "usage_metadata", None)
    if raw_usage is not None:
        usage = TokenUsage(
            prompt_tokens=int(getattr(raw_usage, "prompt_token_count", 0) or 0),
            completion_tokens=int(getattr(raw_usage, "candidates_token_count", 0) or 0),
            total_tokens=int(getattr(raw_usage, "total_token_count", 0) or 0),
        )
    return ChatResult(text=text, provider="gemini", model=used_model, usage=usage)


def _try_gemini_embed(texts: Sequence[str], *, model: Optional[str] = None) -> List[EmbeddingResult]:
    """Single Gemini embedding attempt. Raises :class:`ProviderCallError` on failure.

    Gemini's ``embed_content`` endpoint accepts a single string or a batch
    via the ``contents`` kwarg (plural in the new SDK).  We loop single-string
    calls when given a list so the response shape is uniform with the
    OpenAI branch and per-input errors surface one at a time.
    """
    from google import genai  # lazy import (Issue #8: legacy SDK reached EOL)

    if not texts:
        return []
    client = genai.Client(api_key=settings.gemini_api_key)
    used_model = model or settings.gemini_embedding_model
    out: List[EmbeddingResult] = []
    for chunk in texts:
        # The new SDK uses plural ``contents`` and returns a list of
        # embeddings (one per input string) under ``resp.embeddings``.
        resp = client.models.embed_content(model=used_model, contents=chunk)
        embeddings = getattr(resp, "embeddings", None) or []
        vec = None
        if embeddings:
            vec = getattr(embeddings[0], "values", None)
        if vec is None:
            raise ProviderCallError("Gemini returned no embedding vector")
        # Gemini's embed_content response exposes ``usage_metadata`` with
        # ``prompt_token_count`` (input only; embeddings have no completion).
        usage: Optional[EmbeddingUsage] = None
        raw_usage = getattr(resp, "usage_metadata", None)
        if raw_usage is not None:
            usage = EmbeddingUsage(
                input_tokens=int(getattr(raw_usage, "prompt_token_count", 0) or 0),
            )
        out.append(
            EmbeddingResult(
                vector=list(vec),
                provider="gemini",
                model=used_model,
                usage=usage,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Fallback dispatcher
# ---------------------------------------------------------------------------


def _retrying() -> Retrying:
    return Retrying(
        stop=stop_after_attempt(settings.llm_max_retries_per_provider),
        wait=wait_exponential(
            multiplier=1,
            min=settings.llm_retry_min_seconds,
            max=settings.llm_retry_max_seconds,
        ),
        retry=retry_if_exception_type(ProviderCallError),
        reraise=True,
    )


def _iter_providers() -> Iterable[str]:
    """Configured provider names in priority order, skipping the unavailable."""
    for name in settings.provider_priority:
        if name == "openai" and _openai_available():
            yield "openai"
        elif name == "gemini" and _gemini_available():
            yield "gemini"
        elif name not in {"openai", "gemini"}:
            logger.warning("Unknown LLM provider '%s' in priority list — skipping.", name)


def chat_with_fallback(
    prompt: str,
    *,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    agent: Agent,
) -> ChatResult:
    """Call the configured providers in priority order until one succeeds.

    The inner per-provider call retries up to
    ``settings.llm_max_retries_per_provider`` times with exponential backoff
    before the dispatcher moves to the next provider.

    Issue #19: ``agent`` is a REQUIRED keyword-only argument so the
    cost ledger can attribute every LLM call to the agent that
    initiated it. Missed attribution surfaces as a ``TypeError`` at
    the call site — never as a silent "unknown" dashboard cell.

    Raises :class:`LLMAvailabilityError` when no provider is configured or
    every configured provider fails.
    """
    errors: list[tuple[str, Exception]] = []
    saw_provider = False
    for provider in _iter_providers():
        saw_provider = True
        attempt = _try_openai_chat if provider == "openai" else _try_gemini_chat
        start = time.monotonic()
        try:
            for retry in _retrying():
                with retry:
                    result = attempt(
                        prompt,
                        model=model,
                        system=system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    latency_ms = (time.monotonic() - start) * 1000.0
                    logger.info(
                        "LLM call succeeded via %s in %.2fs",
                        provider,
                        latency_ms / 1000.0,
                    )
                    # Issue #16: cost / quota tracking. Provider SDKs
                    # always populate ``result.usage`` (TokenUsage or
                    # None). A ``None`` usage logs as a zero-cost record.
                    # Issue #19: thread ``agent`` into the cost record
                    # so the per-agent dashboard cell is correct.
                    if result.usage is not None:
                        _cost.record(
                            result.usage,
                            result.model,
                            result.provider,
                            latency_ms=latency_ms,
                            agent=agent,
                        )
                    return result
        except Exception as exc:  # noqa: BLE001
            errors.append((provider, exc))
            logger.warning(
                "LLM provider %s failed after retries (%.2fs): %s",
                provider,
                time.monotonic() - start,
                exc,
            )
            continue
    if not saw_provider:
        raise LLMAvailabilityError(
            "No LLM provider is configured. Set OPENAI_API_KEY and/or GEMINI_API_KEY "
            "in the environment."
        )
    summary = "; ".join(f"{name}: {exc}" for name, exc in errors)
    raise LLMAvailabilityError(f"All configured LLM providers failed — {summary}")


def embed_with_fallback(
    texts: Sequence[str],
    *,
    model: Optional[str] = None,
    agent: Agent,
) -> List[EmbeddingResult]:
    """Embed a batch of texts using the configured providers in priority order.

    Returns vectors from the FIRST provider that succeeds.  All vectors in
    the returned list come from the same provider (and therefore the same
    dimensionality); callers that need a fixed dimension must verify
    ``result[0].model`` and pin the index accordingly.

    Issue #19: ``agent`` is a REQUIRED keyword-only argument so the
    cost ledger can attribute every embedding call to the agent
    that initiated it. Most callers pass ``Agent.EMBEDDING`` (the
    Chroma ingest path) or ``Agent.RETRIEVAL`` (the live retrieval
    path); neither value is implied from the call site.
    """
    if not texts:
        return []
    errors: list[tuple[str, Exception]] = []
    saw_provider = False
    for provider in _iter_providers():
        saw_provider = True
        attempt = _try_openai_embed if provider == "openai" else _try_gemini_embed
        start = time.monotonic()
        try:
            for retry in _retrying():
                with retry:
                    out = attempt(texts, model=model)
                    latency_ms = (time.monotonic() - start) * 1000.0
                    logger.info(
                        "Embedding call succeeded via %s in %.2fs (%d vectors)",
                        provider,
                        latency_ms / 1000.0,
                        len(out),
                    )
                    # Issue #16: cost / quota tracking. Charge one
                    # record per provider response (the batch shares
                    # a single usage block on OpenAI; we divide the
                    # tokens evenly across the batch for clarity in
                    # the ledger snapshot).
                    # Issue #19: thread ``agent`` into the cost record
                    # so the per-agent dashboard cell is correct.
                    if out and out[0].usage is not None and provider == "openai":
                        # OpenAI returns ONE usage for the whole batch.
                        per_input = max(
                            1, out[0].usage.input_tokens // max(1, len(out))
                        )
                        _cost.record(
                            EmbeddingUsage(input_tokens=per_input * len(out)),
                            out[0].model,
                            out[0].provider,
                            latency_ms=latency_ms,
                            agent=agent,
                        )
                    elif out and provider == "gemini":
                        # Gemini returns one usage per input. Charge
                        # one record per input for fidelity.
                        for emb in out:
                            if emb.usage is not None:
                                _cost.record(
                                    emb.usage,
                                    emb.model,
                                    emb.provider,
                                    latency_ms=latency_ms / max(1, len(out)),
                                    agent=agent,
                                )
                    return out
        except Exception as exc:  # noqa: BLE001
            errors.append((provider, exc))
            logger.warning(
                "Embedding provider %s failed after retries (%.2fs): %s",
                provider,
                time.monotonic() - start,
                exc,
            )
            continue
    if not saw_provider:
        raise LLMAvailabilityError(
            "No embedding provider is configured. Set OPENAI_API_KEY and/or GEMINI_API_KEY "
            "in the environment."
        )
    summary = "; ".join(f"{name}: {exc}" for name, exc in errors)
    raise LLMAvailabilityError(f"All configured embedding providers failed — {summary}")


# ---------------------------------------------------------------------------
# Provider introspection helpers
# ---------------------------------------------------------------------------


def available_providers() -> List[str]:
    """Return the list of providers that have credentials and are enabled."""
    return list(_iter_providers())


def provider_status() -> dict[str, Any]:
    """Return a diagnostic dict describing which providers are wired up."""
    priority = settings.provider_priority
    return {
        "priority": priority,
        "openai": {
            "configured": _openai_available(),
            "in_priority": "openai" in priority,
            "chat_model": settings.llm_model if _openai_available() else None,
            "embedding_model": settings.embedding_model if _openai_available() else None,
        },
        "gemini": {
            "configured": _gemini_available(),
            "in_priority": "gemini" in priority,
            "chat_model": settings.gemini_model if _gemini_available() else None,
            "embedding_model": settings.gemini_embedding_model if _gemini_available() else None,
        },
    }
