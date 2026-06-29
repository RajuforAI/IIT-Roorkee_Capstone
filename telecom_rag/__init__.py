"""Telecom RAG package.

Re-exports the LLM fallback layer so callers can do:

    from telecom_rag import chat_with_fallback, embed_with_fallback

And the Issue #7 multi-agent graph so callers can do:

    from telecom_rag import build_graph, TelecomState
    from telecom_rag import get_checkpointer, get_postgres_checkpointer
"""

from telecom_rag.graphs.telecom_graph import build_graph
from telecom_rag.llm import (
    ChatResult,
    EmbeddingResult,
    LLMAvailabilityError,
    ProviderCallError,
    available_providers,
    chat_with_fallback,
    embed_with_fallback,
    provider_status,
)
from telecom_rag.memory.checkpointer import get_checkpointer, get_postgres_checkpointer
from telecom_rag.schemas import TelecomState

__all__ = [
    "ChatResult",
    "EmbeddingResult",
    "LLMAvailabilityError",
    "ProviderCallError",
    "TelecomState",
    "available_providers",
    "build_graph",
    "chat_with_fallback",
    "embed_with_fallback",
    "get_checkpointer",
    "get_postgres_checkpointer",
    "provider_status",
]
