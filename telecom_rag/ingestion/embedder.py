"""ChromaDB collection + embedding storage.

Embedding strategy: try the configured providers in priority order via
:func:`telecom_rag.embed_with_fallback`.  The function returns the
vectors from the FIRST provider that succeeds, so all chunks in a single
batch share the same dimensionality.  When the OpenAI quota is exhausted
the call silently falls through to the Gemini embedder.

NOTE: the dimension of the returned vectors depends on which provider
actually served the request.  We pin the collection's embedding function
to ``None`` (Chroma will accept arbitrary-length vectors) and let
``HNSW`` index the result.  If you need a fixed dimension, pin it
explicitly via ``chromadb.utils.embedding_functions``.
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from telecom_rag import embed_with_fallback
from telecom_rag.observability.cost import Agent

logger = logging.getLogger(__name__)

# Default batch size for upserts.  Keep under 100 to stay within the
# typical 16MB-100MB request body limit for self-hosted ChromaDB.
DEFAULT_BATCH_SIZE = 50

# Minimum keyword-list length ChromaDB will accept (it rejects empty lists).
_MIN_KEYWORDS = 1

# Fallback keyword extraction: top-N longest alphabetic tokens from the text.
_FALLBACK_KEYWORD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{3,}")
_FALLBACK_KEYWORD_COUNT = 5


def _ensure_keywords(metadata: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Return a copy of ``metadata`` with a non-empty ``keywords`` list.

    ChromaDB rejects empty list values for metadata.  When the caller
    supplied no keywords (or supplied an empty list), we derive a few
    distinctive words from the chunk text so the field is both present
    AND meaningful for downstream filtering.
    """
    kw = metadata.get("keywords")
    if isinstance(kw, list) and len(kw) >= _MIN_KEYWORDS:
        return metadata

    # Derive keywords from the chunk text by picking the longest tokens
    # (longer words carry more semantic weight than short ones).
    if not text:
        derived: List[str] = ["telecom"]
    else:
        tokens = _FALLBACK_KEYWORD_TOKEN_RE.findall(text)
        # Preserve order of first appearance, deduplicate, sort by length desc.
        seen = set()
        ordered: List[str] = []
        for t in tokens:
            low = t.lower()
            if low in seen:
                continue
            seen.add(low)
            ordered.append(t)
        ordered.sort(key=len, reverse=True)
        derived = ordered[:_FALLBACK_KEYWORD_COUNT] or ["telecom"]

    new_meta = dict(metadata)
    new_meta["keywords"] = derived
    return new_meta


def _normalize_for_chroma(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce metadata to types ChromaDB's Rust binding will accept.

    Two constraints not enforced by Chroma's Python-side validator but
    enforced by the Rust binding on ``add()``:

    1. ``None`` values are rejected with
       ``TypeError: argument 'metadatas': Cannot convert Python object
       to MetadataValue``.  We coerce them to empty strings — they keep
       the field present for schema consistency without losing the
       filterable-key contract.
    2. ``bool`` values are technically accepted but are a known footgun
       (Python treats ``True`` as ``int``).  We leave them as-is.
    """
    out: Dict[str, Any] = {}
    for k, v in metadata.items():
        if v is None:
            out[k] = ""
        else:
            out[k] = v
    return out


def get_or_create_collection(
    persist_dir: str,
    collection_name: str = "telecom_docs",
) -> chromadb.api.models.Collection:
    """Create (or load) a persistent ChromaDB collection.

    A ``chromadb.PersistentClient`` is created at ``persist_dir``; the
    named collection is fetched or created on first use.
    """
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_texts_via_fallback(texts: List[str]) -> List[List[float]]:
    """Embed a batch via the provider-fallback layer.

    The fallback layer's contract is: return vectors from the FIRST
    provider that succeeds, all in a single call.  We rely on that.
    """
    # Issue #19: attribute embedding cost to the EMBEDDING agent
    # (Chroma ingest path — distinct from live retrieval cost).
    results = embed_with_fallback(texts, agent=Agent.EMBEDDING)
    return [list(r.vector) for r in results]


def embed_and_store(
    chunks: List[Dict[str, Any]],
    collection: chromadb.api.models.Collection,
    batch_size: int = DEFAULT_BATCH_SIZE,
    *,
    ids: Optional[List[str]] = None,
) -> int:
    """Embed a list of chunks and upsert them into the collection.

    Each input chunk dict must have at minimum ``text`` and ``metadata``
    keys.  Returns the number of chunks stored.

    ``ids`` (Issue #20): when provided, the IDs are used verbatim
    (Chroma upsert path). When ``None`` (backwards-compat default), a
    fresh ``uuid.uuid4()`` is generated per chunk — the original
    one-shot ingest path stays unchanged. The re-ingest path
    (``reingest_directory``) passes deterministic IDs from
    :func:`telecom_rag.ingestion.chunker.make_chunk_id` so Chroma's
    ``add(ids=...)`` upserts in place instead of duplicating chunks.
    """
    if not chunks:
        return 0

    stored = 0
    total = len(chunks)
    # Pre-generate IDs so retries don't double-insert.
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in range(total)]
    elif len(ids) != total:
        raise ValueError(
            f"embed_and_store: ids length ({len(ids)}) must match "
            f"chunks length ({total})"
        )

    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        documents = [c["text"] for c in batch]
        # Ensure every metadata has a non-empty keywords list before insert.
        metadatas = [
            _normalize_for_chroma(
                _ensure_keywords(c["metadata"], c.get("text", ""))
            )
            for c in batch
        ]

        embeddings = _embed_texts_via_fallback(documents)
        if len(embeddings) != len(documents):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(embeddings)} for {len(documents)} texts"
            )

        collection.add(
            ids=batch_ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        stored += len(batch)
        logger.info(
            "embed_and_store: stored batch %d-%d / %d",
            start + 1,
            start + len(batch),
            total,
        )
    return stored
