"""Retrieval tool used by agents and the Streamlit chat page.

Wraps a LangChain :class:`Chroma` vectorstore around the same
``chromadb.PersistentClient`` the ingestion pipeline writes to, and
exposes a thin retrieval API:

  - :func:`get_vectorstore` — return the LangChain ``Chroma`` wrapper
    bound to the persisted collection.
  - :func:`retrieve_chunks` — top-k retrieval with optional MMR and
    optional metadata ``where`` filter. Returns plain dicts (not
    LangChain ``Document`` objects) so downstream code doesn't have to
    depend on LangChain core types.
  - :func:`format_sources` — render the retrieved chunks as the
    numbered ``[N] source.pdf — Page X`` citation list the chat UI
    uses.

The embedding function is the same provider-fallback layer the
ingestion pipeline uses (:func:`telecom_rag.llm.embed_with_fallback`),
wrapped in a small LangChain-compatible adapter so we don't duplicate
embedding logic between write-path and read-path.

Concurrency: ChromaDB's ``PersistentClient`` is safe for concurrent
reads against an in-progress write but may briefly return stale
results mid-ingest. Callers that need strict read-after-write should
serialize around the ingest step (the Streamlit upload page does this
implicitly because ``ingest_directory`` runs synchronously on the
Streamlit thread).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from telecom_rag import embed_with_fallback
from telecom_rag.config import settings
from telecom_rag.observability.cost import Agent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embeddings adapter: provider-fallback -> LangChain Embeddings interface
# ---------------------------------------------------------------------------


class _FallbackEmbeddings(Embeddings):
    """Adapter that exposes :func:`telecom_rag.llm.embed_with_fallback`
    as a LangChain ``Embeddings`` instance.

    LangChain's ``Chroma`` wrapper calls ``embed_query(text)`` for the
    query and ``embed_documents(texts)`` for upserts; we route both
    through the same fallback dispatcher so the index and the query
    path always use the same provider priority and dimensionality.
    """

    def embed_query(self, text: str) -> List[float]:
        # embed_with_fallback returns a list[EmbeddingResult]; we want
        # the raw vector of the first (and only) result.
        # Issue #19: attribute embedding cost to the RETRIEVAL agent
        # (live query embedding, not ingest).
        results = embed_with_fallback([text], agent=Agent.RETRIEVAL)
        if not results:
            raise RuntimeError(
                "embed_with_fallback returned no vectors for query "
                f"(provider status: {settings.provider_priority})"
            )
        return list(results[0].vector)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # Issue #19: attribute the batch embedding cost to the
        # RETRIEVAL agent — this method is called from
        # ``retrieve_chunks``'s inline expansion path.
        results = embed_with_fallback(list(texts), agent=Agent.RETRIEVAL)
        if len(results) != len(texts):
            raise RuntimeError(
                f"embed_with_fallback returned {len(results)} vectors "
                f"for {len(texts)} texts — count mismatch."
            )
        return [list(r.vector) for r in results]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_vectorstore(
    persist_dir: Optional[str] = None,
    collection_name: str = "telecom_docs",
) -> Chroma:
    """Return a LangChain ``Chroma`` wrapper around the persisted
    collection at ``persist_dir`` (default: ``settings.chroma_persist_dir``).

    The wrapper shares the same underlying ``chromadb.PersistentClient``
    the ingestion pipeline writes to, so calling :func:`retrieve_chunks`
    sees chunks written by :func:`telecom_rag.ingestion.pipeline.ingest_directory`.
    """
    if persist_dir is None:
        persist_dir = settings.chroma_persist_dir

    # Ensure the directory exists.  The ingestion pipeline already
    # creates it, but get_vectorstore can be called before any
    # ingest (e.g. on a fresh checkout) and we don't want that to
    # crash with FileNotFoundError.
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=_FallbackEmbeddings(),
        persist_directory=persist_dir,
        # collection_metadata={"hnsw:space": "cosine"} matches what
        # telecom_rag.ingestion.embedder.get_or_create_collection sets,
        # so similarity scores are comparable across the read+write path.
    )


def retrieve_chunks(
    query: str,
    vectorstore: Chroma,
    k: int = 5,
    doc_category: Optional[str] = None,
    equipment_type: Optional[str] = None,
    use_mmr: bool = True,
) -> List[Dict[str, Any]]:
    """Return up to ``k`` chunks for ``query`` as plain dicts.

    Each dict has keys::

        {
            "text":         str,   # chunk text
            "source_file":  str,   # basename of the PDF
            "page_number":  int,   # 1-indexed
            "doc_category": str,   # DocCategory enum value (or "" if missing)
            "chunk_index":  int,   # 0-indexed within the document
            "score":        float | None,  # cosine distance; lower = closer.
                                         # None when MMR returns no score.
        }

    With ``use_mmr=True`` (default) we use ``max_marginal_relevance_search``
    with ``fetch_k=k*3`` per the README §6.5 retrieval configuration.
    With ``use_mmr=False`` we use ``similarity_search_with_score`` which
    gives true cosine distances.

    Empty collection or no hits returns ``[]`` (never raises).
    """
    if not query or not query.strip():
        return []
    if k <= 0:
        return []

    where: Optional[Dict[str, Any]] = None
    if doc_category is not None or equipment_type is not None:
        where = {}
        if doc_category is not None:
            where["doc_category"] = str(doc_category)
        if equipment_type is not None:
            where["equipment_type"] = str(equipment_type)

    if use_mmr:
        # MMR reduces redundancy across the top-k; fetch_k=k*3 per README §6.5.
        # max_marginal_relevance_search takes the filter via `filter=`.
        if where is not None:
            docs: List[Document] = vectorstore.max_marginal_relevance_search(
                query, k=k, fetch_k=k * 3, filter=where
            )
        else:
            docs = vectorstore.max_marginal_relevance_search(
                query, k=k, fetch_k=k * 3
            )
        return [_doc_to_chunk_dict(d, score=None) for d in docs]

    # similarity_search_with_score returns list[(Document, float)] where
    # float is cosine distance (lower = more similar) for our cosine-indexed
    # collection.
    if where is not None:
        scored = vectorstore.similarity_search_with_score(
            query, k=k, filter=where
        )
    else:
        scored = vectorstore.similarity_search_with_score(query, k=k)
    return [_doc_to_chunk_dict(d, score=s) for d, s in scored]


def _doc_to_chunk_dict(
    doc: Document, score: Optional[float]
) -> Dict[str, Any]:
    """Convert a LangChain ``Document`` + optional score to our chunk dict.

    Chunk metadata (per ``telecom_rag.schemas.TelecomMetadata``):
      source_file, doc_category, page_number, chunk_index, etc.
    Missing fields default to safe sentinels rather than KeyError —
    the caller always gets a dict with the contracted keys.
    """
    meta = doc.metadata or {}
    return {
        "text": doc.page_content or "",
        "source_file": str(meta.get("source_file", "unknown.pdf")),
        "page_number": int(meta.get("page_number", 0) or 0),
        "doc_category": str(meta.get("doc_category", "") or ""),
        "chunk_index": int(meta.get("chunk_index", 0) or 0),
        "score": score,
    }


def format_sources(chunks: List[Dict[str, Any]]) -> str:
    """Render a numbered citation list of the unique
    ``(source_file, page_number)`` pairs present in ``chunks``.

    One line per chunk, format::

        [1] sop_01.pdf — Page 3
        [2] sop_07.pdf — Page 1
        ...

    Empty input returns an empty string (not None, not a placeholder).
    Duplicate ``(source_file, page_number)`` pairs in the input are
    preserved as separate lines — the caller is responsible for
    pre-deduplication if they want uniqueness.
    """
    if not chunks:
        return ""

    lines: List[str] = []
    for idx, c in enumerate(chunks, start=1):
        src = c.get("source_file", "unknown.pdf") or "unknown.pdf"
        page = c.get("page_number", 0) or 0
        lines.append(f"[{idx}] {src} — Page {page}")
    return "\n".join(lines)


# Maximum number of source-file basenames returned in
# :func:`collection_stats`'s ``recent_sources`` list. The admin page
# uses this for a "recently indexed" panel; 10 covers a typical user
# session without overflowing the Streamlit expander.
_RECENT_SOURCES_LIMIT = 10


def collection_stats(collection) -> Dict[str, Any]:
    """Return a small JSON-serializable snapshot of the collection.

    Accepts a LangChain :class:`langchain_chroma.Chroma` wrapper. Reads
    the underlying ``chromadb`` ``Collection`` via ``collection._collection``
    to access :meth:`count` and :meth:`get` cheaply — LangChain's own
    public surface goes through the embedding function, which is both
    unnecessary for metadata-only reads and triggers provider calls.

    Returns::

        {
            "total_chunks":     int,         # raw count() of the collection
            "distinct_sources": int,         # # of unique source_file values
            "recent_sources":   List[str],   # first N source_file basenames
                                             # in chromadb's natural
                                             # retrieval order (N <= 10)
        }

    Semantics of ``recent_sources``: the ``source_file`` basename of
    each chunk returned by ``_collection.get(include=["metadatas"])``,
    in retrieval order, capped at :data:`_RECENT_SOURCES_LIMIT` (10).
    Chromadb's ``get`` returns rows in its internal id order which, for
    a collection populated by
    :func:`telecom_rag.ingestion.pipeline.ingest_directory`, is
    approximately insertion order — i.e. the basenames the user indexed
    first appear first. This is the most defensible "recent" semantic
    given that chromadb exposes no public recency index and we do not
    want to materialize a per-chunk timestamp.

    On an empty collection returns
    ``{"total_chunks": 0, "distinct_sources": 0, "recent_sources": []}``.
    Never raises: any exception from the chromadb read is logged and
    swallowed so a degraded collection still yields a renderable dict
    (the admin page surfaces "0 chunks" rather than crashing).
    """
    empty: Dict[str, Any] = {
        "total_chunks": 0,
        "distinct_sources": 0,
        "recent_sources": [],
    }
    try:
        # LangChain's Chroma wrapper exposes the underlying chromadb
        # Collection at ``_collection``. Going through it avoids
        # routing metadata reads through the embedding adapter, which
        # would be both wasteful and would fail when no LLM provider
        # is configured.
        underlying = getattr(collection, "_collection", None)
        if underlying is None:
            logger.warning(
                "collection_stats: passed object has no _collection; "
                "type=%s",
                type(collection).__name__,
            )
            return empty

        total = int(underlying.count())

        metadatas_raw = underlying.get(include=["metadatas"])
        metadatas = metadatas_raw.get("metadatas") if metadatas_raw else None
        # ``get()`` can legitimately return ``None`` or a list of
        # ``None`` for an empty collection depending on the chromadb
        # version; coerce both to an empty list.
        if not metadatas:
            sources: List[str] = []
        else:
            sources = [
                str(m.get("source_file", ""))
                for m in metadatas
                if isinstance(m, dict) and m.get("source_file")
            ]

        distinct_sources = len(set(sources))
        recent_sources = sources[:_RECENT_SOURCES_LIMIT]

        return {
            "total_chunks": total,
            "distinct_sources": distinct_sources,
            "recent_sources": recent_sources,
        }
    except Exception:  # noqa: BLE001
        # The admin page must render even if the collection is in a
        # degraded state. Log and return the empty-shape default.
        logger.exception("collection_stats: failed to read collection")
        return empty
