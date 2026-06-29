"""Page -> chunk splitter.

Wraps LangChain's :class:`RecursiveCharacterTextSplitter` with the chunk
size and overlap configured on :class:`Settings`, and enriches each
chunk with the :class:`TelecomMetadata` fields required downstream.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from telecom_rag.config import settings
from telecom_rag.schemas import DocCategory


def make_chunk_id(source_file: str, chunk_index: int, text: str) -> str:
    """Produce a deterministic 24-hex-char chunk ID for the re-ingest path.

    Format:
        sha256(f"{source_file}|{chunk_index}|{text[:120]}").hexdigest()[:24]

    Three reasons this shape (vs. a full-content hash):

    1. **Bounded length.** 24 hex chars = 96 bits — collision-resistant
       within a 100k-chunk collection (birthday bound ~2^48), short
       enough to fit in the Chroma ID field comfortably. A full
       64-hex-char SHA-256 wastes index space; 16 hex chars (64 bits)
       collides at ~65k chunks.
    2. **``source_file`` prefix in the hash inputs.** A file rename
       produces new IDs even if the content is identical — the old ID
       space is freed via the delete-by-source step. This prevents
       "renamed file + new chunks + old chunks still present" drift.
    3. **First-120-chars prefix as a content seal.** Catches in-place
       edits to the PDF without paying the full content hash cost.
       False negatives are acceptable here — a missed edit triggers a
       re-embed (correct behavior); a false positive (different content
       with same first-120-chars) is vanishingly rare for telecom docs.

    The re-ingest path (``reingest_directory``) is the ONLY consumer.
    ``ingest_directory`` keeps using UUID4 IDs (Issue #20 explicitly
    opts NOT to migrate the existing collection). This split keeps the
    contract simple: deterministic IDs are scoped to the re-ingest
    flow; UUIDs continue to work for one-shot uploads.
    """
    payload = f"{source_file}|{chunk_index}|{text[:120]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def _resolve_doc_category(value: Any) -> str:
    """Coerce the doc_category argument to its canonical string value.

    Accepts a :class:`DocCategory` member, a string, or ``None``
    (defaults to :data:`DocCategory.GENERAL`).
    """
    if value is None:
        return DocCategory.GENERAL.value
    if isinstance(value, DocCategory):
        return value.value
    text = str(value).strip().lower()
    if not text:
        return DocCategory.GENERAL.value
    # Accept either the enum name ("SOP") or its value ("sop").
    for member in DocCategory:
        if text == member.name.lower() or text == member.value:
            return member.value
    # Fall back to GENERAL — better to store something than to crash.
    return DocCategory.GENERAL.value


def chunk_pages(
    pages: List[Dict[str, Any]],
    doc_category: str = DocCategory.GENERAL.value,
    metadata_extras: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Split parsed pages into chunks with TelecomMetadata attached.

    Each chunk dict::

        {
            "text":         str,
            "metadata":     dict,    # TelecomMetadata as plain dict
            "chunk_index":  int,     # 0-indexed within the document
            "total_chunks": int,     # total chunks for the document
            "source_file":  str,     # basename of the source PDF
            "page_number":  int,     # page the chunk text came from
        }

    ``doc_category`` may be a string or :class:`DocCategory` member.  The
    canonical string form is stored in the metadata.
    """
    metadata_extras = dict(metadata_extras or {})
    category_value = _resolve_doc_category(doc_category)
    title = metadata_extras.pop("title", None) or "Untitled"
    source_file = metadata_extras.pop("source_file", None) or "unknown.pdf"
    equipment_type = metadata_extras.get("equipment_type")
    protocol = metadata_extras.get("protocol")
    vendor = metadata_extras.get("vendor")
    keywords = metadata_extras.get("keywords") or []
    created_at = metadata_extras.get("created_at") or datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")

    splitter = _build_splitter()

    # First pass: split every page, remembering which page each chunk came from.
    raw_chunks: List[Dict[str, Any]] = []
    for page in pages:
        page_number = page["page_number"]
        for piece in splitter.split_text(page["text"]):
            cleaned = piece.strip()
            if not cleaned:
                continue
            raw_chunks.append({"text": cleaned, "page_number": page_number})

    total = len(raw_chunks)
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_chunks):
        metadata = {
            "source_file": source_file,
            "doc_category": category_value,
            "title": title,
            "page_number": item["page_number"],
            "chunk_index": idx,
            "total_chunks": total,
            "equipment_type": equipment_type,
            "protocol": protocol,
            "vendor": vendor,
            "keywords": list(keywords),
            "created_at": created_at,
        }
        out.append(
            {
                "text": item["text"],
                "metadata": metadata,
                "chunk_index": idx,
                "total_chunks": total,
                "source_file": source_file,
                "page_number": item["page_number"],
            }
        )
    return out
