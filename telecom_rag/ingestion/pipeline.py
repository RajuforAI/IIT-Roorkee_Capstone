"""End-to-end ingestion pipeline: parse -> chunk -> embed -> store.

The pipeline is intentionally thin: it orchestrates the four ingestion
stages and records per-file errors so a single bad file never aborts the
whole batch.

Doc-category inference: by default we look at the filename prefix
(``sop_*`` -> SOP, ``manual_*`` -> MANUAL, ``spec_*`` -> SPEC,
``troubleshooting_*`` -> TROUBLESHOOTING, anything else -> GENERAL).
Callers can override this by passing ``doc_category`` explicitly.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from telecom_rag.ingestion.chunker import chunk_pages, make_chunk_id
from telecom_rag.ingestion.docling_parser import parse_pdf
from telecom_rag.ingestion.embedder import embed_and_store, get_or_create_collection
from telecom_rag.schemas import DocCategory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Issue #20 — module-level re-scan state (admin page reads from here)
# ---------------------------------------------------------------------------
#
# ``LAST_RESCAN`` is the most recent completed re-scan summary (or None
# if no re-scan has ever completed in this process). The admin page's
# "Last re-scan" sub-section renders this dict. Stored at module level
# because Streamlit's ``@st.cache_resource`` decorators on individual
# admin-page sections create independent function-local caches that
# would each need a seam — module-level state is the simplest shared
# read surface.
#
# This is intentionally in-process only (matches the CostLedger pattern
# in Issue #16). For durable per-call history, query CloudWatch Logs
# Insights with ``event=ingest_completed``.

LAST_RESCAN: Optional[Dict[str, Any]] = None


def get_last_rescan() -> Optional[Dict[str, Any]]:
    """Return the most recent ``reingest_directory(apply=True)`` summary.

    Returns ``None`` if no re-scan has ever completed in this process.
    Admin page ("Last re-scan" sub-section) calls this. Tests can
    monkeypatch or reset via :func:`reset_last_rescan`.
    """
    return LAST_RESCAN


def reset_last_rescan() -> None:
    """Reset the module-level ``LAST_RESCAN`` state. Test helper."""
    global LAST_RESCAN
    LAST_RESCAN = None


_PREFIX_TO_CATEGORY: Dict[str, DocCategory] = {
    "sop": DocCategory.SOP,
    "manual": DocCategory.MANUAL,
    "spec": DocCategory.SPEC,
    "troubleshooting": DocCategory.TROUBLESHOOTING,
}


def infer_doc_category(source_file: str) -> DocCategory:
    """Map a source filename to a :class:`DocCategory` enum value.

    The first underscore-separated token of the basename (e.g.
    ``sop_01.pdf`` -> ``sop``) is looked up against the known prefixes.
    Unknown prefixes fall back to :data:`DocCategory.GENERAL`.
    """
    name = Path(source_file).stem.lower()
    head = name.split("_", 1)[0]
    return _PREFIX_TO_CATEGORY.get(head, DocCategory.GENERAL)


def ingest_file(
    file_path: str,
    collection,
    doc_category: Optional[str] = None,
) -> int:
    """Parse, chunk, embed, and store a single PDF.

    Returns the number of chunks stored for this file.  Raises on hard
    failures (unreadable PDF, zero pages after filtering) — callers in
    :func:`ingest_directory` catch and record these.
    """
    path = Path(file_path)
    pages = parse_pdf(str(path))
    if not pages:
        raise ValueError(f"No usable pages extracted from {path.name}")

    category = doc_category if doc_category is not None else infer_doc_category(path.name)
    # Use the file basename as the title; real systems would extract from
    # the PDF metadata.  This is good enough for the synthetic corpus.
    title = path.stem.replace("_", " ").title()
    metadata_extras = {
        "title": title,
        "source_file": path.name,
    }

    chunks = chunk_pages(pages, doc_category=category, metadata_extras=metadata_extras)
    if not chunks:
        raise ValueError(f"No chunks produced from {path.name}")

    return embed_and_store(chunks, collection)


def ingest_directory(
    dir_path: str,
    persist_dir: str,
    collection_name: str = "telecom_docs",
    file_allowlist: List[str] | None = None,
) -> Dict[str, Any]:
    """Ingest every ``*.pdf`` file in ``dir_path`` into a Chroma collection.

    Returns a summary dict::

        {
            "total_files":    int,
            "total_chunks":   int,
            "errors":         [{"file": str, "error": str}, ...],
            "chunks_by_file": {filename: int, ...},  # successful files only
        }

    ``chunks_by_file`` is keyed by the file basename (the same convention
    as ``errors[].file``). It contains ONLY files that successfully
    ingested -- a failed file contributes its error to ``errors[]``
    rather than to ``chunks_by_file``. The invariant
    ``sum(chunks_by_file.values()) == total_chunks`` holds for the
    successful subset of the batch.

    Issue #15: ``file_allowlist`` is an optional list of basenames to
    ingest. When provided, files in ``dir_path`` whose basename is
    NOT in the allowlist are skipped silently — they remain on
    local disk for the caller to clean up, but no Chroma / LangSmith
    work is performed for them. Used by the upload page to skip
    files that were already blocked by the pre-upload PII/secret
    scanner (``upload._ingest_upload``), so the per-file table
    doesn't surface duplicate "failed" rows for blocked files.
    Pass ``None`` (the default) to keep the original
    ingest-everything behavior.
    """
    directory = Path(dir_path)
    all_pdfs = sorted(p for p in directory.glob("*.pdf") if p.is_file())
    if file_allowlist is not None:
        allowset = set(file_allowlist)
        pdfs = [p for p in all_pdfs if p.name in allowset]
    else:
        pdfs = all_pdfs

    collection = get_or_create_collection(persist_dir=persist_dir, collection_name=collection_name)

    total_files = 0
    total_chunks = 0
    errors: List[Dict[str, str]] = []
    chunks_by_file: Dict[str, int] = {}

    for pdf_path in pdfs:
        try:
            n = ingest_file(str(pdf_path), collection)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to ingest %s", pdf_path.name)
            errors.append({"file": pdf_path.name, "error": str(exc)})
            continue
        total_files += 1
        total_chunks += n
        chunks_by_file[pdf_path.name] = n
        logger.info("Ingested %s: %d chunks", pdf_path.name, n)

    return {
        "total_files": total_files,
        "total_chunks": total_chunks,
        "errors": errors,
        "chunks_by_file": chunks_by_file,
    }


# ---------------------------------------------------------------------------
# Issue #20 — reingest_directory: idempotent re-ingest with diff + dry-run
# ---------------------------------------------------------------------------


def _existing_source_files(collection) -> set[str]:
    """Return the set of ``source_file`` basenames currently indexed in
    ``collection``. Used by the reingest diff to detect STALE files."""
    if collection.count() == 0:
        return set()
    rows = collection.get(include=["metadatas"])
    return {
        m.get("source_file") for m in (rows.get("metadatas") or [])
        if m.get("source_file")
    }


def _count_for_source(collection, source_basename: str) -> int:
    """Return how many chunks in ``collection`` have
    ``metadata.source_file == source_basename``. Used by the diff to
    record ``old_chunks`` per stale file."""
    if collection.count() == 0:
        return 0
    rows = collection.get(where={"source_file": source_basename}, include=[])
    return len(rows.get("ids", []))


def _diff_against_collection(
    on_disk: List[Path],
    collection,
    file_allowlist: Optional[List[str]] = None,
) -> Tuple[
    List[Dict[str, Any]],  # new
    List[Dict[str, Any]],  # changed
    List[Dict[str, Any]],  # stale
    List[Dict[str, Any]],  # unchanged
    List[Dict[str, str]],  # errors (files that can't be parsed at all)
]:
    """Compute the (new, changed, stale, unchanged) partition.

    - **new**: on disk, not in collection
    - **changed**: on disk AND in collection; ``first_120_chars`` of
      chunk[0] differs from the first chunk in the collection for
      that source
    - **stale**: in collection but not on disk (or excluded by allowlist)
    - **unchanged**: on disk AND in collection; ``first_120_chars`` matches
    - **errors**: on disk BUT unparseable (empty file, corrupt PDF,
      etc.) — recorded separately so the diff isn't silent about
      files the pipeline can't ingest. The apply path then surfaces
      these as ``event=ingest_file_failed`` records.

    ``est_chunks`` for new/changed is an ESTIMATE based on page count
    (chunk count is page-bounded for the synthetic corpus).
    """
    if file_allowlist is not None:
        allowset = set(file_allowlist)
        on_disk_paths = [p for p in on_disk if p.name in allowset]
    else:
        on_disk_paths = list(on_disk)

    indexed = _existing_source_files(collection)

    # Read existing metadata to detect content edits via the first chunk's
    # text prefix. Chroma stores the chunk text in ``documents``; we
    # only need the first 120 chars per source_file for the seal check.
    indexed_first_chunk_prefix: Dict[str, str] = {}
    if collection.count() > 0:
        rows = collection.get(include=["documents", "metadatas"])
        for doc, meta in zip(
            rows.get("documents") or [], rows.get("metadatas") or []
        ):
            src = meta.get("source_file") if meta else None
            if not src or src in indexed_first_chunk_prefix:
                continue
            indexed_first_chunk_prefix[src] = (doc or "")[:120]

    new_entries: List[Dict[str, Any]] = []
    changed_entries: List[Dict[str, Any]] = []
    unchanged_entries: List[Dict[str, Any]] = []
    error_entries: List[Dict[str, str]] = []

    for pdf_path in on_disk_paths:
        name = pdf_path.name
        # Read first chunk's text from disk to compute the on-disk seal.
        try:
            pages = parse_pdf(str(pdf_path))
        except Exception as exc:  # noqa: BLE001
            # Unparseable file (empty / corrupt). Record it as an error
            # so the diff isn't silent; the apply path then emits an
            # event=ingest_file_failed record for it.
            error_entries.append({"file": name, "error": str(exc)})
            continue
        if not pages:
            error_entries.append({
                "file": name,
                "error": "no usable pages extracted",
            })
            continue
        # Take the first page's text as a representative seal. The
        # chunker will produce more chunks, but the diff only needs
        # a stable content signature to detect edits.
        first_text = (pages[0]["text"] or "")[:120]

        if name not in indexed:
            new_entries.append({"file": name, "est_chunks": len(pages)})
            continue
        indexed_seal = indexed_first_chunk_prefix.get(name)
        if indexed_seal is None:
            # Indexed but no seal (e.g. legacy UUID-only chunks).
            # Treat as changed so a re-scan picks them up cleanly.
            changed_entries.append({
                "file": name,
                "old_chunks": _count_for_source(collection, name),
                "est_chunks": len(pages),
            })
            continue
        if first_text != indexed_seal:
            old_chunks = _count_for_source(collection, name)
            changed_entries.append({
                "file": name,
                "old_chunks": old_chunks,
                "est_chunks": len(pages),
            })
        else:
            unchanged_entries.append({
                "file": name,
                "chunks": _count_for_source(collection, name),
            })

    # Stale: indexed but not on disk.
    on_disk_basename_set = {p.name for p in on_disk_paths}
    stale_entries: List[Dict[str, Any]] = []
    for src in sorted(indexed):
        if src not in on_disk_basename_set:
            stale_entries.append({
                "file": src,
                "old_chunks": _count_for_source(collection, src),
            })

    return new_entries, changed_entries, stale_entries, unchanged_entries, error_entries


def reingest_directory(
    dir_path: str,
    persist_dir: str,
    *,
    apply: bool = False,
    collection_name: str = "telecom_docs",
    file_allowlist: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Re-ingest a directory of PDFs into the Chroma collection idempotently.

    **Contract (Issue #20 AC1 + AC2):**

    - Returns a diff dict in both modes; the dict's ``applied`` flag
      tells the caller which mode was used.
    - When ``apply=False`` (the default — dry-run), no writes happen.
      This is the safe mode for operators to inspect what would change.
    - When ``apply=True``, deletes stale chunks per file via
      ``collection.delete(where={"source_file": X})`` BEFORE re-embedding
      the file. The collection never holds both old and new chunks for
      the same source.
    - Deterministic chunk IDs (``make_chunk_id``) are used so re-ingest
      upserts in place instead of duplicating chunks.

    Diff shape (exact, per spec AC2 + Issue #20 amend)::

        {
            "new":       [{"file": str, "est_chunks": int}, ...],
            "changed":   [{"file": str, "old_chunks": int, "est_chunks": int}, ...],
            "stale":     [{"file": str, "old_chunks": int}, ...],
            "unchanged": [{"file": str, "chunks": int}, ...],
            "totals": {"new": int, "changed": int, "stale": int, "unchanged": int,
                       "new_chunks_est": int, "stale_chunks": int,
                       "failed": int},
            "applied":  bool,
        }

    Concurrency note: Chroma's PersistentClient is safe for concurrent
    reads against an in-progress write. Operators running this against
    a live Streamlit instance should expect transient partial retrieval
    results during the re-scan window.
    """
    directory = Path(dir_path)
    all_pdfs = sorted(p for p in directory.glob("*.pdf") if p.is_file())

    collection = get_or_create_collection(
        persist_dir=persist_dir, collection_name=collection_name
    )

    new, changed, stale, unchanged, errors = _diff_against_collection(
        all_pdfs, collection, file_allowlist=file_allowlist
    )

    totals = {
        "new": len(new),
        "changed": len(changed),
        "stale": len(stale),
        "unchanged": len(unchanged),
        "new_chunks_est": sum(e["est_chunks"] for e in new + changed),
        "stale_chunks": sum(e["old_chunks"] for e in stale),
        "failed": len(errors),
    }

    diff: Dict[str, Any] = {
        "new": new,
        "changed": changed,
        "stale": stale,
        "unchanged": unchanged,
        "totals": totals,
        "applied": apply,
    }

    if not apply:
        logger.info(
            "reingest_directory: dry-run (no writes)",
            extra={
                "event": "ingest_preview",
                "source_dir": str(directory),
                "persist_dir": persist_dir,
                "collection_name": collection_name,
                "totals": totals,
            },
        )
        return diff

    # ---- apply=True path ----
    batch_id = uuid.uuid4().hex[:12]
    started_at_ms = time.monotonic()
    logger.info(
        "reingest_directory: apply started",
        extra={
            "event": "ingest_started",
            "batch_id": batch_id,
            "source_dir": str(directory),
            "persist_dir": persist_dir,
            "totals": totals,
        },
    )

    # Step 0: surface parse failures from the diff as ``ingest_file_failed``
    # events. The diff loop already records them in ``errors``; the apply
    # path now mirrors that into the structured-log stream so CloudWatch
    # and the admin page see the failure alongside successful events.
    for err in errors:
        logger.info(
            "reingest_directory: file failed (parse)",
            extra={
                "event": "ingest_file_failed",
                "batch_id": batch_id,
                "file": err["file"],
                "error": err["error"],
            },
        )

    # Step 1: delete stale chunks (per file, atomic w.r.t. source_file).
    for entry in stale:
        try:
            collection.delete(where={"source_file": entry["file"]})
            logger.info(
                "reingest_directory: stale chunks deleted",
                extra={
                    "event": "ingest_stale_chunks_deleted",
                    "batch_id": batch_id,
                    "file": entry["file"],
                    "deleted_chunks": entry["old_chunks"],
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "reingest_directory: failed to delete stale chunks for %s",
                entry["file"],
            )
            logger.info(
                "reingest_directory: file failed",
                extra={
                    "event": "ingest_file_failed",
                    "batch_id": batch_id,
                    "file": entry["file"],
                    "error": f"delete-stale failed: {exc}",
                },
            )

    # Step 2: re-ingest new + changed files (deterministic IDs).
    for entry in new + changed:
        file_name = entry["file"]
        # Find the on-disk path for this entry.
        pdf_path = next((p for p in all_pdfs if p.name == file_name), None)
        if pdf_path is None:
            continue
        try:
            chunks = ingest_file_with_deterministic_ids(
                str(pdf_path), collection
            )
            logger.info(
                "reingest_directory: file done",
                extra={
                    "event": "ingest_file_done",
                    "batch_id": batch_id,
                    "file": file_name,
                    "chunks": chunks,
                    "action": "changed" if entry in changed else "new",
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "reingest_directory: file failed",
                exc,
            )
            logger.info(
                "reingest_directory: file failed",
                extra={
                    "event": "ingest_file_failed",
                    "batch_id": batch_id,
                    "file": file_name,
                    "error": str(exc),
                },
            )

    duration_ms = int((time.monotonic() - started_at_ms) * 1000)
    logger.info(
        "reingest_directory: apply completed",
        extra={
            "event": "ingest_completed",
            "batch_id": batch_id,
            "totals": totals,
            "duration_ms": duration_ms,
        },
    )

    # Update module-level state for the admin page.
    global LAST_RESCAN
    LAST_RESCAN = {
        "batch_id": batch_id,
        "totals": totals,
        "duration_ms": duration_ms,
        "applied": True,
        "source_dir": str(directory),
        "completed_at": time.time(),
    }

    return diff


def ingest_file_with_deterministic_ids(
    file_path: str,
    collection,
    doc_category: Optional[str] = None,
) -> int:
    """Issue #20: parse + chunk + embed + store a single PDF with
    deterministic IDs (one ID per chunk, derived from
    :func:`make_chunk_id`).

    Returns the number of chunks stored. Raises on hard failures.
    This is the re-ingest path's per-file primitive.
    """
    path = Path(file_path)
    pages = parse_pdf(str(path))
    if not pages:
        raise ValueError(f"No usable pages extracted from {path.name}")

    category = (
        doc_category if doc_category is not None
        else infer_doc_category(path.name)
    )
    title = path.stem.replace("_", " ").title()
    metadata_extras = {
        "title": title,
        "source_file": path.name,
    }
    chunks = chunk_pages(pages, doc_category=category, metadata_extras=metadata_extras)
    if not chunks:
        raise ValueError(f"No chunks produced from {path.name}")

    # Deterministic IDs — one per chunk, stable across re-ingests.
    chunk_ids: List[str] = [
        make_chunk_id(c["source_file"], c["chunk_index"], c["text"])
        for c in chunks
    ]
    return embed_and_store(chunks, collection, ids=chunk_ids)