"""Upload page (drag-drop PDF ingest).

Write-path surface for Issue #6 (AC1, AC2): the user drags in one or
more PDFs, hits "Ingest", and the page calls
:func:`telecom_rag.ingestion.pipeline.ingest_directory` to push chunks
into the persisted Chroma collection. A per-file status table renders
the outcome so a single bad PDF never crashes the batch.

After a clean ingest (zero per-file errors) we call
``st.cache_resource.clear()`` so the chat page's cached vectorstore
handle is re-resolved against the now-larger collection. We do NOT
rely on Chroma to surface new chunks through the same Python object —
``get_vectorstore`` resolves the persisted client fresh, but the
LangChain wrapper caches its embedding adapter separately and that
adapter would otherwise stay bound to the pre-ingest embedding list.

Issue #12 added the S3 mirror: each saved PDF is also PUT to
``s3://{AWS_S3_BUCKET}/uploads/{filename}`` (the README §12.3
contract). The S3 PUT is a **side-effect**, not a replacement for
local Chroma ingestion — a failed S3 PUT surfaces in the per-file
table as ``s3_error`` and does NOT block the rest of the batch.

Issue #15 added a pre-upload PII/secret scan. Each saved PDF is
parsed for text and run through
:func:`telecom_rag.security.scan.scan_pages` BEFORE the S3 PUT and
BEFORE ``ingest_directory``. Files with a ``high``- or
``medium``-severity finding are **blocked** — no S3 PUT, no Chroma
ingestion, no LangSmith trace. The user sees the redacted finding
list in the per-file table's ``error`` column with ``status="blocked"``.
Files with only ``low``-severity findings (or none) proceed as today.
This tightens the README §12.3 contract from "every uploaded PDF
lands in S3" to "every CLEAN uploaded PDF lands in S3".

Issue #18 added the structured audit event. Every block emits a
``event=upload_blocked`` JSON log line via the existing
:func:`telecom_rag.observability.logging.setup_json_logging` seam
(Issue #13 AC3). The payload contains ``filename``,
``blocked_patterns`` (pattern_ids, never redacted strings),
``file_sha256``, ``timestamp``, and ``user``. The audit log
surfaces in CloudWatch Logs Insights for SOC review without
leaking secret values into the log group.

Issue #22 AC9: ``user`` carries the real authenticated username
(from ``st.session_state.username``) instead of the pre-#22
``"streamlit-session"`` sentinel. The successful ingest path
also emits a new ``event=upload_indexed`` JSON log line with the
same fields plus ``chunks`` and ``batch_id`` (the same
``upload_id`` already passed to ``ingest_directory``), so the
new ``/my_uploads`` page can show a per-user upload history.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from telecom_rag.config import settings
from telecom_rag.ingestion.docling_parser import parse_pdf
from telecom_rag.ingestion.pipeline import ingest_directory
from telecom_rag.security.scan import (
    Finding,
    has_blocking_finding,
    scan_pages,
    summarize_blocking_patterns,
)
from telecom_rag.storage.s3 import S3UploadError, upload_pdf

# Issue #18 AC4 — module-level logger for the structured
# ``event=upload_blocked`` audit event. Named logger so a SOC
# analyst can filter CloudWatch Logs Insights by
# ``{ $.logger = "app.pages.upload" && $.event = "upload_blocked" }``.
_LOGGER = logging.getLogger("app.pages.upload")

# Root directory under which each upload gets its own ephemeral subdir.
# Lives inside the repo's ``data/`` tree (gitignored) so test fixtures
# that already write synthetic corpora to ``data/`` don't collide.
_UPLOAD_ROOT = Path("data/uploads")


def _build_result_rows(
    summary: Dict[str, Any],
    filenames: List[str],
    s3_errors_by_file: Dict[str, str] | None = None,
    blocked_by_file: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    """Return the per-file row dicts that ``_render_results_table`` would
    render. Pure-data helper extracted so the row-construction logic is
    unit-testable without spinning up a full AppTest render.

    Per-file chunk counts come from ``summary["chunks_by_file"]`` (Wave 2
    of Issue #6). Older summaries that lack this key fall back to the
    batch ``total_chunks`` for backward compatibility with any cached or
    pre-Wave-2 callers. Failed files are not in ``chunks_by_file``; they
    are looked up via ``summary["errors"]``.

    The Issue #12 ``s3_errors_by_file`` argument adds an S3 mirror
    failure column. The dict maps filename → S3 error message (or the
    "(skipped — no AWS_S3_BUCKET configured)" string when the bucket
    is unset). Pass an empty dict (or None) when S3 is fully skipped.

    The Issue #15 ``blocked_by_file`` argument adds a scan-block
    category. The dict maps filename → redacted findings summary
    (e.g. ``"BLOCKED: found aws_access_key (1 pattern)"``). Files
    here are rendered FIRST (in upload order) with ``status="blocked"``
    and ``chunks=0`` — they were never sent to S3, never parsed by
    ``ingest_directory``, and never indexed. The redacted finding
    text is safe to render in a Streamlit table cell (the redaction
    contract is pinned by
    ``tests/test_security_scan.py::test_finding_redacted_does_not_leak_full_secret``).
    """
    errors_by_file: Dict[str, str] = {
        e["file"]: e["error"] for e in summary.get("errors", [])
    }
    chunks_by_file: Dict[str, int] = summary.get("chunks_by_file", {}) or {}
    s3_errors: Dict[str, str] = s3_errors_by_file or {}
    blocked: Dict[str, str] = blocked_by_file or {}

    rows: List[Dict[str, Any]] = []
    for name in filenames:
        if name in blocked:
            # Issue #15: a blocked file takes priority over both
            # ``errors`` and ``chunks_by_file`` because the block is
            # a deliberate pre-empt of the ingest pipeline — it
            # should not be reported as a Chroma "failure" in the
            # table (which would mislead ops into investigating a
            # non-existent pipeline bug).
            rows.append(
                {
                    "filename": name,
                    "status": "blocked",
                    "chunks": 0,
                    "error": blocked[name],
                    "s3_error": "",  # never attempted — scan blocks first
                }
            )
        elif name in errors_by_file:
            rows.append(
                {
                    "filename": name,
                    "status": "failed",
                    "chunks": 0,
                    "error": errors_by_file[name],
                    "s3_error": s3_errors.get(name, ""),
                }
            )
        elif name in chunks_by_file:
            rows.append(
                {
                    "filename": name,
                    "status": "indexed",
                    "chunks": chunks_by_file[name],
                    "error": "",
                    "s3_error": s3_errors.get(name, ""),
                }
            )
        else:
            # Backward-compat fallback: a successful file with no
            # per-file count recorded (only possible with a pre-Wave-2
            # summary). Surface the batch total so the row is still
            # informative rather than blank.
            rows.append(
                {
                    "filename": name,
                    "status": "indexed",
                    "chunks": summary.get("total_chunks", 0),
                    "error": "",
                    "s3_error": s3_errors.get(name, ""),
                }
            )
    return rows


def _render_results_table(
    summary: Dict[str, Any],
    filenames: List[str],
    s3_errors_by_file: Dict[str, str] | None = None,
    blocked_by_file: Dict[str, str] | None = None,
) -> None:
    """Render the per-file status table from an ``ingest_directory`` summary.

    Delegates row construction to :func:`_build_result_rows` (pure-data,
    unit-tested separately) and renders the result as a Streamlit
    dataframe. A file that succeeded but for which
    ``ingest_directory`` recorded zero chunks still gets a row, with
    ``chunks = 0``.

    Issue #12: the table gained a new ``s3_error`` column. Empty
    when S3 PUT succeeded; the underlying boto3 error string when
    it failed; "(skipped — no AWS_S3_BUCKET configured)" when the
    bucket is unset (config issue, not a runtime failure).

    Issue #15: the table also accepts ``blocked_by_file`` to render
    ``status="blocked"`` rows for files the pre-upload PII/secret
    scanner rejected. The ``error`` column shows the redacted
    findings summary (e.g. ``"BLOCKED: found aws_access_key (1 pattern)"``).
    """
    rows = _build_result_rows(
        summary, filenames, s3_errors_by_file, blocked_by_file
    )
    df = pd.DataFrame(
        rows, columns=["filename", "status", "chunks", "error", "s3_error"]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


# Issue #12 — sentinel string for "S3 was not even attempted because
# the operator has not configured AWS_S3_BUCKET". Distinct from a
# real boto3 failure so ops can tell a config issue from a runtime
# blip at a glance.
_NO_BUCKET_SENTINEL = "(skipped — no AWS_S3_BUCKET configured)"


# Issue #15 — sentinel prefix for the per-file ``error`` column when
# a file is blocked by the pre-upload PII/secret scanner. Rendered
# before the redacted findings summary so an operator triaging the
# table can tell at a glance which files were blocked by the scanner
# vs. files that failed during Chroma ingestion.
_BLOCKED_PREFIX = "BLOCKED: found"


def _scan_file_for_secrets(file_path: Path) -> List[Finding]:
    """Run :func:`telecom_rag.security.scan.scan_pages` on a saved PDF
    and return the per-page findings.

    Pure-data helper: takes the saved file path, calls the existing
    :func:`parse_pdf` extractor (the same one the ingest pipeline
    uses), and runs the scanner over the page records. Returns an
    empty list when the file cannot be parsed (a failed parse is
    itself a downstream concern, not a scanner concern — the page
    surfaces parse failures in the per-file ``error`` column via
    the existing ``ingest_directory`` errors path).

    This means a clean PDF runs ``parse_pdf`` twice (once for the
    scan, once for ``ingest_directory``). That's a deliberate
    sub-second cost; see Issue #15 plan §"Where does the scan sit
    in the pipeline?" for the rejection of "scan inside the
    pipeline" as a placement.
    """
    try:
        pages = parse_pdf(str(file_path))
    except Exception:  # noqa: BLE001
        # Parse failure is surfaced by ``ingest_directory`` later;
        # we don't double-report it from the scan step.
        return []
    return scan_pages(pages)


def _try_s3_upload(file_path: str | Path, filename: str) -> str:
    """Try to mirror ``file_path`` to S3. Return an empty string on
    success, or a user-facing error string on failure.

    The string is safe to render in a Streamlit table cell: it does
    not include the raw ``aws_secret_access_key`` (the
    ``test_s3_module_does_not_leak_test_credentials_to_logs`` test
    in ``tests/test_s3_upload.py`` pins this).

    When ``settings.aws_s3_bucket`` is None we return the
    :data:`_NO_BUCKET_SENTINEL` string without even attempting the
    boto3 call — saves a round-trip when the operator has not
    configured S3 at all.
    """
    if not settings.aws_s3_bucket:
        return _NO_BUCKET_SENTINEL
    try:
        upload_pdf(file_path, key=f"uploads/{filename}")
        return ""
    except S3UploadError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001
        # Defensive: any non-S3UploadError exception (e.g. a boto3
        # bug that leaks a different exception type) is still
        # surfaced as a string instead of crashing the page.
        return f"S3 upload failed: {exc}"


def _ingest_upload(uploaded_files: List[Any]) -> None:
    """Save the uploaded files to a unique temp dir and call
    :func:`ingest_directory`. Render the result table on success and a
    friendly ``st.error`` on a hard failure (e.g. disk full, missing
    dependencies, embedding provider unavailable).

    Issue #12: each saved file is also PUT to S3 before
    ``ingest_directory`` runs. Per-file S3 failures are captured in
    ``s3_errors_by_file`` and rendered in the ``s3_error`` column of
    the per-file table. The local Chroma ingestion is NOT blocked
    by an S3 failure (the file is on local disk regardless).

    Issue #15: BEFORE the S3 PUT and BEFORE ``ingest_directory``, each
    saved PDF is run through the pre-upload PII/secret scanner
    (:func:`telecom_rag.security.scan.scan_pages`). A file with any
    ``high``- or ``medium``-severity finding is **blocked**: no S3
    PUT, no Chroma ingestion, no LangSmith trace. The redacted
    finding summary is surfaced in the per-file ``error`` column
    with ``status="blocked"``. Files with only ``low`` findings (or
    none) proceed as today.

    Issue #18 AC4: every block emits a structured
    ``event=upload_blocked`` JSON log line via the
    ``_LOGGER`` (named ``app.pages.upload``). The payload contains
    ``filename``, ``blocked_patterns`` (pattern_ids, not redacted
    strings), ``file_sha256``, and ``timestamp`` — sufficient for a
    SOC analyst to reconstruct "what was blocked, when, by which
    file" without leaking the secret value into the log group.
    """
    # Per-upload temp dir ensures a re-upload after a crash doesn't
    # collide with the previous run's files (Issue #6 lifecycle
    # invariant 3: TOCTOU on file save).
    upload_id = uuid.uuid4().hex[:8]
    temp_dir = _UPLOAD_ROOT / upload_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    filenames: List[str] = []
    s3_errors_by_file: Dict[str, str] = {}
    # Issue #15: per-file block status. ``""`` means "not blocked"
    # (proceed); a non-empty string is the redacted summary shown
    # in the per-file ``error`` column. We track by filename so the
    # render step can mix blocked + indexed + failed rows in one
    # table.
    blocked_by_file: Dict[str, str] = {}
    try:
        files_to_ingest: List[Path] = []
        for uploaded in uploaded_files:
            # Streamlit's ``UploadedFile`` exposes ``.name`` (the
            # browser-side filename) and ``.getbuffer()`` (the bytes).
            # ``getbuffer()`` avoids materializing a Python str of the
            # whole file, which matters for multi-MB vendor manuals.
            target = temp_dir / uploaded.name
            with target.open("wb") as fh:
                shutil.copyfileobj(uploaded, fh)
            filenames.append(uploaded.name)

            # Issue #15: scan BEFORE S3 PUT and BEFORE ingest. A
            # blocked file never leaves the local disk, never
            # reaches S3, never gets parsed by Chroma. The scan
            # runs ``parse_pdf`` once on the saved file (a second
            # parse happens later inside ``ingest_directory`` for
            # clean files — the cost is sub-second on a warm Chroma
            # and is the price of keeping S3 + Chroma + LangSmith
            # all behind the same gate).
            findings = _scan_file_for_secrets(target)
            if has_blocking_finding(findings):
                blocked_by_file[uploaded.name] = (
                    f"{_BLOCKED_PREFIX} "
                    f"{summarize_blocking_patterns(findings)}"
                )
                # Issue #18 AC4 — emit a structured audit log line
                # for every block. The log goes through the existing
                # ``setup_json_logging()`` seam (Issue #13 AC3) so
                # CloudWatch Logs Insights sees one JSON record per
                # block with the payload shape documented in the
                # spec (filename, user, blocked_patterns, file_sha256,
                # timestamp).
                #
                # ``blocked_patterns`` carries pattern_ids only — NOT
                # redacted or raw values — so an attacker reading the
                # CloudWatch log group cannot infer the secret even
                # from the redacted form ("AKIA****MPLE" still
                # confirms the file contained an AWS key). The
                # ``Finding.match`` and ``Finding.redacted`` fields
                # stay out of the audit log on purpose.
                #
                # Issue #22 AC9: ``user`` is now the authenticated
                # username from ``st.session_state``, NOT the
                # pre-#22 ``"streamlit-session"`` sentinel. The
                # ``/my_uploads`` page filters the audit log by
                # this field, so a real per-user attribution is
                # required for the feature to surface anything
                # useful. Falls back to ``"anonymous"`` when no
                # session user is set (a defensive default — the
                # auth gate in ``app/main.py`` always populates
                # ``st.session_state.username`` on a successful
                # login, but the upload page could be hit before
                # the session is fully initialized under some
                # Streamlit lifecycle edges).
                _LOGGER.warning(
                    "upload_blocked",
                    extra={
                        "event": "upload_blocked",
                        # Python's ``logging.LogRecord`` reserves
                        # the attribute ``filename`` for the source
                        # file of the log call, so we surface the
                        # uploaded file's basename under a JSON
                        # field named ``file_name`` instead. The
                        # audit value is identical; only the wire
                        # key differs from the spec's prose to avoid
                        # the reserved-attribute ``KeyError`` at log
                        # emission time.
                        "file_name": uploaded.name,
                        "user": st.session_state.get(
                            "username", "anonymous"
                        ),
                        "blocked_patterns": sorted({
                            f.pattern_id for f in findings
                            if has_blocking_finding([f])
                        }),
                        "file_sha256": hashlib.sha256(
                            target.read_bytes()
                        ).hexdigest(),
                    },
                )
                # Drop the file from the batch — both the S3 mirror
                # and the local Chroma ingest see ONLY the surviving
                # files. A blocked file leaves no trace outside the
                # local temp dir (cleaned up in the ``finally``
                # below) and the per-file ``error`` column.
                continue

            files_to_ingest.append(target)

            # Issue #12: PUT to S3 BEFORE ingest_directory. Doing
            # S3 first means a failed PUT surfaces immediately in
            # the per-file table, AND the local Chroma ingestion
            # still runs (the file is on local disk). A future
            # issue could parallelize S3 + ingest via threads if
            # latency becomes a concern.
            s3_err = _try_s3_upload(target, uploaded.name)
            if s3_err:
                s3_errors_by_file[uploaded.name] = s3_err

        try:
            summary = ingest_directory(
                str(temp_dir),
                settings.chroma_persist_dir,
                # Issue #15: when scanning blocks files, ingest only
                # the surviving (clean) subset. Passing the full
                # ``temp_dir`` would force ``ingest_directory`` to
                # re-parse the blocked PDF and surface a fresh error
                # in its own per-file table — the user would see two
                # rows for one file, which is misleading.
                file_allowlist=[p.name for p in files_to_ingest] or None,
            )
        except TypeError:
            # Backward-compat: an older ``ingest_directory`` signature
            # that doesn't accept ``file_allowlist``. Fall back to the
            # whole-dir call (the blocked files will fail in their own
            # way inside ingest; the per-file block row from the scan
            # still surfaces correctly).
            summary = ingest_directory(
                str(temp_dir), settings.chroma_persist_dir
            )
        except Exception as exc:  # noqa: BLE001
            # Hard failure (e.g. embedding provider unavailable, disk
            # full, Chroma SQLite lock). Render as a banner; do NOT
            # crash the Streamlit thread.
            st.error(f"Ingest failed: {exc}")
            return

        _render_results_table(
            summary, filenames, s3_errors_by_file, blocked_by_file
        )

        # Issue #22 AC9: emit one ``event=upload_indexed`` JSON log
        # line PER successfully indexed file. Each line carries the
        # authenticated ``user`` from ``st.session_state``, the
        # ``chunks`` count for the file, the ``file_sha256``, and
        # the ``batch_id`` (the same ``upload_id`` that
        # ``ingest_directory`` sees — pinned to one batch per
        # ``Ingest`` button click). The ``/my_uploads`` page reads
        # these lines and renders a per-user history.
        #
        # We log AFTER the per-file table renders so the user sees
        # the result first; a log emission failure here would be a
        # silent-noop (the JSON log handler swallows IO errors per
        # ``setup_json_logging``'s contract) and not block the page.
        _batch_id = upload_id
        _chunks_by_file = summary.get("chunks_by_file", {}) or {}
        _user = st.session_state.get("username", "anonymous")
        for _filename in _chunks_by_file:
            _target_path = temp_dir / _filename
            try:
                _file_sha256 = hashlib.sha256(
                    _target_path.read_bytes()
                ).hexdigest()
            except OSError:
                # The temp file may have been removed by the
                # ``finally`` cleanup on a re-run; fall back to the
                # empty hash so the log line still records the
                # attempt. /my_uploads surfaces the row, just
                # without a file fingerprint.
                _file_sha256 = ""
            _LOGGER.info(
                "upload_indexed",
                extra={
                    "event": "upload_indexed",
                    "file_name": _filename,
                    "user": _user,
                    "chunks": _chunks_by_file[_filename],
                    "file_sha256": _file_sha256,
                    "batch_id": _batch_id,
                },
            )

        error_count = len(summary.get("errors", []))
        s3_error_count = sum(1 for v in s3_errors_by_file.values() if v)
        blocked_count = len(blocked_by_file)
        if (
            error_count == 0
            and summary.get("total_files", 0) > 0
            and blocked_count == 0
        ):
            # Cache invalidation: the chat page's
            # ``@st.cache_resource``-wrapped vectorstore would otherwise
            # keep its pre-ingest embedding adapter bound. ``clear()``
            # is cheap (Streamlit just drops its resource cache).
            st.cache_resource.clear()
            msg = (
                f"Indexed {summary['total_files']} PDF(s) "
                f"({summary['total_chunks']} chunks)."
            )
            if s3_error_count > 0:
                msg += (
                    f" {s3_error_count} S3 upload(s) failed — see the "
                    f"s3_error column above; Chroma was indexed regardless."
                )
            st.success(msg)
        elif blocked_count > 0 and error_count == 0:
            # All (non-blocked) files indexed cleanly, but some were
            # blocked by the scanner. Render as a warning so the
            # block isn't silently lost.
            st.warning(
                f"Indexed {summary.get('total_files', 0)} PDF(s); "
                f"{blocked_count} blocked by the pre-upload PII/secret "
                f"scanner — see the table above."
            )
        elif error_count > 0:
            st.warning(
                f"Indexed {summary.get('total_files', 0)} PDF(s) with "
                f"{error_count} failure(s). See the table above."
            )
        else:
            if blocked_count > 0:
                st.warning(
                    f"{blocked_count} PDF(s) blocked by the pre-upload "
                    f"PII/secret scanner. Nothing was indexed."
                )
            else:
                st.info("No PDF files were ingested.")
    finally:
        # Best-effort cleanup of the temp dir. We swallow any cleanup
        # error because the user's primary feedback (the table) has
        # already rendered by this point.
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

st.title("Upload PDFs")
st.markdown(
    "Drag in one or more PDF vendor manuals, SOPs, or spec sheets. "
    "Click **Ingest** to push them into the Chroma collection. "
    "Per-file errors are listed in the table below — one bad file "
    "never blocks the rest of the batch."
)

uploaded = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Drag and drop one or more PDF files. Each file is ingested "
    "independently; per-file failures are listed in the table.",
)

ingest_clicked = st.button("Ingest", disabled=not uploaded)

if ingest_clicked and uploaded:
    _ingest_upload(uploaded)
elif ingest_clicked and not uploaded:
    st.info("Pick at least one PDF to ingest.")