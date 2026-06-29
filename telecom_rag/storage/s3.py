"""S3 upload helper (Issue #12).

Wires the README §12.3 contract that every PDF uploaded through the
Streamlit upload page is also PUT to S3 at
``uploads/{filename}``. The S3 object is a **side-effect** of the
local Chroma ingestion — NOT a replacement for it. A failed PUT
surfaces as :class:`S3UploadError` so the upload page's per-file
table can render the failure without crashing the Streamlit thread.

Auth resolution order (per the Issue #12 decision captured
2026-06-26):

1. Explicit env-var credentials (``AWS_ACCESS_KEY_ID`` +
   ``AWS_SECRET_ACCESS_KEY``) — built via
   ``boto3.Session(aws_access_key_id=..., aws_secret_access_key=...)``.
   This is the README §13 expectation for non-EC2 deployments.
2. boto3 default credential chain — ``boto3.Session()`` with no args.
   This covers the EC2 instance-profile story (IAM role attached to
   the EC2 instance) and any developer who has ``~/.aws/credentials``.
3. No credentials at all — :class:`botocore.exceptions.NoCredentialsError`
   propagates from ``boto3.Session(...).client(...)``; we translate
   that into :class:`S3UploadError` so callers can ``except`` it
   without coupling to botocore.

Failure semantics: ANY boto3 failure during ``upload_file`` (network
blip, AccessDenied, bucket not found, etc.) is wrapped in
:class:`S3UploadError` with the underlying botocore message. The
underlying exception's ``str()`` is included verbatim so the
upload page's per-file table can render it. We do NOT include the
raw ``aws_secret_access_key`` in the wrapped message — a
regression-guard test (``test_s3_upload.py``) pins this.

Usage::

    from telecom_rag.storage.s3 import upload_pdf, S3UploadError
    try:
        uri = upload_pdf("/tmp/manual.pdf")
    except S3UploadError as exc:
        st.error(f"S3 upload failed: {exc}")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError


class S3UploadError(RuntimeError):
    """Raised when an S3 PUT fails for any reason.

    The ``args[0]`` is the user-facing message — safe to render in
    a Streamlit cell or log line. It deliberately does NOT include
    raw credentials (the test suite pins this contract).
    """


def _resolve_credentials() -> dict:
    """Build the kwargs to pass to ``boto3.Session(...)``.

    Returns a dict whose keys map to the ``boto3.Session`` parameters.
    An empty dict means "use the default credential chain" — i.e.
    ``boto3.Session()`` with no kwargs, which is what boto3 expects
    when no explicit creds are present.

    The explicit env-var path matches the README §13 expectation:
    on a non-EC2 host, the developer sets
    ``AWS_ACCESS_KEY_ID`` + ``AWS_SECRET_ACCESS_KEY`` in ``.env``
    and boto3 picks them up directly. On EC2, the IAM role attached
    to the instance provides creds via the metadata service and
    boto3 picks them up via the default chain — the env-var path
    is NOT required.
    """
    access_key = os.environ.get("AWS_ACCESS_KEY_ID") or None
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or None
    region = os.environ.get("AWS_DEFAULT_REGION") or None
    if access_key and secret_key:
        return {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "region_name": region,
        }
    return {}


def _resolve_bucket(bucket: Optional[str]) -> str:
    """Resolve the bucket name to use.

    If the caller passes a non-empty ``bucket`` we use that
    (caller-supplied wins). Otherwise we fall back to
    ``settings.aws_s3_bucket`` — read lazily so tests can
    monkeypatch ``telecom_rag.config.settings`` without circular
    import at module-load time.
    """
    if bucket:
        return bucket
    # Lazy import: ``telecom_rag.config`` triggers pydantic-settings
    # which reads .env at module-load. Tests that monkeypatch the
    # settings dict should patch AFTER import; this lazy read keeps
    # the surface clean.
    from telecom_rag.config import settings  # noqa: PLC0415

    configured = getattr(settings, "aws_s3_bucket", None) or ""
    if not configured:
        raise S3UploadError(
            "no AWS_S3_BUCKET configured: set AWS_S3_BUCKET in .env "
            "or pass bucket=... explicitly to upload_pdf"
        )
    return configured


def upload_pdf(
    file_path: str | Path,
    *,
    bucket: Optional[str] = None,
    key: Optional[str] = None,
    region: Optional[str] = None,
) -> str:
    """Upload a single PDF to S3 and return the object URI.

    Parameters
    ----------
    file_path:
        Local path to the PDF file. The file must exist and be readable.
    bucket:
        S3 bucket name. Defaults to ``settings.aws_s3_bucket``.
    key:
        S3 object key. Defaults to ``uploads/{Path(file_path).name}``
        per README §12.3.
    region:
        AWS region for the client. Defaults to ``settings.aws_default_region``.

    Returns
    -------
    str
        The S3 URI of the uploaded object: ``s3://{bucket}/{key}``.

    Raises
    ------
    S3UploadError
        If the bucket is unset, credentials cannot be resolved, or
        the underlying ``boto3.client.upload_file`` call fails for
        any reason (network blip, AccessDenied, etc.).
    """
    resolved_bucket = _resolve_bucket(bucket)
    resolved_key = key or f"uploads/{Path(file_path).name}"

    creds = _resolve_credentials()
    if region and "region_name" not in creds:
        creds["region_name"] = region

    try:
        session = boto3.Session(**creds) if creds else boto3.Session()
        client = session.client("s3")
    except NoCredentialsError as exc:
        raise S3UploadError("no AWS credentials found") from exc
    except BotoCoreError as exc:
        # BotoCoreError is the parent of NoCredentialsError; this
        # branch catches sibling exceptions like
        # ``PartialCredentialsError`` so we surface a friendly
        # message instead of leaking botocore internals.
        raise S3UploadError(f"AWS client init failed: {exc}") from exc

    try:
        client.upload_file(str(file_path), resolved_bucket, resolved_key)
    except ClientError as exc:
        # ClientError carries the AWS response (Error.Code +
        # Error.Message). ``str(exc)`` includes the message verbatim
        # without the access key.
        raise S3UploadError(str(exc)) from exc
    except BotoCoreError as exc:
        raise S3UploadError(f"S3 upload failed: {exc}") from exc

    return f"s3://{resolved_bucket}/{resolved_key}"