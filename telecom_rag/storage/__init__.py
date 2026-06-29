"""Storage package.

Contains S3 / blob-storage helpers. Currently home to the
Issue #12 ``upload_pdf`` helper. Future surface (Issue: S3
listing in admin, S3 lifecycle automation) lands here too.
"""

from telecom_rag.storage.s3 import S3UploadError, upload_pdf

__all__ = ["S3UploadError", "upload_pdf"]