"""Security scanning package.

Home to the pre-upload PII/secret scanner used by the upload page
(``app/pages/upload.py``). The scanner extracts text from uploaded
PDFs and runs a regex catalog of well-known secret and PII patterns
against it. Any ``high``-severity (secret formats) or
``medium``-severity (strong PII) finding blocks the entire upload
pipeline: no S3 PUT, no Chroma ingestion, no LangSmith trace. The
``low``-severity tier (heuristics like email, phone, IP) is logged
but does NOT block — vendor manuals legitimately contain support
contact info.

This module is intentionally pure-data: no I/O, no S3, no Chroma, no
Streamlit. Every function takes text or page records and returns
findings. The integration with ``app/pages/upload.py`` lives in the
upload page, not here.
"""