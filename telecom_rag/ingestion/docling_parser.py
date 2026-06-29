"""PDF parser.

Extracts page-level text from a PDF using ``pypdf``.  Pages with fewer
than 50 characters of extracted text are skipped (they are typically
covers, blank pages, or pure-image pages that would yield no useful
chunks downstream).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from pypdf import PdfReader

# Skip pages with fewer than this many extractable characters.
MIN_PAGE_CHARS = 50

# Collapse runs of whitespace (including newlines) into a single space.
_WHITESPACE_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    """Strip excess whitespace from extracted page text."""
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Parse a PDF and return one record per non-empty page.

    Each record::

        {
            "page_number": int,    # 1-indexed
            "text":        str,    # whitespace-collapsed page text
            "source_file": str,    # basename of the PDF
            "num_pages":   int,    # total pages in the PDF
        }

    Pages with fewer than :data:`MIN_PAGE_CHARS` characters are skipped.
    """
    path = Path(file_path)
    reader = PdfReader(str(path))
    num_pages = len(reader.pages)
    source_file = path.name

    out: List[Dict[str, Any]] = []
    for idx, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception:
            # Single-page failure should not abort the whole file.
            raw = ""
        text = _clean(raw)
        if len(text) < MIN_PAGE_CHARS:
            continue
        out.append(
            {
                "page_number": idx + 1,
                "text": text,
                "source_file": source_file,
                "num_pages": num_pages,
            }
        )
    return out
