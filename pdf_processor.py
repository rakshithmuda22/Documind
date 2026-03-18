"""
PDF processing and smart chunking for DocuMind.

Extracts text from PDFs page-by-page using PyMuPDF, then applies
paragraph-aware chunking with configurable size and overlap.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import List, Tuple

import fitz  # PyMuPDF

from models import DocumentChunk

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

MAX_CHUNK_TOKENS: int = 500
OVERLAP_TOKENS: int = 50
MIN_TEXT_LENGTH: int = 50  # Minimum total text to consider a PDF valid


# ──────────────────────────────────────────────────────────────────────
# Public functions
# ──────────────────────────────────────────────────────────────────────


def process_pdf(
    file_bytes: bytes,
    filename: str,
) -> List[DocumentChunk]:
    """Extract text from a PDF and split it into chunks.

    Args:
        file_bytes: Raw PDF file content.
        filename: Original filename for source tracking.

    Returns:
        List of DocumentChunk objects ready for embedding.

    Raises:
        ValueError: If the PDF is empty, encrypted, or has no extractable text.
    """
    validate_pdf(file_bytes)
    pages = extract_pages(file_bytes)
    doc_hash = compute_doc_hash(file_bytes)

    total_text = "".join(text for text, _ in pages)
    if len(total_text.strip()) < MIN_TEXT_LENGTH:
        raise ValueError(
            "This PDF appears to be scanned or image-based. "
            "DocuMind requires PDFs with selectable text. "
            "Try running OCR on the document first."
        )

    chunks = chunk_pages(pages, filename, doc_hash)

    logger.info(
        "Processed '%s': %d pages, %d chunks, hash=%s",
        filename,
        len(pages),
        len(chunks),
        doc_hash[:8],
    )
    return chunks


def validate_pdf(file_bytes: bytes) -> None:
    """Validate that file_bytes is a readable PDF.

    Args:
        file_bytes: Raw file content.

    Raises:
        ValueError: If the file is not a valid, readable PDF.
    """
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:
        raise ValueError(
            "Unable to read this file as a PDF. "
            "Please ensure it is a valid PDF document."
        )

    if doc.is_encrypted:
        doc.close()
        raise ValueError(
            "This PDF is encrypted/password-protected. "
            "Please remove the password and try again."
        )

    if doc.page_count == 0:
        doc.close()
        raise ValueError("This PDF has no pages.")

    doc.close()


def compute_doc_hash(file_bytes: bytes) -> str:
    """Compute a short SHA-256 hash of the file content.

    Args:
        file_bytes: Raw file content.

    Returns:
        First 16 characters of the hex digest.
    """
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def estimate_tokens(text: str) -> int:
    """Estimate the token count of a text string.

    Uses the rough heuristic of ~4 characters per token.

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    return len(text) // 4


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────


def extract_pages(file_bytes: bytes) -> List[Tuple[str, int]]:
    """Extract text from each page of a PDF.

    Args:
        file_bytes: Raw PDF content.

    Returns:
        List of (page_text, page_number) tuples. Page numbers are 1-based.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        text = _clean_text(text)
        if text.strip():
            pages.append((text, page_num + 1))  # 1-based page numbers
    doc.close()
    return pages


def chunk_pages(
    pages: List[Tuple[str, int]],
    filename: str,
    doc_hash: str,
) -> List[DocumentChunk]:
    """Split page texts into overlapping chunks.

    Strategy:
    1. Split each page into paragraphs (double newlines).
    2. Merge small paragraphs until reaching max chunk size.
    3. Split paragraphs that exceed max chunk size with overlap.

    Args:
        pages: List of (text, page_number) from extract_pages.
        filename: Source filename.
        doc_hash: Document hash for chunk ID generation.

    Returns:
        Ordered list of DocumentChunk objects.
    """
    chunks: List[DocumentChunk] = []
    chunk_index = 0

    for page_text, page_number in pages:
        paragraphs = _split_into_paragraphs(page_text)
        buffer = ""

        for para in paragraphs:
            para_tokens = estimate_tokens(para)

            # If a single paragraph exceeds max, split it
            if para_tokens > MAX_CHUNK_TOKENS:
                # Flush the buffer first
                if buffer.strip():
                    chunks.append(_make_chunk(
                        buffer.strip(), page_number, filename,
                        doc_hash, chunk_index,
                    ))
                    chunk_index += 1
                    buffer = ""

                # Split the large paragraph
                sub_chunks = _split_large_text(para, MAX_CHUNK_TOKENS, OVERLAP_TOKENS)
                for sub in sub_chunks:
                    chunks.append(_make_chunk(
                        sub.strip(), page_number, filename,
                        doc_hash, chunk_index,
                    ))
                    chunk_index += 1
                continue

            # Check if adding this paragraph exceeds the limit
            combined = buffer + "\n\n" + para if buffer else para
            if estimate_tokens(combined) > MAX_CHUNK_TOKENS:
                # Flush the buffer
                if buffer.strip():
                    chunks.append(_make_chunk(
                        buffer.strip(), page_number, filename,
                        doc_hash, chunk_index,
                    ))
                    chunk_index += 1
                buffer = para
            else:
                buffer = combined

        # Flush remaining buffer for this page
        if buffer.strip():
            chunks.append(_make_chunk(
                buffer.strip(), page_number, filename,
                doc_hash, chunk_index,
            ))
            chunk_index += 1

    return chunks


def _make_chunk(
    text: str,
    page_number: int,
    filename: str,
    doc_hash: str,
    chunk_index: int,
) -> DocumentChunk:
    """Create a DocumentChunk instance."""
    return DocumentChunk(
        id=f"{doc_hash}_{chunk_index}",
        text=text,
        page_number=page_number,
        source_filename=filename,
        chunk_index=chunk_index,
    )


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs by double newlines.

    Args:
        text: Raw page text.

    Returns:
        List of non-empty paragraph strings.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_large_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    """Split a large text block into overlapping chunks by character count.

    Args:
        text: Text to split.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Number of overlap tokens between chunks.

    Returns:
        List of text chunks.
    """
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4
    step = max_chars - overlap_chars

    if step <= 0:
        step = max_chars // 2

    parts = []
    start = 0
    while start < len(text):
        end = start + max_chars
        parts.append(text[start:end])
        start += step

    return parts


def _clean_text(text: str) -> str:
    """Clean extracted PDF text.

    Fixes common PDF extraction artifacts:
    - Removes excessive whitespace
    - Fixes hyphenation across line breaks
    - Normalises line endings

    Args:
        text: Raw text from PyMuPDF.

    Returns:
        Cleaned text string.
    """
    # Fix hyphenation across line breaks (e.g., "docu-\nment" → "document")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Replace single newlines within a paragraph with spaces
    # (keep double newlines as paragraph separators)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
