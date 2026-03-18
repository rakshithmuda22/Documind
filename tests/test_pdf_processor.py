"""Tests for pdf_processor module."""

from __future__ import annotations

import pytest
import fitz  # PyMuPDF

from pdf_processor import (
    _clean_text,
    _split_into_paragraphs,
    _split_large_text,
    chunk_pages,
    compute_doc_hash,
    estimate_tokens,
    extract_pages,
    process_pdf,
    validate_pdf,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers — synthetic PDF generation
# ──────────────────────────────────────────────────────────────────────


def make_pdf(pages_text: list[str]) -> bytes:
    """Create a minimal in-memory PDF with the given page texts.

    Args:
        pages_text: List of strings, one per page.

    Returns:
        PDF file content as bytes.
    """
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_empty_pdf() -> bytes:
    """Create a PDF with one blank page (no text)."""
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ──────────────────────────────────────────────────────────────────────
# Tests — validation
# ──────────────────────────────────────────────────────────────────────


class TestValidatePdf:
    """Tests for the validate_pdf function."""

    def test_valid_pdf_passes(self):
        pdf = make_pdf(["Hello world"])
        validate_pdf(pdf)  # Should not raise

    def test_empty_bytes_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_pdf(b"")

    def test_invalid_bytes_raises(self):
        with pytest.raises(ValueError, match="valid PDF"):
            validate_pdf(b"not a pdf at all")

    def test_zero_pages_raises(self):
        # A valid PDF header but manipulated to have no pages is hard
        # to create, so we test with garbage that fitz rejects.
        with pytest.raises(ValueError):
            validate_pdf(b"%PDF-1.0 garbage")


# ──────────────────────────────────────────────────────────────────────
# Tests — text extraction
# ──────────────────────────────────────────────────────────────────────


class TestExtractPages:
    """Tests for the extract_pages function."""

    def test_single_page(self):
        pdf = make_pdf(["Hello from page one"])
        pages = extract_pages(pdf)
        assert len(pages) == 1
        assert "Hello from page one" in pages[0][0]
        assert pages[0][1] == 1  # 1-based page number

    def test_multi_page(self):
        pdf = make_pdf(["Page 1 text", "Page 2 text", "Page 3 text"])
        pages = extract_pages(pdf)
        assert len(pages) == 3
        assert pages[0][1] == 1
        assert pages[1][1] == 2
        assert pages[2][1] == 3

    def test_skips_empty_pages(self):
        """Pages with no extractable text are skipped."""
        doc = fitz.open()
        doc.new_page()  # Empty page
        page2 = doc.new_page()
        page2.insert_text((72, 72), "Content here")
        pdf_bytes = doc.tobytes()
        doc.close()

        pages = extract_pages(pdf_bytes)
        assert len(pages) == 1
        assert pages[0][1] == 2  # Page 2 (1-based)


# ──────────────────────────────────────────────────────────────────────
# Tests — process_pdf (integration)
# ──────────────────────────────────────────────────────────────────────


class TestProcessPdf:
    """Integration tests for the process_pdf function."""

    def test_basic_processing(self):
        # Text must exceed MIN_TEXT_LENGTH (50 chars) to be considered valid
        long_text = "This is a test document with enough text content. " * 5
        pdf = make_pdf([long_text])
        chunks = process_pdf(pdf, "test.pdf")
        assert len(chunks) >= 1
        assert chunks[0].source_filename == "test.pdf"
        assert chunks[0].page_number == 1

    def test_chunk_ids_are_unique(self):
        pdf = make_pdf(["A" * 3000, "B" * 3000])  # Two pages with lots of text
        chunks = process_pdf(pdf, "test.pdf")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_scanned_pdf_raises(self):
        """A PDF with no extractable text should raise ValueError."""
        pdf = make_empty_pdf()
        with pytest.raises(ValueError, match="scanned|image"):
            process_pdf(pdf, "scanned.pdf")

    def test_page_numbers_preserved(self):
        # Each page needs enough text to exceed MIN_TEXT_LENGTH
        long_text = "Page content with enough text to be considered valid. " * 3
        pdf = make_pdf([long_text, long_text])
        chunks = process_pdf(pdf, "multi.pdf")
        page_nums = {c.page_number for c in chunks}
        assert 1 in page_nums
        assert 2 in page_nums


# ──────────────────────────────────────────────────────────────────────
# Tests — chunking
# ──────────────────────────────────────────────────────────────────────


class TestChunking:
    """Tests for the chunking logic."""

    def test_short_text_single_chunk(self):
        pages = [("Short paragraph.", 1)]
        chunks = chunk_pages(pages, "test.pdf", "abc123")
        assert len(chunks) == 1
        assert chunks[0].text == "Short paragraph."

    def test_long_text_multiple_chunks(self):
        # Create text longer than 500 tokens (~2000 chars)
        long_text = "Word " * 600  # ~3000 chars = ~750 tokens
        pages = [(long_text, 1)]
        chunks = chunk_pages(pages, "test.pdf", "abc123")
        assert len(chunks) > 1

    def test_chunk_metadata(self):
        pages = [("Some text content.", 3)]
        chunks = chunk_pages(pages, "report.pdf", "hash123")
        assert chunks[0].source_filename == "report.pdf"
        assert chunks[0].page_number == 3
        assert chunks[0].id.startswith("hash123_")


# ──────────────────────────────────────────────────────────────────────
# Tests — utilities
# ──────────────────────────────────────────────────────────────────────


class TestUtilities:
    """Tests for utility functions."""

    def test_estimate_tokens(self):
        assert estimate_tokens("hello world!") == 3  # 12 chars // 4

    def test_compute_doc_hash_deterministic(self):
        data = b"test content"
        h1 = compute_doc_hash(data)
        h2 = compute_doc_hash(data)
        assert h1 == h2
        assert len(h1) == 16

    def test_compute_doc_hash_varies(self):
        h1 = compute_doc_hash(b"content A")
        h2 = compute_doc_hash(b"content B")
        assert h1 != h2

    def test_clean_text_fixes_hyphenation(self):
        result = _clean_text("docu-\nment")
        assert "document" in result

    def test_split_into_paragraphs(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        paras = _split_into_paragraphs(text)
        assert len(paras) == 3

    def test_split_large_text(self):
        text = "x" * 4000  # ~1000 tokens
        parts = _split_large_text(text, max_tokens=500, overlap_tokens=50)
        assert len(parts) > 1
        # Check overlap exists
        assert parts[0][-200:] == parts[1][:200]
