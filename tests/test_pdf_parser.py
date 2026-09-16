"""
Tests for the PDF parser (ingestion/parsers/pdf_parser.py).

Test structure:
  - TestCleanText         → unit tests for the internal text cleaning function
  - TestInferSourceLabel  → unit tests for folder-name-to-source-label mapping
  - TestParsePdf          → integration tests against real single PDF files
  - TestParsePdfFolder    → integration tests against a real folder of PDFs

Unit tests (TestCleanText, TestInferSourceLabel) use no files — they test
pure logic with controlled inputs. They are fast and always run.

Integration tests (TestParsePdf, TestParsePdfFolder) run against PDFs that
must be present in ingestion/sources/pdfs/. If a PDF is missing, the test
is skipped rather than failed, so the suite still works on any machine.
"""

import pytest
from pathlib import Path
from ingestion.parsers.pdf_parser import parse_pdf, parse_pdf_folder, infer_source_label, _clean_text
from ingestion.document import Document

# ── File paths to real PDFs on disk ───────────────────────────────────────────
SOURCES_DIR = Path(__file__).parent.parent / "ingestion" / "sources" / "pdfs"

SEC_PDF    = SOURCES_DIR / "sec"  / "build-wealth-over-time-through-saving-and-investing.pdf"
CFPB_PDF   = SOURCES_DIR / "cfpb" / "cfpb_your-money-your-goals_financial-empowerment_toolkit.pdf"
FED_FOLDER = SOURCES_DIR / "fed"


# ══════════════════════════════════════════════════════════════════════════════
# TestCleanText
# Tests the private _clean_text() helper in isolation.
# These are pure unit tests — no files, no I/O, just string in → string out.
# If any of these fail, the parser is silently storing dirty text in the DB.
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanText:

    def test_rejoins_hyphenated_line_breaks(self):
        """
        PDFs frequently split words across lines with a hyphen (e.g. "invest-\nment").
        _clean_text should rejoin these into a single word ("investment").
        Without this, the chunker and embedder see broken vocabulary.
        """
        result = _clean_text("invest-\nment returns")
        assert "investment returns" in result

    def test_collapses_excessive_newlines(self):
        """
        Raw PDF text often has 4-6 consecutive newlines between sections.
        _clean_text should collapse anything 3+ newlines down to 2.
        This keeps paragraph structure intact without creating huge empty gaps
        that would waste token space in chunks.
        """
        result = _clean_text("paragraph one\n\n\n\n\nparagraph two")
        assert "\n\n\n" not in result

    def test_strips_surrounding_whitespace(self):
        """
        Pages often have leading/trailing spaces or newlines from PDF margins.
        _clean_text should strip these so Documents don't start/end with whitespace.
        """
        result = _clean_text("   some text   ")
        assert result == "some text"

    def test_empty_string_stays_empty(self):
        """
        Some PDF pages are genuinely blank (e.g. divider pages, back covers).
        _clean_text should return an empty string, which the parser uses to
        skip the page entirely rather than storing an empty Document.
        """
        assert _clean_text("") == ""


# ══════════════════════════════════════════════════════════════════════════════
# TestInferSourceLabel
# Tests the folder-name → human-readable source label mapping.
# This label is stored as metadata on every Document and Chunk, so it must
# be correct — it's what the eval harness uses to filter by source.
# ══════════════════════════════════════════════════════════════════════════════

class TestInferSourceLabel:

    def test_sec_folder_maps_correctly(self):
        """
        PDFs stored in .../pdfs/sec/ should be labelled "SEC/Investor.gov".
        This is the label that will appear in retrieved chunk metadata.
        """
        label = infer_source_label(str(SOURCES_DIR / "sec" / "any.pdf"))
        assert label == "SEC/Investor.gov"

    def test_cfpb_folder_maps_correctly(self):
        """
        PDFs stored in .../pdfs/cfpb/ should be labelled "CFPB".
        """
        label = infer_source_label(str(SOURCES_DIR / "cfpb" / "any.pdf"))
        assert label == "CFPB"

    def test_fed_folder_maps_correctly(self):
        """
        PDFs stored in .../pdfs/fed/ should be labelled "Federal Reserve".
        """
        label = infer_source_label(str(SOURCES_DIR / "fed" / "any.pdf"))
        assert label == "Federal Reserve"

    def test_finra_folder_maps_correctly(self):
        """
        PDFs stored in .../pdfs/finra/ should be labelled "FINRA".
        No PDFs are downloaded yet, but the label mapping exists in SOURCE_LABELS
        so this unit test runs without any files on disk.
        """
        label = infer_source_label(str(SOURCES_DIR / "finra" / "any.pdf"))
        assert label == "FINRA"

    def test_unknown_folder_falls_back_to_uppercase(self):
        """
        If the folder name isn't in SOURCE_LABELS, infer_source_label should
        fall back to the folder name in uppercase rather than crashing.
        This allows new sources to be added without changing the parser code.
        """
        label = infer_source_label("/some/unknown/folder/file.pdf")
        assert label == "FOLDER"


# ══════════════════════════════════════════════════════════════════════════════
# TestParsePdf
# Integration tests against real single PDF files.
# These verify that parse_pdf() works end-to-end on your actual source files.
# Skipped automatically if the PDF file is not present on this machine.
# ══════════════════════════════════════════════════════════════════════════════

class TestParsePdf:

    def test_raises_if_file_not_found(self):
        """
        If a caller passes a path that doesn't exist, parse_pdf should raise
        FileNotFoundError immediately rather than crashing inside PyMuPDF
        with a cryptic error. Ensures clear failure messages during ingestion.
        """
        with pytest.raises(FileNotFoundError):
            parse_pdf("/nonexistent/path/file.pdf", "Test")

    @pytest.mark.skipif(not SEC_PDF.exists(), reason="SEC PDF not present")
    def test_sec_pdf_returns_documents(self):
        """
        Parsing the SEC wealth-building guide should produce at least one Document.
        If zero Documents are returned, either the PDF is unreadable or every
        page was below the 50-character threshold and got filtered out.
        """
        docs = parse_pdf(str(SEC_PDF), "SEC/Investor.gov")
        assert len(docs) > 0, "Expected at least one page from the SEC PDF"

    @pytest.mark.skipif(not SEC_PDF.exists(), reason="SEC PDF not present")
    def test_sec_pdf_documents_have_required_fields(self):
        """
        Every Document produced by the parser must have all required metadata fields
        populated and non-empty. These fields are used downstream by the chunker,
        the vector store, and the eval harness. Missing metadata means broken retrieval.

        Checks: text is non-empty, source matches what we passed in,
        section follows the page_N format, url and doc_id are set.
        """
        docs = parse_pdf(str(SEC_PDF), "SEC/Investor.gov")
        for doc in docs:
            assert isinstance(doc, Document)
            assert doc.text.strip(),                    "text should not be empty"
            assert doc.source == "SEC/Investor.gov",    "source label should match input"
            assert doc.section.startswith("page_"),     "section should follow page_N format"
            assert doc.url != "",                       "url should be set"
            assert doc.doc_id != "",                    "doc_id should be set"

    @pytest.mark.skipif(not SEC_PDF.exists(), reason="SEC PDF not present")
    def test_sec_pdf_doc_ids_are_unique_per_page(self):
        """
        Each page must produce a unique doc_id. If two pages share a doc_id,
        the vector store treats them as the same document — one silently overwrites
        the other. This was a real bug caught during development: all pages hashed
        to the same ID because the url was identical across pages. Fixed by
        appending #page_N to the url before hashing.
        """
        docs = parse_pdf(str(SEC_PDF), "SEC/Investor.gov")
        ids = {d.doc_id for d in docs}   # set removes duplicates
        assert len(ids) == len(docs), (
            f"Expected {len(docs)} unique doc_ids but got {len(ids)}. "
            "Duplicate IDs will cause silent data loss in the vector store."
        )

    @pytest.mark.skipif(not SEC_PDF.exists(), reason="SEC PDF not present")
    def test_sec_pdf_no_hyphenated_line_breaks_in_output(self):
        """
        Confirms that _clean_text was actually applied to all pages.
        If any page still contains '-\\n', the cleaning step was skipped,
        which means the embedder will see broken words and produce worse vectors.
        """
        docs = parse_pdf(str(SEC_PDF), "SEC/Investor.gov")
        for doc in docs:
            assert "-\n" not in doc.text, (
                f"Page {doc.section} still contains hyphenated line breaks"
            )

    @pytest.mark.skipif(not CFPB_PDF.exists(), reason="CFPB PDF not present")
    def test_cfpb_pdf_returns_documents(self):
        """
        The CFPB toolkit is a large multi-page PDF (~100+ pages).
        This confirms the parser can handle a large document without crashing.
        """
        docs = parse_pdf(str(CFPB_PDF), "CFPB")
        assert len(docs) > 0

    @pytest.mark.skipif(not CFPB_PDF.exists(), reason="CFPB PDF not present")
    def test_cfpb_pdf_text_is_meaningful_length(self):
        """
        Checks that extracted text is substantive, not just noise.
        An average page length under 100 characters suggests the parser is
        failing to extract real content — e.g. the PDF is scanned image-based
        rather than text-based, which requires OCR (not currently supported).
        """
        docs = parse_pdf(str(CFPB_PDF), "CFPB")
        avg_len = sum(len(d.text) for d in docs) / len(docs)
        assert avg_len > 100, (
            f"Average page text is only {avg_len:.0f} chars — "
            "PDF may be image-based and require OCR."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TestParsePdfFolder
# Integration tests against the full Federal Reserve folder (6 PDFs).
# Verifies that parse_pdf_folder() correctly aggregates across multiple files.
# ══════════════════════════════════════════════════════════════════════════════

class TestParsePdfFolder:

    def test_raises_if_folder_not_found(self):
        """
        parse_pdf_folder should raise FileNotFoundError immediately if the
        folder path doesn't exist, rather than silently returning an empty list.
        An empty result would be indistinguishable from a folder with no PDFs.
        """
        with pytest.raises(FileNotFoundError):
            parse_pdf_folder("/nonexistent/folder", "Test")

    @pytest.mark.skipif(not FED_FOLDER.exists(), reason="Fed folder not present")
    def test_fed_folder_returns_documents(self):
        """
        The fed/ folder contains 6 PDFs. parse_pdf_folder should process all
        of them and return a combined list of Documents with length > 0.
        """
        docs = parse_pdf_folder(str(FED_FOLDER), "Federal Reserve")
        assert len(docs) > 0

    @pytest.mark.skipif(not FED_FOLDER.exists(), reason="Fed folder not present")
    def test_fed_folder_all_documents_have_correct_source(self):
        """
        Every Document produced from the fed/ folder should carry the source
        label we passed in ("Federal Reserve"), regardless of which PDF it
        came from. This ensures metadata is consistent across all files in a folder.
        """
        docs = parse_pdf_folder(str(FED_FOLDER), "Federal Reserve")
        for doc in docs:
            assert doc.source == "Federal Reserve", (
                f"Document from {doc.url} has incorrect source: '{doc.source}'"
            )
