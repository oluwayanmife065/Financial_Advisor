"""
Tests for the ingestion pipeline (ingestion/pipeline.py).

Two types:
  - Unit tests (TestPipelineUnit): mock all sub-components to test pipeline
    logic in isolation — no real PDFs, no real model, no real LanceDB.

  - Integration test (TestPipelineIntegration): runs the full pipeline
    against your real PDFs in ingestion/sources/pdfs/. This is the test
    that proves the entire system works end-to-end.

The integration test is slow (~30s) because it loads the embedding model
and processes real documents. The unit tests are fast (<1s).
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ingestion.document import Document, Chunk
from ingestion.pipeline import run


# Path to the real PDF sources for integration testing
REAL_SOURCES = Path(__file__).parent.parent / "ingestion" / "sources" / "pdfs"


# ══════════════════════════════════════════════════════════════════════════════
# TestPipelineUnit
# Tests pipeline logic using mocks — no files, no model, no DB.
# Each test isolates one behaviour of run().
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineUnit:

    def _make_doc(self, text="Some financial content."):
        """Helper — creates a minimal Document."""
        return Document(
            text=text, source="Test", section="page_1",
            url="file:///test/doc.pdf#page_1"
        )

    def _make_chunk(self, doc, index=0):
        """Helper — creates a Chunk from a Document with a dummy embedding."""
        c = Chunk(
            text=doc.text, source=doc.source, section=doc.section,
            url=doc.url, doc_id=doc.doc_id, chunk_index=index
        )
        c.embedding = [0.0] * 384
        return c

    @patch("ingestion.pipeline.write")
    @patch("ingestion.pipeline.embed")
    @patch("ingestion.pipeline.chunk_documents")
    @patch("ingestion.pipeline.parse_pdf_folder")
    @patch("ingestion.pipeline.count")
    def test_pipeline_returns_summary_dict(
        self, mock_count, mock_parse, mock_chunk, mock_embed, mock_write, tmp_path
    ):
        """
        run() must always return a dict with all five summary keys.
        The pipeline's return value is used by callers to verify the run
        and display progress. Missing keys would cause AttributeErrors downstream.
        """
        doc = self._make_doc()
        chunk = self._make_chunk(doc)

        mock_parse.return_value = [doc]
        mock_chunk.return_value = [chunk]
        mock_embed.side_effect = lambda chunks: chunks   # return same list
        mock_write.return_value = 1
        mock_count.return_value = 1

        # Create a fake sec/ folder with one PDF so the pipeline doesn't skip it
        sec_dir = tmp_path / "sec"
        sec_dir.mkdir()
        (sec_dir / "fake.pdf").touch()

        result = run(sources_root=tmp_path)

        required_keys = {
            "sources_processed", "documents_parsed",
            "chunks_created", "chunks_written", "elapsed_seconds"
        }
        assert required_keys == set(result.keys()), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    @patch("ingestion.pipeline.write")
    @patch("ingestion.pipeline.embed")
    @patch("ingestion.pipeline.chunk_documents")
    @patch("ingestion.pipeline.parse_pdf_folder")
    @patch("ingestion.pipeline.count")
    def test_pipeline_calls_each_stage_in_order(
        self, mock_count, mock_parse, mock_chunk, mock_embed, mock_write, tmp_path
    ):
        """
        The pipeline must call parse → chunk → embed → write in order.
        If order is wrong (e.g. embed before chunk), chunks won't have text yet.
        Uses call_order tracking to verify sequence.
        """
        call_order = []
        doc = self._make_doc()
        chunk = self._make_chunk(doc)

        mock_parse.side_effect = lambda *a, **kw: (call_order.append("parse"), [doc])[1]
        mock_chunk.side_effect = lambda *a, **kw: (call_order.append("chunk"), [chunk])[1]
        mock_embed.side_effect = lambda chunks: (call_order.append("embed"), chunks)[1]
        mock_write.side_effect = lambda chunks: (call_order.append("write"), 1)[1]
        mock_count.return_value = 1

        sec_dir = tmp_path / "sec"
        sec_dir.mkdir()
        (sec_dir / "fake.pdf").touch()

        run(sources_root=tmp_path)

        assert call_order == ["parse", "chunk", "embed", "write"], (
            f"Expected parse→chunk→embed→write, got: {call_order}"
        )

    @patch("ingestion.pipeline.write")
    @patch("ingestion.pipeline.embed")
    @patch("ingestion.pipeline.chunk_documents")
    @patch("ingestion.pipeline.parse_pdf_folder")
    @patch("ingestion.pipeline.count")
    def test_pipeline_skips_missing_source_folders(
        self, mock_count, mock_parse, mock_chunk, mock_embed, mock_write, tmp_path
    ):
        """
        If a source folder (e.g. finra/) doesn't exist, the pipeline should
        skip it gracefully rather than crashing. Only present folders are processed.
        Here, tmp_path has no subfolders — parse should never be called.
        """
        mock_write.return_value = 0
        mock_count.return_value = 0
        mock_chunk.return_value = []
        mock_embed.side_effect = lambda c: c

        result = run(sources_root=tmp_path)

        mock_parse.assert_not_called()
        assert result["documents_parsed"] == 0
        assert result["chunks_written"] == 0

    @patch("ingestion.pipeline.write")
    @patch("ingestion.pipeline.embed")
    @patch("ingestion.pipeline.chunk_documents")
    @patch("ingestion.pipeline.parse_pdf_folder")
    @patch("ingestion.pipeline.count")
    def test_pipeline_passes_correct_counts_to_summary(
        self, mock_count, mock_parse, mock_chunk, mock_embed, mock_write, tmp_path
    ):
        """
        The summary dict must accurately reflect what happened in this run.
        documents_parsed = number of Documents returned by the parser.
        chunks_created   = number of Chunks returned by the chunker.
        chunks_written   = number returned by write().
        """
        docs = [self._make_doc() for _ in range(4)]
        chunks = [self._make_chunk(d, i) for i, d in enumerate(docs * 3)]   # 12 chunks

        mock_parse.return_value = docs
        mock_chunk.return_value = chunks
        mock_embed.side_effect = lambda c: c
        mock_write.return_value = 12
        mock_count.return_value = 12

        sec_dir = tmp_path / "sec"
        sec_dir.mkdir()
        (sec_dir / "fake.pdf").touch()

        result = run(sources_root=tmp_path)

        assert result["documents_parsed"] == 4
        assert result["chunks_created"] == 12
        assert result["chunks_written"] == 12


# ══════════════════════════════════════════════════════════════════════════════
# TestPipelineIntegration
# Runs the full pipeline end-to-end against real PDFs.
# Slow — loads the embedding model and processes real documents.
# Skipped if no PDFs are present.
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineIntegration:

    @pytest.fixture(autouse=True)
    def use_test_db(self, tmp_path, monkeypatch):
        """
        Redirect LanceDB to a temp directory so the integration test
        never writes to the real ./data/lancedb database.
        """
        import lancedb as _lancedb

        test_db_path = tmp_path / "lancedb"

        def _test_connect():
            test_db_path.mkdir(parents=True, exist_ok=True)
            return _lancedb.connect(str(test_db_path))

        monkeypatch.setattr("retrieval.lancedb_retriever._connect", _test_connect)

    @pytest.mark.skipif(
        not any((REAL_SOURCES / folder).glob("*.pdf") for folder in ["sec", "cfpb", "fed", "finra"]),
        reason="No PDFs found in ingestion/sources/pdfs/"
    )
    def test_full_pipeline_produces_nonzero_chunks(self):
        """
        Running the full pipeline against real PDFs must result in at least
        one chunk written to LanceDB. Zero chunks means something in the
        parse → chunk → embed → write chain silently failed.
        """
        result = run(sources_root=REAL_SOURCES)
        assert result["chunks_written"] > 0, (
            "Pipeline ran but wrote 0 chunks. "
            "Check that PDFs are readable and the parser returns content."
        )

    @pytest.mark.skipif(
        not any((REAL_SOURCES / folder).glob("*.pdf") for folder in ["sec", "cfpb", "fed", "finra"]),
        reason="No PDFs found in ingestion/sources/pdfs/"
    )
    def test_full_pipeline_summary_counts_are_consistent(self):
        """
        The summary dict must be internally consistent:
          - chunks_created >= documents_parsed (each doc produces >= 1 chunk)
          - chunks_written == chunks_created (everything embedded gets stored)
          - elapsed_seconds > 0 (time was measured)
        """
        result = run(sources_root=REAL_SOURCES)
        assert result["chunks_created"] >= result["documents_parsed"], (
            "Fewer chunks than documents — some documents produced zero chunks"
        )
        assert result["chunks_written"] == result["chunks_created"], (
            "Not all chunks were written to LanceDB"
        )
        assert result["elapsed_seconds"] > 0

