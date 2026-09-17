"""
Tests for query logging (query_logging/query_logger.py).

Tests check:
  - format_chunk_trace creates clean, truncated preview
  - log_query appends valid JSONL lines
  - logs accurately preserve latency, query, answer, and chunk metadata
  - isolates filesystem writes using tmp_path
"""

import json
import pytest
from pathlib import Path
from ingestion.document import Chunk
from query_logging.query_logger import format_chunk_trace, log_query, QueryLogEntry


def make_chunk(text: str, source: str = "SEC", section: str = "page_2") -> Chunk:
    """Helper to create a test Chunk."""
    return Chunk(
        text=text,
        source=source,
        section=section,
        url="file:///mock.pdf",
        doc_id="doc_xyz",
        chunk_index=3,
    )


class TestFormatChunkTrace:

    def test_trace_contains_metadata_and_preview(self):
        """Chunk trace should include chunk_id, doc_id, source, section, url, preview."""
        chunk = make_chunk("Short chunk content.")
        trace = format_chunk_trace(chunk)

        assert trace["chunk_id"] == chunk.chunk_id
        assert trace["doc_id"] == "doc_xyz"
        assert trace["source"] == "SEC"
        assert trace["section"] == "page_2"
        assert trace["url"] == "file:///mock.pdf"
        assert trace["text_preview"] == "Short chunk content."

    def test_trace_truncates_long_texts(self):
        """Long text should be truncated to max_preview_chars with ellipsis."""
        long_text = "Word " * 100
        chunk = make_chunk(long_text)
        trace = format_chunk_trace(chunk, max_preview_chars=50)

        assert len(trace["text_preview"]) <= 53  # 50 chars + "..."
        assert trace["text_preview"].endswith("...")


class TestLogQuery:

    def test_log_query_creates_valid_jsonl(self, tmp_path):
        """log_query should create log file if missing and write valid JSON."""
        log_file = tmp_path / "logs" / "test_log.jsonl"
        chunk = make_chunk("Investing carries market risk.")

        entry = log_query(
            query="What is risk?",
            answer="Risk refers to uncertainty of returns.",
            chunks=[chunk],
            retrieval_latency_ms=45.2,
            generation_latency_ms=320.5,
            model="qwen2.5:7b",
            log_file_path=log_file,
        )

        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

        logged_data = json.loads(lines[0])
        assert logged_data["query"] == "What is risk?"
        assert logged_data["answer"] == "Risk refers to uncertainty of returns."
        assert logged_data["retrieval_latency_ms"] == 45.2
        assert logged_data["generation_latency_ms"] == 320.5
        assert logged_data["total_latency_ms"] == 365.7
        assert logged_data["model"] == "qwen2.5:7b"
        assert logged_data["k"] == 1
        assert len(logged_data["retrieved_chunks"]) == 1
        assert logged_data["retrieved_chunks"][0]["source"] == "SEC"

    def test_log_query_appends_multiple_entries(self, tmp_path):
        """Multiple calls to log_query should append lines rather than overwriting."""
        log_file = tmp_path / "test_log.jsonl"

        log_query("Q1", "A1", [], 10.0, 50.0, log_file_path=log_file)
        log_query("Q2", "A2", [], 15.0, 60.0, log_file_path=log_file)

        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        entry2 = json.loads(lines[1])
        assert entry1["query"] == "Q1"
        assert entry2["query"] == "Q2"

