"""
Tests for the LanceDB retriever (retrieval/lancedb_retriever.py).

These are integration tests — they actually create a temporary LanceDB
database on disk, run operations against it, then clean it up.

A temporary path is used (./data/lancedb_test) so tests never touch
the real database at ./data/lancedb.

Test structure:
  - TestWrite   → writing chunks to LanceDB
  - TestSearch  → searching for nearest neighbours
  - TestCount   → row count helper
"""

import math
import pytest
from pathlib import Path
from unittest.mock import patch

from ingestion.document import Chunk
from retrieval.lancedb_retriever import write, search, count


# ── Shared fixtures ───────────────────────────────────────────────────────────

def make_embedded_chunk(text: str, index: int, embedding: list[float] = None) -> Chunk:
    """
    Create a Chunk with a real embedding for testing.

    If no embedding is provided, generates a simple unit vector (all equal values).
    """
    if embedding is None:
        val = 1.0 / math.sqrt(384)   # unit vector: all dimensions equal
        embedding = [val] * 384
    chunk = Chunk(
        text=text,
        source="Test Source",
        section="page_1",
        url=f"file:///test/doc.pdf#page_1",
        doc_id="testdoc0001",
        chunk_index=index,
    )
    chunk.embedding = embedding
    return chunk


def make_chunk_no_embedding(text: str, index: int = 0) -> Chunk:
    """Create a Chunk without an embedding (embedding=None)."""
    return Chunk(
        text=text,
        source="Test Source",
        section="page_1",
        url="file:///test/doc.pdf#page_1",
        doc_id="testdoc0001",
        chunk_index=index,
    )


@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    """
    Redirect all LanceDB operations to a fresh temporary directory per test.

    Patches the private _connect() function so every write/search/count call
    in this test session uses an isolated temp DB — never the real ./data/lancedb.

    tmp_path is a pytest built-in that provides a unique temp directory per test,
    automatically cleaned up after each test completes.
    """
    import lancedb as _lancedb

    test_db_path = tmp_path / "lancedb"

    def _test_connect():
        test_db_path.mkdir(parents=True, exist_ok=True)
        return _lancedb.connect(str(test_db_path))

    monkeypatch.setattr("retrieval.lancedb_retriever._connect", _test_connect)


# ══════════════════════════════════════════════════════════════════════════════
# TestWrite
# Tests write() — persists chunks to LanceDB, overwriting any existing table.
# ══════════════════════════════════════════════════════════════════════════════

class TestWrite:

    def test_write_returns_chunk_count(self):
        """
        write() should return the exact number of chunks passed in.
        This is used by the pipeline to report how many chunks were stored.
        """
        chunks = [make_embedded_chunk("Mutual funds pool money.", i) for i in range(3)]
        result = write(chunks)
        assert result == 3

    def test_write_persists_chunks_to_disk(self):
        """
        After write() completes, count() should return the number of rows written.
        Verifies that data actually reaches the database, not just memory.
        """
        chunks = [make_embedded_chunk("Some financial content.", i) for i in range(5)]
        write(chunks)
        assert count() == 5

    def test_write_overwrites_existing_table(self):
        """
        Running write() twice should result in only the second batch being stored.
        This is the core overwrite guarantee: no stale data from previous runs.
        The first write stores 3 chunks; the second write stores 2 different chunks.
        Final count should be 2, not 5.
        """
        first_batch = [make_embedded_chunk("First batch content.", i) for i in range(3)]
        write(first_batch)
        assert count() == 3

        second_batch = [make_embedded_chunk("Second batch content.", i) for i in range(2)]
        write(second_batch)
        assert count() == 2, (
            "Expected 2 chunks after overwrite, but old data was not cleared."
        )

    def test_write_raises_if_embedding_is_missing(self):
        """
        write() must raise ValueError if any chunk has embedding=None.
        Storing unembedded chunks would silently corrupt the vector index —
        searches would either fail or return garbage results.
        """
        chunks = [make_chunk_no_embedding("Chunk without embedding.")]
        with pytest.raises(ValueError, match="no embedding"):
            write(chunks)

    def test_write_empty_list_creates_empty_table(self):
        """
        write() with an empty list should succeed and create a table with 0 rows.
        The pipeline may produce zero chunks if all source documents are empty.
        """
        write([])
        assert count() == 0


# ══════════════════════════════════════════════════════════════════════════════
# TestSearch
# Tests search() — returns the top-k most similar chunks for a query vector.
# ══════════════════════════════════════════════════════════════════════════════

class TestSearch:

    def test_search_returns_correct_number_of_results(self):
        """
        search() with k=3 should return exactly 3 chunks (assuming at least 3 exist).
        The retrieval layer depends on this to inject the right number of
        context chunks into the LLM prompt.
        """
        chunks = [make_embedded_chunk(f"Content chunk {i}.", i) for i in range(10)]
        write(chunks)
        query_vec = [1.0 / math.sqrt(384)] * 384
        results = search(query_vec, k=3)
        assert len(results) == 3

    def test_search_returns_chunk_objects(self):
        """
        Every item returned by search() must be a Chunk instance.
        The generation layer passes these directly to the LLM prompt builder.
        """
        chunks = [make_embedded_chunk("SEC investor content.", 0)]
        write(chunks)
        query_vec = [1.0 / math.sqrt(384)] * 384
        results = search(query_vec, k=1)
        assert all(isinstance(r, Chunk) for r in results)

    def test_search_returned_chunks_have_correct_text(self):
        """
        The text field on returned chunks must match what was written.
        If text is missing or corrupted, the LLM receives empty context
        and cannot generate a grounded answer.
        """
        chunk = make_embedded_chunk("Dollar-cost averaging reduces timing risk.", 0)
        write([chunk])
        query_vec = [1.0 / math.sqrt(384)] * 384
        results = search(query_vec, k=1)
        assert results[0].text == "Dollar-cost averaging reduces timing risk."

    def test_search_raises_if_table_does_not_exist(self):
        """
        search() must raise RuntimeError with a clear message if the table
        hasn't been created yet. This gives the user an actionable error:
        "run the ingestion pipeline first" — rather than a cryptic LanceDB error.
        """
        with pytest.raises(RuntimeError, match="pipeline"):
            search([1.0 / math.sqrt(384)] * 384, k=5)

    def test_search_returns_fewer_results_when_table_is_small(self):
        """
        If the table has fewer rows than k, search() should return however
        many rows exist — not crash or pad with empty results.
        """
        chunks = [make_embedded_chunk("Only chunk in DB.", 0)]
        write(chunks)
        query_vec = [1.0 / math.sqrt(384)] * 384
        results = search(query_vec, k=5)   # asking for 5, only 1 exists
        assert len(results) == 1


# ══════════════════════════════════════════════════════════════════════════════
# TestCount
# Tests the count() helper — returns total rows in the table.
# ══════════════════════════════════════════════════════════════════════════════

class TestCount:

    def test_count_returns_zero_when_table_missing(self):
        """
        count() should return 0 if the table doesn't exist yet.
        Used by the pipeline to detect a fresh/empty state without crashing.
        """
        assert count() == 0

    def test_count_matches_written_chunks(self):
        """
        count() should equal the number of chunks passed to write().
        Provides a quick sanity check after running the pipeline.
        """
        chunks = [make_embedded_chunk("Content.", i) for i in range(7)]
        write(chunks)
        assert count() == 7
