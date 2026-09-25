"""
Unit tests for retrieval/pinecone_retriever.py

All Pinecone SDK calls are mocked so no API key or network access is required.
Tests mirror the structure of test_lancedb_retriever.py for symmetry.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from ingestion.document import Chunk


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_chunk(idx: int, embedding: list[float] | None = None) -> Chunk:
    """Helper: create a minimal Chunk with optional embedding."""
    return Chunk(
        text=f"Test chunk text number {idx}.",
        source="SEC/Investor.gov",
        section=f"page_{idx}",
        url=f"https://example.com/doc_{idx}.pdf",
        doc_id=f"deadbeef{idx:04d}",
        chunk_index=idx,
        embedding=embedding or [0.1] * 384,
    )


def _make_query_vector() -> list[float]:
    return [0.5] * 384


# ── Mock helpers ─────────────────────────────────────────────────────────────

def _make_index_list(names: list[str]):
    """Return a mock IndexList with a .names() method — mirrors SDK v3 API."""
    mock = MagicMock()
    mock.names.return_value = names
    return mock


def _make_query_response(matches: list[dict]):
    """Return a mock QueryResponse with a .matches list — mirrors SDK v3 API.

    Each item in matches is a dict with 'id', 'score', and 'metadata'.
    """
    scored_vectors = []
    for m in matches:
        sv = MagicMock()
        sv.id = m["id"]
        sv.score = m.get("score", 0.9)
        sv.metadata = m.get("metadata", {})
        scored_vectors.append(sv)
    resp = MagicMock()
    resp.matches = scored_vectors
    return resp


def _mock_pinecone(existing_indexes: list[str] | None = None):
    """Return a fully-configured mock Pinecone client."""
    pc = MagicMock()
    names = existing_indexes or []
    pc.list_indexes.return_value = _make_index_list(names)
    mock_index = MagicMock()
    pc.Index.return_value = mock_index
    mock_index.describe_index_stats.return_value = MagicMock(
        total_vector_count=len(names) * 10
    )
    return pc, mock_index



# ── write() tests ─────────────────────────────────────────────────────────────

class TestPineconeWrite:

    @patch("retrieval.pinecone_retriever._get_client")
    @patch("retrieval.pinecone_retriever._wait_for_index_ready")
    def test_write_creates_index_when_absent(self, mock_wait, mock_get_client):
        """write() should call create_index when the index does not yet exist."""
        pc, mock_index = _mock_pinecone(existing_indexes=[])
        mock_get_client.return_value = pc

        chunks = [_make_chunk(i) for i in range(5)]
        from retrieval import pinecone_retriever
        result = pinecone_retriever.write(chunks)

        pc.create_index.assert_called_once()
        mock_wait.assert_called_once()
        assert result == 5

    @patch("retrieval.pinecone_retriever._get_client")
    @patch("retrieval.pinecone_retriever._wait_for_index_ready")
    def test_write_skips_create_when_index_exists(self, mock_wait, mock_get_client):
        """write() should NOT call create_index when the index already exists."""
        pc, mock_index = _mock_pinecone(existing_indexes=["fin-rag"])
        mock_get_client.return_value = pc

        chunks = [_make_chunk(i) for i in range(3)]
        from retrieval import pinecone_retriever
        pinecone_retriever.write(chunks)

        pc.create_index.assert_not_called()
        mock_wait.assert_not_called()

    @patch("retrieval.pinecone_retriever._get_client")
    @patch("retrieval.pinecone_retriever._wait_for_index_ready")
    def test_write_upserts_in_batches(self, _mock_wait, mock_get_client):
        """write() should split large chunk lists into batches of 100."""
        pc, mock_index = _mock_pinecone(existing_indexes=["fin-rag"])
        mock_get_client.return_value = pc

        chunks = [_make_chunk(i) for i in range(250)]
        from retrieval import pinecone_retriever
        result = pinecone_retriever.write(chunks)

        # 250 chunks → 3 batches: 100, 100, 50
        assert mock_index.upsert.call_count == 3
        batch_sizes = [len(c.kwargs.get("vectors", c.args[0] if c.args else [])) for c in mock_index.upsert.call_args_list]
        assert batch_sizes == [100, 100, 50]
        assert result == 250

    @patch("retrieval.pinecone_retriever._get_client")
    @patch("retrieval.pinecone_retriever._wait_for_index_ready")
    def test_write_deletes_all_before_upsert(self, _mock_wait, mock_get_client):
        """write() should call delete(delete_all=True) before upserting."""
        pc, mock_index = _mock_pinecone(existing_indexes=["fin-rag"])
        mock_get_client.return_value = pc

        chunks = [_make_chunk(0)]
        from retrieval import pinecone_retriever
        pinecone_retriever.write(chunks)

        mock_index.delete.assert_called_once_with(delete_all=True)

    @patch("retrieval.pinecone_retriever._get_client")
    def test_write_raises_on_missing_embedding(self, mock_get_client):
        """write() should raise ValueError if any chunk lacks an embedding."""
        pc, _ = _mock_pinecone(existing_indexes=["fin-rag"])
        mock_get_client.return_value = pc

        chunks = [_make_chunk(0, embedding=None)]
        chunks[0].embedding = None  # force null

        from retrieval import pinecone_retriever
        with pytest.raises(ValueError, match="no embedding"):
            pinecone_retriever.write(chunks)

    @patch("retrieval.pinecone_retriever._get_client")
    @patch("retrieval.pinecone_retriever._wait_for_index_ready")
    def test_write_vector_metadata_shape(self, _mock_wait, mock_get_client):
        """Each upserted vector should carry all six Chunk metadata fields."""
        pc, mock_index = _mock_pinecone(existing_indexes=["fin-rag"])
        mock_get_client.return_value = pc

        chunk = _make_chunk(42)
        from retrieval import pinecone_retriever
        pinecone_retriever.write([chunk])

        upserted = mock_index.upsert.call_args
        vectors = upserted.kwargs.get("vectors", upserted.args[0] if upserted.args else [])
        assert len(vectors) == 1
        vec = vectors[0]
        assert vec["id"] == chunk.chunk_id
        assert vec["values"] == chunk.embedding
        meta = vec["metadata"]
        assert meta["text"] == chunk.text
        assert meta["source"] == chunk.source
        assert meta["section"] == chunk.section
        assert meta["url"] == chunk.url
        assert meta["doc_id"] == chunk.doc_id
        assert meta["chunk_index"] == chunk.chunk_index


# ── search() tests ────────────────────────────────────────────────────────────

class TestPineconeSearch:

    @patch("retrieval.pinecone_retriever._get_client")
    def test_search_returns_chunks(self, mock_get_client):
        """search() should reconstruct Chunk objects from Pinecone match metadata."""
        pc, mock_index = _mock_pinecone(existing_indexes=["fin-rag"])
        mock_get_client.return_value = pc

        mock_index.query.return_value = _make_query_response([
            {
                "id": "abc_0000",
                "score": 0.95,
                "metadata": {
                    "text": "Diversification reduces risk.",
                    "source": "SEC/Investor.gov",
                    "section": "page_3",
                    "url": "https://investor.gov/doc.pdf",
                    "doc_id": "abc",
                    "chunk_index": 0,
                },
            },
            {
                "id": "def_0001",
                "score": 0.88,
                "metadata": {
                    "text": "Bonds are less volatile than stocks.",
                    "source": "Federal Reserve",
                    "section": "page_7",
                    "url": "https://fed.gov/doc.pdf",
                    "doc_id": "def",
                    "chunk_index": 1,
                },
            },
        ])

        from retrieval import pinecone_retriever
        results = pinecone_retriever.search(_make_query_vector(), k=2)

        assert len(results) == 2
        assert results[0].text == "Diversification reduces risk."
        assert results[0].source == "SEC/Investor.gov"
        assert results[0].chunk_index == 0
        assert results[1].text == "Bonds are less volatile than stocks."

    @patch("retrieval.pinecone_retriever._get_client")
    def test_search_empty_results(self, mock_get_client):
        """search() should return an empty list when Pinecone returns no matches."""
        pc, mock_index = _mock_pinecone(existing_indexes=["fin-rag"])
        mock_get_client.return_value = pc
        mock_index.query.return_value = _make_query_response([])

        from retrieval import pinecone_retriever
        results = pinecone_retriever.search(_make_query_vector(), k=5)
        assert results == []


    @patch("retrieval.pinecone_retriever._get_client")
    def test_search_raises_when_index_missing(self, mock_get_client):
        """search() should raise RuntimeError when the index doesn't exist."""
        pc, _ = _mock_pinecone(existing_indexes=[])
        mock_get_client.return_value = pc

        from retrieval import pinecone_retriever
        with pytest.raises(RuntimeError, match="does not exist"):
            pinecone_retriever.search(_make_query_vector(), k=5)

    @patch("retrieval.pinecone_retriever._get_client")
    def test_search_uses_default_top_k(self, mock_get_client):
        """search() should fall back to settings.top_k when k is not provided."""
        pc, mock_index = _mock_pinecone(existing_indexes=["fin-rag"])
        mock_get_client.return_value = pc
        mock_index.query.return_value = _make_query_response([])

        from retrieval import pinecone_retriever
        from config import settings
        pinecone_retriever.search(_make_query_vector())

        mock_index.query.assert_called_once()
        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs["top_k"] == settings.top_k

    @patch("retrieval.pinecone_retriever._get_client")
    def test_search_passes_include_metadata(self, mock_get_client):
        """search() should request metadata from Pinecone so Chunks can be reconstructed."""
        pc, mock_index = _mock_pinecone(existing_indexes=["fin-rag"])
        mock_get_client.return_value = pc
        mock_index.query.return_value = _make_query_response([])

        from retrieval import pinecone_retriever
        pinecone_retriever.search(_make_query_vector(), k=3)

        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs.get("include_metadata") is True


# ── count() tests ─────────────────────────────────────────────────────────────

class TestPineconeCount:

    @patch("retrieval.pinecone_retriever._get_client")
    def test_count_returns_vector_count(self, mock_get_client):
        """count() should return the total_vector_count from describe_index_stats."""
        pc, mock_index = _mock_pinecone(existing_indexes=["fin-rag"])
        # Use a MagicMock with total_vector_count as an attribute (SDK v3 returns a typed object)
        mock_index.describe_index_stats.return_value = MagicMock(total_vector_count=488)
        mock_get_client.return_value = pc

        from retrieval import pinecone_retriever
        result = pinecone_retriever.count()
        assert result == 488


    @patch("retrieval.pinecone_retriever._get_client")
    def test_count_returns_zero_when_index_missing(self, mock_get_client):
        """count() should return 0 when the index does not exist."""
        pc, _ = _mock_pinecone(existing_indexes=[])
        mock_get_client.return_value = pc

        from retrieval import pinecone_retriever
        result = pinecone_retriever.count()
        assert result == 0


# ── _get_client() tests ───────────────────────────────────────────────────────

class TestGetClient:

    def test_get_client_raises_without_api_key(self):
        """_get_client() should raise RuntimeError if PINECONE_API_KEY is empty."""
        from retrieval import pinecone_retriever
        from config import settings

        original = settings.pinecone_api_key
        try:
            settings.pinecone_api_key = ""
            with pytest.raises(RuntimeError, match="PINECONE_API_KEY"):
                pinecone_retriever._get_client()
        finally:
            settings.pinecone_api_key = original

