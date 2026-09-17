"""
Tests for the embedder (ingestion/embedder.py).

Structured in two classes:
  - TestEmbedQuery  → tests the single-string query embedding function
  - TestEmbed       → tests batch chunk embedding

These are integration tests — the model (bge-small-en-v1.5) is actually
loaded and run. First run will download the model (~33MB) if not cached.
Subsequent runs use the local cache and are fast.

We do NOT test for specific numeric values in embeddings because those
are model internals. Instead we test the structural properties that our
system depends on: shape, type, normalization, and semantic ordering.
"""

import math
import pytest
from ingestion.document import Document, Chunk
from ingestion.embedder import embed, embed_query


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_chunk(text: str, index: int = 0) -> Chunk:
    """
    Create a minimal Chunk for testing with no embedding set.
    """
    return Chunk(
        text=text,
        source="Test",
        section="page_1",
        url="file:///test.pdf#page_1",
        doc_id="testdoc0001",
        chunk_index=index,
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Used to verify semantic relationships between embeddings.
    """
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ══════════════════════════════════════════════════════════════════════════════
# TestEmbedQuery
# Tests embed_query() — embeds a single string for retrieval use.
# This is the function called at query time, not during ingestion.
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbedQuery:

    def test_returns_a_list(self):
        """
        embed_query() must return a plain Python list, not a numpy array.
        The retrieval layer and vector stores expect list[float], not ndarray.
        """
        result = embed_query("What is a mutual fund?")
        assert isinstance(result, list)

    def test_embedding_has_correct_dimensions(self):
        """
        bge-small-en-v1.5 produces 384-dimensional vectors.
        If this fails, the wrong model was loaded or the model was updated.
        A dimension mismatch between query and stored chunk vectors will
        cause all similarity searches to fail or return wrong results.
        """
        result = embed_query("What is a mutual fund?")
        assert len(result) == 384, (
            f"Expected 384 dimensions, got {len(result)}. "
            "Check that EMBEDDING_MODEL is set to BAAI/bge-small-en-v1.5"
        )

    def test_embedding_values_are_floats(self):
        """
        Every element in the embedding must be a Python float.
        The vector store clients (LanceDB, Pinecone) expect float values.
        """
        result = embed_query("What is a mutual fund?")
        assert all(isinstance(v, float) for v in result)

    def test_embedding_is_normalized(self):
        """
        With normalize_embeddings=True, the vector magnitude should be ~1.0.
        Normalized vectors make cosine similarity equivalent to dot product,
        which is what most vector stores use internally for fast ANN search.
        Magnitude exactly 1.0 is not guaranteed due to floating point —
        we check within a small tolerance.
        """
        result = embed_query("What is a mutual fund?")
        magnitude = math.sqrt(sum(v ** 2 for v in result))
        assert abs(magnitude - 1.0) < 1e-4, (
            f"Expected unit vector (magnitude ≈ 1.0), got {magnitude:.6f}. "
            "Embeddings may not be normalized."
        )

    def test_similar_queries_have_high_similarity(self):
        """
        Two semantically similar questions should produce embeddings that are
        close in vector space (high cosine similarity). This is the fundamental
        property the entire RAG retrieval system is built on.

        If this fails, the model is not understanding semantic similarity —
        retrieval quality will be very poor.
        """
        vec1 = embed_query("What is a mutual fund?")
        vec2 = embed_query("Can you explain what mutual funds are?")
        similarity = cosine_similarity(vec1, vec2)
        assert similarity > 0.85, (
            f"Similar queries had low similarity: {similarity:.3f}. "
            "Expected > 0.85."
        )

    def test_unrelated_queries_have_low_similarity(self):
        """
        Two semantically unrelated strings should produce embeddings that are
        far apart in vector space (low cosine similarity). This ensures the
        model discriminates between topics and doesn't return irrelevant chunks.
        """
        vec1 = embed_query("What is a mutual fund?")
        vec2 = embed_query("How do I bake chocolate chip cookies?")
        similarity = cosine_similarity(vec1, vec2)
        assert similarity < 0.75, (
            f"Unrelated queries had unexpectedly high similarity: {similarity:.3f}. "
            "Expected < 0.75."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TestEmbed
# Tests embed() — batch-embeds a list of Chunk objects.
# This is the function called during ingestion, not at query time.
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbed:

    def test_empty_list_returns_empty_list(self):
        """
        Passing an empty list should return an empty list without errors.
        The pipeline may call embed() with zero chunks if a document was empty.
        """
        result = embed([])
        assert result == []

    def test_all_chunks_receive_embeddings(self):
        """
        After embed() runs, every chunk's embedding field must be set (not None).
        If any chunk still has embedding=None, it will be rejected by the
        vector store and silently dropped from the index.
        """
        chunks = [make_chunk("Mutual funds pool investor money.", 0),
                  make_chunk("The SEC regulates securities.", 1)]
        embed(chunks)
        for chunk in chunks:
            assert chunk.embedding is not None, (
                f"Chunk {chunk.chunk_index} still has embedding=None after embed()"
            )

    def test_embeddings_have_correct_dimensions(self):
        """
        Each chunk's embedding must be 384 dimensions — matching bge-small-en-v1.5.
        A mismatch means the stored vectors are incompatible with query vectors,
        breaking all similarity searches.
        """
        chunks = [make_chunk("Dollar-cost averaging reduces timing risk.", 0)]
        embed(chunks)
        assert len(chunks[0].embedding) == 384

    def test_embed_returns_same_list(self):
        """
        embed() mutates chunks in place AND returns the same list object.
        This allows chaining: chunks = embed(chunk_documents(docs)).
        Verifies the return value is the same object, not a copy.
        """
        chunks = [make_chunk("Some financial content.", 0)]
        returned = embed(chunks)
        assert returned is chunks, "embed() should return the same list, not a copy"

    def test_embeddings_are_lists_of_floats(self):
        """
        Each embedding must be a list of Python floats (not numpy arrays).
        LanceDB and Pinecone clients both expect list[float].
        """
        chunks = [make_chunk("Index funds track market benchmarks.", 0)]
        embed(chunks)
        assert isinstance(chunks[0].embedding, list)
        assert all(isinstance(v, float) for v in chunks[0].embedding)

    def test_different_texts_produce_different_embeddings(self):
        """
        Two chunks with different text must produce different embedding vectors.
        If two unrelated texts produce identical vectors, the model is broken
        or all text is being truncated to the same prefix.
        """
        chunks = [
            make_chunk("A stock represents ownership in a company.", 0),
            make_chunk("A bond is a fixed income debt instrument.", 1),
        ]
        embed(chunks)
        assert chunks[0].embedding != chunks[1].embedding, (
            "Different texts produced identical embeddings"
        )

    def test_same_text_produces_consistent_embedding(self):
        """
        Embedding the same text twice should produce the same vector.
        SentenceTransformers models are deterministic — identical input
        always yields identical output. If this fails, there may be
        random dropout active in the model (should be off in eval mode).
        """
        text = "What is dollar-cost averaging?"
        chunk1 = make_chunk(text, 0)
        chunk2 = make_chunk(text, 1)
        embed([chunk1])
        embed([chunk2])
        assert chunk1.embedding == chunk2.embedding, (
            "Same text produced different embeddings — model may not be deterministic"
        )

