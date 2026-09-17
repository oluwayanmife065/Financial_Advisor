"""
Embedder — converts Chunk text into dense vector embeddings.

Uses the SentenceTransformers library with the BAAI/bge-small-en-v1.5 model.
This model produces 384-dimensional vectors optimised specifically for
retrieval tasks (finding relevant passages), which makes it a better choice
than all-MiniLM-L6-v2 for RAG systems.

The model is loaded once at import time (lazy-loaded on first use) and reused
across all embed() calls. Loading takes ~1-2s; individual embeddings are fast.
"""

from sentence_transformers import SentenceTransformer
from ingestion.document import Chunk
from config import settings


# Module-level model instance — loaded once, reused for all calls.
# This avoids the ~1-2s reload penalty on every embed() call.
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    Lazily load and cache the embedding model.

    The model is only loaded the first time embed() or embed_query() is called.
    Subsequent calls reuse the same instance.

    Returns:
        Loaded SentenceTransformer model.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed(chunks: list[Chunk]) -> list[Chunk]:
    """
    Compute embeddings for a list of Chunks, mutating each chunk in place.

    Sets the `embedding` field on every Chunk to a list of floats (384 dims).
    Processes all chunks in a single batch call for efficiency — much faster
    than embedding one chunk at a time.

    Args:
        chunks: List of Chunk objects. Each must have a non-empty `text` field.

    Returns:
        The same list of Chunk objects, each now with `embedding` populated.
    """
    if not chunks:
        return chunks

    model = _get_model()

    # Extract all texts in order, batch-embed them in one call
    texts = [chunk.text for chunk in chunks]
    vectors = model.encode(
        texts,
        batch_size=64,          # process 64 chunks at a time
        show_progress_bar=False,
        normalize_embeddings=True,  # L2-normalize for cosine similarity
    )

    # Assign each vector back to its corresponding chunk
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector.tolist()

    return chunks


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string for use at retrieval time.

    The query is embedded with the same model and normalization as the chunks,
    so cosine similarity comparisons are valid.

    Args:
        query: The user's question or search string.

    Returns:
        384-dimensional embedding as a list of floats.
    """
    model = _get_model()
    vector = model.encode(
        query,
        normalize_embeddings=True,
    )
    return vector.tolist()

