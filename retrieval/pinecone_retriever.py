"""
Pinecone retriever — handles writing chunks to Pinecone and searching them.

Pinecone is a fully managed, serverless cloud vector database. This module
mirrors the module-level API of lancedb_retriever.py (write / search / count)
so that the eval runner and app can swap backends by changing a single import.

Architecture:
  - Uses the Pinecone Python SDK v3+ (``from pinecone import Pinecone``).
  - Creates a *serverless* index on the first write() call if it does not yet
    exist (cloud=aws, region=us-east-1 by default — both configurable via
    .env / environment variables).
  - Upserts are batched in groups of 100 to stay well within Pinecone's
    request-size limits.
  - Each vector's metadata carries all six Chunk fields so that search() can
    reconstruct complete Chunk objects from Pinecone's results without a
    separate database lookup.
  - cosine similarity is used to match LanceDB's normalised-vector behaviour.

SDK v3 notes:
  - pc.list_indexes() → IndexList; use .names() to get list[str]
  - pc.describe_index(name) → IndexModel; status accessed via .status.ready
  - index.query() → QueryResponse; matches accessed via .matches (list of ScoredVector)
  - ScoredVector.metadata is a plain dict
  - index.describe_index_stats() → DescribeIndexStatsResponse; .total_vector_count is an int

Environment variables (all read via config.py / .env):
  PINECONE_API_KEY      — required; your Pinecone project API key
  PINECONE_INDEX_NAME   — name of the Pinecone index (default: fin-rag)
  PINECONE_CLOUD        — serverless cloud provider (default: aws)
  PINECONE_REGION       — serverless region (default: us-east-1)
"""

import time

from pinecone import Pinecone, ServerlessSpec

from ingestion.document import Chunk
from config import settings

# Batch size for upsert calls — Pinecone's recommended upper bound per request
_UPSERT_BATCH_SIZE = 100


def _get_client() -> Pinecone:
    """Return an authenticated Pinecone client using the key from config."""
    if not settings.pinecone_api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. "
            "Add it to your .env file or environment variables."
        )
    return Pinecone(api_key=settings.pinecone_api_key)


def _get_or_create_index(pc: Pinecone):
    """Return the Pinecone Index object, creating it if it doesn't exist.

    Index is serverless (no pod configuration needed). Uses cosine similarity
    to match LanceDB's behaviour on unit-normalised bge-small-en-v1.5 vectors.

    Args:
        pc: Authenticated Pinecone client.

    Returns:
        A Pinecone Index object connected to settings.pinecone_index_name.
    """
    index_name = settings.pinecone_index_name

    # IndexList.names() returns a plain list[str] in SDK v3
    existing_names = pc.list_indexes().names()

    if index_name not in existing_names:
        pc.create_index(
            name=index_name,
            dimension=settings.embedding_dim,  # 384 for bge-small-en-v1.5
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        # Wait for the index to be ready before returning
        _wait_for_index_ready(pc, index_name)

    return pc.Index(index_name)


def _wait_for_index_ready(pc: Pinecone, index_name: str, timeout: int = 120) -> None:
    """Poll until the index status is Ready or the timeout elapses.

    Args:
        pc: Authenticated Pinecone client.
        index_name: Name of the index to wait for.
        timeout: Maximum seconds to wait before raising RuntimeError.

    Raises:
        RuntimeError: If the index is not ready within the timeout window.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        # describe_index() returns an IndexModel — status is an attribute, not a dict
        desc = pc.describe_index(index_name)
        if desc.status.ready:
            return
        time.sleep(2)
    raise RuntimeError(
        f"Pinecone index '{index_name}' did not become ready within {timeout}s."
    )


def write(chunks: list[Chunk]) -> int:
    """Upsert embedded Chunks into the Pinecone index.

    Creates the serverless index on first call if it doesn't exist.
    Deletes all existing vectors before upserting so the index always
    mirrors the current corpus (equivalent to LanceDB's overwrite strategy).

    Args:
        chunks: List of Chunk objects. Each must have embedding set (not None).

    Returns:
        Number of chunks upserted.

    Raises:
        ValueError: If any chunk is missing its embedding.
        RuntimeError: If PINECONE_API_KEY is not configured.
    """
    missing = [c.chunk_id for c in chunks if c.embedding is None]
    if missing:
        raise ValueError(
            f"{len(missing)} chunks have no embedding. "
            "Run embed() before write(). "
            f"First missing: {missing[0]}"
        )

    pc = _get_client()
    index = _get_or_create_index(pc)

    # Clear all existing vectors to keep index in sync with the corpus.
    # Only call delete if the index already has content — Pinecone returns 404
    # when attempting to delete from an empty or non-existent namespace.
    stats = index.describe_index_stats()
    if stats.total_vector_count > 0:
        index.delete(delete_all=True)

    # Build Pinecone vector records.
    # Metadata stores all Chunk fields so search() can reconstruct Chunk objects.
    vectors = [
        {
            "id": chunk.chunk_id,
            "values": chunk.embedding,
            "metadata": {
                "text": chunk.text,
                "source": chunk.source,
                "section": chunk.section,
                "url": chunk.url,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
            },
        }
        for chunk in chunks
    ]

    # Upsert in batches of _UPSERT_BATCH_SIZE
    for batch_start in range(0, len(vectors), _UPSERT_BATCH_SIZE):
        batch = vectors[batch_start : batch_start + _UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch)

    return len(vectors)


def search(query_vector: list[float], k: int = None) -> list[Chunk]:
    """Search the Pinecone index for the top-k chunks closest to the query.

    Args:
        query_vector: 384-dimensional query embedding from embed_query().
        k: Number of results to return. Defaults to settings.top_k.

    Returns:
        List of Chunk objects ordered by relevance (most similar first).
        Embeddings are NOT repopulated on returned chunks (not needed for display).

    Raises:
        RuntimeError: If PINECONE_API_KEY is not configured.
        RuntimeError: If the index does not exist (write() hasn't been called).
    """
    if k is None:
        k = settings.top_k

    pc = _get_client()
    index_name = settings.pinecone_index_name

    if index_name not in pc.list_indexes().names():
        raise RuntimeError(
            f"Pinecone index '{index_name}' does not exist. "
            "Run write() / the ingestion pipeline first."
        )

    index = pc.Index(index_name)

    # query() returns a QueryResponse object; .matches is a list of ScoredVector
    response = index.query(
        vector=query_vector,
        top_k=k,
        include_metadata=True,
    )

    chunks = []
    for match in response.matches:
        # ScoredVector.metadata is a plain dict
        meta = match.metadata or {}
        chunk = Chunk(
            text=meta.get("text", ""),
            source=meta.get("source", ""),
            section=meta.get("section", ""),
            url=meta.get("url", ""),
            doc_id=meta.get("doc_id", ""),
            chunk_index=int(meta.get("chunk_index", 0)),
        )
        chunks.append(chunk)

    return chunks


def count() -> int:
    """Return the total number of vectors currently stored in the index.

    Returns:
        Vector count, or 0 if the index does not exist.

    Raises:
        RuntimeError: If PINECONE_API_KEY is not configured.
    """
    pc = _get_client()
    index_name = settings.pinecone_index_name

    if index_name not in pc.list_indexes().names():
        return 0

    index = pc.Index(index_name)
    # DescribeIndexStatsResponse exposes total_vector_count as an int attribute
    stats = index.describe_index_stats()
    return stats.total_vector_count
