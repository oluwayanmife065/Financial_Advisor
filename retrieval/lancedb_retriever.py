"""
LanceDB retriever — handles writing chunks to LanceDB and searching them.

LanceDB is an embedded vector database: no server, no Docker, just a file
on disk at the path set in config.py (default: ./data/lancedb).

Responsibilities:
  - write()  : Takes embedded Chunks and persists them to LanceDB.
               Always overwrites the existing table for simplicity.
  - search() : Takes a query vector and returns the top-k most similar Chunks.

Why overwrite instead of append?
  doc_id is a hash of url + date_ingested, not a sequential counter.
  Adding a new source (e.g. Robinhood) would change what gets processed first
  and make deduplication unreliable. A full rebuild ensures the DB always
  reflects exactly what's in the source files — clean and predictable.
"""

import lancedb
import pyarrow as pa
from pathlib import Path

from ingestion.document import Chunk
from config import settings


# PyArrow schema — defines the column types for the LanceDB table.
# Must match the Chunk dataclass fields exactly.
# pa.list_(pa.float32(), 384) is the vector column LanceDB indexes for ANN search.
SCHEMA = pa.schema([
    pa.field("chunk_id",  pa.string()),
    pa.field("text",      pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), 384)),
    pa.field("source",    pa.string()),
    pa.field("section",   pa.string()),
    pa.field("url",       pa.string()),
    pa.field("doc_id",    pa.string()),
])


def _connect():
    """
    Open (or create) the LanceDB database at the configured path.

    Creates the directory if it doesn't already exist.

    Returns:
        lancedb.DBConnection object.
    """
    db_path = Path(settings.lancedb_path)
    db_path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(db_path))


def write(chunks: list[Chunk]) -> int:
    """
    Persist embedded Chunks to LanceDB, overwriting any existing table.

    Drops and recreates the table on every call. This ensures the DB always
    reflects exactly the current corpus — no stale data, no deduplication logic.

    Args:
        chunks: List of Chunk objects. Each must have embedding set (not None).

    Returns:
        Number of chunks written.

    Raises:
        ValueError: If any chunk is missing its embedding.
    """
    missing = [c.chunk_id for c in chunks if c.embedding is None]
    if missing:
        raise ValueError(
            f"{len(missing)} chunks have no embedding. "
            "Run embed() before write(). "
            f"First missing: {missing[0]}"
        )

    db = _connect()

    # Drop existing table if it exists — full overwrite strategy
    if settings.lancedb_table in db.list_tables().tables:
        db.drop_table(settings.lancedb_table)

    # Convert Chunk objects to plain dicts for LanceDB
    rows = [
        {
            "chunk_id":  chunk.chunk_id,
            "text":      chunk.text,
            "embedding": chunk.embedding,
            "source":    chunk.source,
            "section":   chunk.section,
            "url":       chunk.url,
            "doc_id":    chunk.doc_id,
        }
        for chunk in chunks
    ]

    db.create_table(settings.lancedb_table, data=rows, schema=SCHEMA)
    return len(rows)


def search(query_vector: list[float], k: int = None) -> list[Chunk]:
    """
    Search the LanceDB table for the top-k chunks closest to the query vector.

    Uses cosine similarity (via L2 distance on normalized vectors — equivalent
    because all vectors are unit-normalized by the embedder).

    Args:
        query_vector: 384-dimensional query embedding from embed_query().
        k: Number of results to return. Defaults to settings.top_k.

    Returns:
        List of Chunk objects ordered by relevance (most similar first).
        Embeddings are not repopulated on returned chunks (not needed for display).

    Raises:
        RuntimeError: If the table doesn't exist (pipeline hasn't been run yet).
    """
    if k is None:
        k = settings.top_k

    db = _connect()

    if settings.lancedb_table not in db.list_tables().tables:
        raise RuntimeError(
            f"Table '{settings.lancedb_table}' not found in LanceDB. "
            "Run the ingestion pipeline first."
        )

    table = db.open_table(settings.lancedb_table)

    results = (
        table.search(query_vector)
            .limit(k)
            .to_list()
    )

    # Convert raw dicts back to Chunk objects
    return [
        Chunk(
            text=row["text"],
            source=row["source"],
            section=row["section"],
            url=row["url"],
            doc_id=row["doc_id"],
            chunk_index=int(row["chunk_id"].split("_")[-1]),
        )
        for row in results
    ]


def count() -> int:
    """
    Return the total number of chunks currently stored in the table.

    Useful for quickly verifying the pipeline ran successfully.

    Returns:
        Row count, or 0 if the table doesn't exist.
    """
    db = _connect()
    if settings.lancedb_table not in db.list_tables().tables:
        return 0
    return db.open_table(settings.lancedb_table).count_rows()
