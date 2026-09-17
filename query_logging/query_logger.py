"""
Structured Query Logger — records each user query, latency, and retrieved chunk traces.

Logs are written as JSON Lines (.jsonl) to the path defined in config.py
(default: ./logs/query_log.jsonl).

Why JSON Lines?
  - Append-only and safe for stream/batch logging.
  - Easily queryable with jq, Pandas (pd.read_json(..., lines=True)), or observability tools.
  - Keeps full audit trail: query, answer, retrieval latency, generation latency,
    and exact chunk metadata traces.
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ingestion.document import Chunk
from config import settings


@dataclass
class QueryLogEntry:
    """
    Data model for a single RAG query interaction log.
    """
    timestamp: str
    query: str
    answer: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    model: str
    retriever: str
    k: int
    retrieved_chunks: list[dict[str, Any]]


def format_chunk_trace(chunk: Chunk, max_preview_chars: int = 150) -> dict[str, Any]:
    """
    Extract key metadata and a text snippet from a retrieved Chunk for logging.

    Args:
        chunk: The retrieved Chunk.
        max_preview_chars: Character limit for the text snippet.

    Returns:
        Dictionary with chunk metadata and snippet.
    """
    clean_text = chunk.text.replace("\n", " ").strip()
    preview = clean_text[:max_preview_chars] + ("..." if len(clean_text) > max_preview_chars else "")
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source": chunk.source,
        "section": chunk.section,
        "url": chunk.url,
        "text_preview": preview,
    }


def log_query(
    query: str,
    answer: str,
    chunks: list[Chunk],
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    model: str | None = None,
    retriever_name: str = "lancedb",
    log_file_path: str | Path | None = None,
) -> QueryLogEntry:
    """
    Record a completed query interaction and append it to the JSONL log file.

    Args:
        query: User's original prompt.
        answer: Generated LLM response.
        chunks: List of context chunks retrieved for this query.
        retrieval_latency_ms: Milliseconds spent in embedding & vector search.
        generation_latency_ms: Milliseconds spent in LLM completion.
        model: Model name used. Defaults to settings.ollama_model.
        retriever_name: Name of the vector store retriever (e.g. "lancedb", "pinecone").
        log_file_path: Optional override for the log file path.

    Returns:
        The constructed QueryLogEntry.
    """
    target_path = Path(log_file_path or settings.log_file)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    entry = QueryLogEntry(
        timestamp=datetime.now().isoformat(),
        query=query,
        answer=answer,
        retrieval_latency_ms=round(retrieval_latency_ms, 2),
        generation_latency_ms=round(generation_latency_ms, 2),
        total_latency_ms=round(retrieval_latency_ms + generation_latency_ms, 2),
        model=model or settings.ollama_model,
        retriever=retriever_name,
        k=len(chunks),
        retrieved_chunks=[format_chunk_trace(c) for c in chunks],
    )

    # Append JSON line to the target log file
    with open(target_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry)) + "\n")

    return entry

