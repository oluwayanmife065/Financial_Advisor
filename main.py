"""
Main CLI Interface — Interactive Personal Finance Literacy RAG Assistant.

Run with:
    python3.11 main.py

Features:
  - Interactive terminal chat loop
  - Grounded RAG answers from your ingested regulatory & educational corpus
  - Sources & citations display for transparency
  - Real-time latency tracking (retrieval ms, generation ms, total ms)
  - Automatic JSON Lines query logging for observability and failure analysis
"""

import sys
import time
from pathlib import Path

from config import settings
from ingestion.embedder import embed_query
from retrieval.lancedb_retriever import search, count
from generation.llm import generate_answer
from query_logging.query_logger import log_query


def print_banner():
    """Display startup banner and current system configuration."""
    print("=" * 65)
    print(" 📊 Personal Finance Literacy RAG Assistant (Local & Private)")
    print("=" * 65)
    print(f" LLM Model:        {settings.ollama_model} (Ollama local)")
    print(f" Embedder:         {settings.embedding_model}")
    print(f" Retrieval k:      {settings.top_k} chunks")
    print(f" Vector Store:     LanceDB ({count()} chunks indexed)")
    print(f" Query Logs:       {settings.log_file}")
    print("=" * 65)
    print(" Type your question below, or 'exit' / 'quit' to exit.")
    print(" Special commands: /sources (toggle source display), /k <num>\n")


def ask_rag(
    query: str,
    k: int = settings.top_k,
    show_sources: bool = True,
    model: str | None = None,
) -> str:
    """
    Execute end-to-end RAG pipeline for a single query and log the interaction.

    Args:
        query: The user's question.
        k: Number of chunks to retrieve.
        show_sources: Whether to print source citations to terminal.
        model: Optional LLM model override.

    Returns:
        Generated answer string.
    """
    # 1. Retrieval
    retrieval_start = time.perf_counter()
    query_vector = embed_query(query)
    chunks = search(query_vector, k=k)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    # 2. Generation
    generation_start = time.perf_counter()
    answer = generate_answer(query, chunks, model=model)
    generation_ms = (time.perf_counter() - generation_start) * 1000

    # 3. Observability / Logging
    log_query(
        query=query,
        answer=answer,
        chunks=chunks,
        retrieval_latency_ms=retrieval_ms,
        generation_latency_ms=generation_ms,
        model=model or settings.ollama_model,
        retriever_name="lancedb",
    )

    # 4. Display Results
    print(f"\nAssistant:\n{answer}\n")

    if show_sources and chunks:
        print("── Retrieved Context Sources ───────────────────────")
        for i, chunk in enumerate(chunks, 1):
            print(f" [{i}] {chunk.source} ({chunk.section})")
            preview = chunk.text.replace("\n", " ")[:120]
            print(f"     \"{preview}...\"")
        print("────────────────────────────────────────────────────")

    total_s = (retrieval_ms + generation_ms) / 1000
    print(f"⏱️  [Retrieval: {retrieval_ms:.0f}ms | Generation: {generation_ms/1000:.2f}s | Total: {total_s:.2f}s]\n")

    return answer


def main():
    """Main CLI loop."""
    print_banner()

    # Sanity check: Ensure database has chunks
    total_chunks = count()
    if total_chunks == 0:
        print("⚠️  Warning: LanceDB table appears empty.")
        print("   Please run 'python3.11 -m ingestion.pipeline' first to index your documents.\n")

    current_k = settings.top_k
    show_sources = True

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        # Handle commands
        if user_input.startswith("/k"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].isdigit():
                current_k = int(parts[1])
                print(f"ℹ️  Updated retrieval top_k to {current_k}\n")
            else:
                print("Usage: /k <integer_number>\n")
            continue

        if user_input == "/sources":
            show_sources = not show_sources
            status = "ON" if show_sources else "OFF"
            print(f"ℹ️  Source display is now {status}\n")
            continue

        # Execute query
        try:
            ask_rag(user_input, k=current_k, show_sources=show_sources)
        except Exception as err:
            print(f"❌ Error processing query: {err}\n")


if __name__ == "__main__":
    main()

