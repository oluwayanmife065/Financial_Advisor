"""
Ingestion pipeline — orchestrates the full flow from raw files to a vector store.

This is the main entry point for ingestion. Running it:
  python3.11 -m ingestion.pipeline                 # default: LanceDB only
  python3.11 -m ingestion.pipeline --target pinecone
  python3.11 -m ingestion.pipeline --target both
  python3.11 -m ingestion.pipeline --from-lancedb  # skip parse/embed, upsert to Pinecone from existing LanceDB

Flow (default):
  1. Scan ingestion/sources/pdfs/ for PDFs, organised by source folder
  2. Parse each PDF into Documents (one per page)
  3. Chunk all Documents into Chunks
  4. Embed all Chunks (adds the 384-float vector to each Chunk)
  5. Write all Chunks to the selected vector store(s)

The pipeline is idempotent — re-running always produces a clean, consistent
store reflecting exactly what's in the sources/ folder.

Note on imports:
  write() and count() are imported at module level from lancedb_retriever so
  that existing unit-test mocks targeting `ingestion.pipeline.write` and
  `ingestion.pipeline.count` continue to work.
  The Pinecone retriever is imported lazily inside run() so that the SDK is
  not required when only LanceDB is being used.
"""

import argparse
import time
from pathlib import Path

from ingestion.parsers.pdf_parser import parse_pdf_folder
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed
from retrieval.lancedb_retriever import write, count
from config import settings


# Root folder containing all source subfolders
SOURCES_ROOT = Path(__file__).parent / "sources" / "pdfs"

# Map subfolder name → source label (must match SOURCE_LABELS in pdf_parser.py)
PDF_SOURCES = {
    "sec":   "SEC/Investor.gov",
    "cfpb":  "CFPB",
    "finra": "FINRA",
    "fed":   "Federal Reserve",
}


def _read_lancedb_chunks() -> list:
    """Read all embedded chunks out of LanceDB so they can be upserted elsewhere.

    Used by --from-lancedb to avoid re-parsing and re-embedding PDFs.

    Returns:
        List of Chunk objects reconstructed from LanceDB rows, with embeddings.

    Raises:
        RuntimeError: If the LanceDB table does not exist.
    """
    import lancedb as _lancedb
    from ingestion.document import Chunk

    db = _lancedb.connect(settings.lancedb_path)

    if settings.lancedb_table not in db.list_tables().tables:
        raise RuntimeError(
            f"LanceDB table '{settings.lancedb_table}' not found. "
            "Run the full pipeline (without --from-lancedb) first."
        )

    table = db.open_table(settings.lancedb_table)
    rows = table.to_pandas().to_dict("records")

    chunks = []
    for row in rows:
        chunk = Chunk(
            text=row["text"],
            source=row["source"],
            section=row["section"],
            url=row["url"],
            doc_id=row["doc_id"],
            chunk_index=int(row["chunk_id"].split("_")[-1]),
            # LanceDB returns embeddings as numpy float32 arrays; convert to
            # plain Python floats so Pinecone's JSON serialiser accepts them.
            embedding=[float(v) for v in row["embedding"]],
        )
        chunks.append(chunk)

    return chunks


def run(
    sources_root: Path = SOURCES_ROOT,
    target: str = "lancedb",
    from_lancedb: bool = False,
) -> dict:
    """Run the full ingestion pipeline for all PDF sources.

    Args:
        sources_root: Root directory containing source subfolders (sec/, cfpb/, etc.)
                      Defaults to ingestion/sources/pdfs/.
        target: Which vector store(s) to write to — 'lancedb', 'pinecone', or 'both'.
        from_lancedb: If True, skip parse/chunk/embed and read existing chunks
                      directly from LanceDB. Only valid when target includes 'pinecone'.

    Returns:
        Summary dict with counts per stage:
        {
            "sources_processed": int,
            "documents_parsed": int,
            "chunks_created": int,
            "chunks_written": int,   # combined total across all targets
            "elapsed_seconds": float,
        }
    """
    start = time.time()

    print("\n── Ingestion Pipeline ──────────────────────────────")
    print(f"   Target store(s): {target.upper()}")

    # ── Fast path: read from LanceDB, upsert to Pinecone ─────────────────────
    if from_lancedb:
        if target not in ("pinecone", "both"):
            raise ValueError(
                "--from-lancedb only makes sense with --target pinecone or --target both"
            )

        print("\n[FAST] Reading existing chunks from LanceDB...")
        all_chunks = _read_lancedb_chunks()
        print(f"  Loaded {len(all_chunks)} embedded chunks from LanceDB")

        from retrieval import pinecone_retriever
        print(f"\n[→] Upserting to Pinecone index '{settings.pinecone_index_name}'...")
        written = pinecone_retriever.write(all_chunks)
        print(f"  Written: {written} vectors")
        print(f"  Verified: {pinecone_retriever.count()} vectors in Pinecone")

        elapsed = round(time.time() - start, 2)
        print(f"\n── Done in {elapsed}s ──────────────────────────────\n")
        return {
            "sources_processed": 0,
            "documents_parsed": 0,
            "chunks_created": len(all_chunks),
            "chunks_written": written,
            "elapsed_seconds": elapsed,
        }

    # ── Stage 1: Parse ────────────────────────────────────────────────────────
    stage_count = 4 if target == "lancedb" else (5 if target == "both" else 4)
    print(f"\n[1/{stage_count}] Parsing PDF sources...")

    all_documents = []
    for folder_name, source_label in PDF_SOURCES.items():
        folder_path = sources_root / folder_name

        if not folder_path.exists():
            print(f"  [SKIP] {folder_name}/ not found")
            continue

        pdf_files = list(folder_path.glob("*.pdf"))
        if not pdf_files:
            print(f"  [SKIP] {folder_name}/ has no PDFs")
            continue

        docs = parse_pdf_folder(str(folder_path), source_label)
        all_documents.extend(docs)
        print(f"  [OK]   {source_label}: {len(pdf_files)} PDFs → {len(docs)} pages")

    if not all_documents:
        print("\n  [WARN] No documents found. Add PDFs to ingestion/sources/pdfs/")
        return {
            "sources_processed": 0,
            "documents_parsed": 0,
            "chunks_created": 0,
            "chunks_written": 0,
            "elapsed_seconds": round(time.time() - start, 2),
        }

    print(f"\n  Total documents parsed: {len(all_documents)}")

    # ── Stage 2: Chunk ────────────────────────────────────────────────────────
    print(f"\n[2/{stage_count}] Chunking (size={settings.chunk_size}, overlap={settings.chunk_overlap})...")
    all_chunks = chunk_documents(
        all_documents,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    print(f"  Total chunks created: {len(all_chunks)}")

    # ── Stage 3: Embed ────────────────────────────────────────────────────────
    print(f"\n[3/{stage_count}] Embedding with {settings.embedding_model}...")
    embed_start = time.time()
    embed(all_chunks)
    embed_elapsed = round(time.time() - embed_start, 2)
    print(f"  Embedded {len(all_chunks)} chunks in {embed_elapsed}s")

    total_written = 0
    next_stage = 4

    # ── Stage 4: Write to LanceDB ─────────────────────────────────────────────
    if target in ("lancedb", "both"):
        print(f"\n[{next_stage}/{stage_count}] Writing to LanceDB at {settings.lancedb_path}...")
        written_lance = write(all_chunks)
        total_written += written_lance
        print(f"  Written: {written_lance} chunks")
        print(f"  Verified: {count()} rows in table '{settings.lancedb_table}'")
        next_stage += 1

    # ── Stage 5: Write to Pinecone ────────────────────────────────────────────
    if target in ("pinecone", "both"):
        from retrieval import pinecone_retriever
        print(f"\n[{next_stage}/{stage_count}] Writing to Pinecone index '{settings.pinecone_index_name}'...")
        written_pine = pinecone_retriever.write(all_chunks)
        total_written += written_pine
        print(f"  Written: {written_pine} vectors")
        print(f"  Verified: {pinecone_retriever.count()} vectors in Pinecone")

    elapsed = round(time.time() - start, 2)
    sources_processed = sum(
        1 for f in PDF_SOURCES
        if (sources_root / f).exists() and list((sources_root / f).glob("*.pdf"))
    )

    print(f"\n── Done in {elapsed}s ──────────────────────────────\n")

    return {
        "sources_processed": sources_processed,
        "documents_parsed": len(all_documents),
        "chunks_created": len(all_chunks),
        "chunks_written": total_written,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Personal Finance Literacy RAG — Ingestion Pipeline"
    )
    parser.add_argument(
        "--target",
        choices=["lancedb", "pinecone", "both"],
        default="lancedb",
        help="Which vector store(s) to write to (default: lancedb)",
    )
    parser.add_argument(
        "--from-lancedb",
        action="store_true",
        help=(
            "Skip parse/chunk/embed — read existing embedded chunks from LanceDB "
            "and upsert them directly to Pinecone. "
            "Requires --target pinecone or --target both."
        ),
    )
    args = parser.parse_args()
    run(target=args.target, from_lancedb=args.from_lancedb)
