"""
Ingestion pipeline — orchestrates the full flow from raw files to LanceDB.

This is the main entry point for ingestion. Running it:
  python3.11 -m ingestion.pipeline

Flow:
  1. Scan ingestion/sources/pdfs/ for PDFs, organised by source folder
  2. Parse each PDF into Documents (one per page)
  3. Chunk all Documents into Chunks
  4. Embed all Chunks (adds the 384-float vector to each Chunk)
  5. Write all Chunks to LanceDB (full overwrite)

The pipeline is idempotent in the sense that re-running it always produces
a clean, consistent DB reflecting exactly what's in the sources/ folder.
"""

import time
from pathlib import Path

from ingestion.parsers.pdf_parser import parse_pdf_folder, infer_source_label
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


def run(sources_root: Path = SOURCES_ROOT) -> dict:
    """
    Run the full ingestion pipeline for all PDF sources.

    Scans each subfolder in sources_root, parses PDFs, chunks, embeds,
    and writes to LanceDB. Prints progress at each stage.

    Args:
        sources_root: Root directory containing source subfolders (sec/, cfpb/, etc.)
                      Defaults to ingestion/sources/pdfs/.

    Returns:
        Summary dict with counts per stage:
        {
            "sources_processed": int,
            "documents_parsed": int,
            "chunks_created": int,
            "chunks_written": int,
            "elapsed_seconds": float,
        }
    """
    start = time.time()
    all_documents = []

    print("\n── Ingestion Pipeline ──────────────────────────────")

    # ── Stage 1: Parse ────────────────────────────────────────────────────────
    print("\n[1/4] Parsing PDF sources...")

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
    print(f"\n[2/4] Chunking (size={settings.chunk_size}, overlap={settings.chunk_overlap})...")
    all_chunks = chunk_documents(
        all_documents,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    print(f"  Total chunks created: {len(all_chunks)}")

    # ── Stage 3: Embed ────────────────────────────────────────────────────────
    print(f"\n[3/4] Embedding with {settings.embedding_model}...")
    embed_start = time.time()
    embed(all_chunks)
    embed_elapsed = round(time.time() - embed_start, 2)
    print(f"  Embedded {len(all_chunks)} chunks in {embed_elapsed}s")

    # ── Stage 4: Write to LanceDB ─────────────────────────────────────────────
    print(f"\n[4/4] Writing to LanceDB at {settings.lancedb_path}...")
    written = write(all_chunks)
    print(f"  Written: {written} chunks")
    print(f"  Verified: {count()} rows in table '{settings.lancedb_table}'")

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
        "chunks_written": written,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    run()

