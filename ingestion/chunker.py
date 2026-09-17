"""
Chunker — splits Document objects into smaller Chunk objects for embedding.

Uses a sliding window approach: each chunk is up to `chunk_size` characters,
with `overlap` characters shared with the next chunk. Overlap ensures that
ideas spanning a boundary aren't cut in half and lost from retrieval.

Why character-based instead of token-based?
Token counts vary by model. Character counts are model-agnostic and fast.
At ~4 chars/token, chunk_size=2000 chars ≈ 500 tokens — well within the
384-512 token sweet spot for bge-small-en-v1.5.
"""

from ingestion.document import Document, Chunk


def chunk_document(
    document: Document,
    chunk_size: int = 2000,
    overlap: int = 200,
) -> list[Chunk]:
    """
    Split a single Document into a list of overlapping Chunks.

    Each chunk inherits all metadata from the parent Document (source,
    section, url, doc_id) and gets its own chunk_index to distinguish it.

    Args:
        document: The parsed Document to split.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters to repeat at the start of the next chunk.
                 Prevents ideas from being severed at chunk boundaries.

    Returns:
        List of Chunk objects. Returns a single chunk if the document text
        is shorter than chunk_size.
    """
    text = document.text
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        # Skip chunks that are only whitespace after stripping
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                source=document.source,
                section=document.section,
                url=document.url,
                doc_id=document.doc_id,
                chunk_index=index,
            ))
            index += 1

        # Slide forward by (chunk_size - overlap) so the next chunk
        # starts overlap characters before where this one ended
        start += chunk_size - overlap

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 2000,
    overlap: int = 200,
) -> list[Chunk]:
    """
    Split a list of Documents into Chunks, processing each document in order.

    Convenience wrapper around chunk_document() for pipeline use.

    Args:
        documents: List of Documents from the parser.
        chunk_size: Maximum characters per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        Flat list of all Chunks from all Documents combined.
    """
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size, overlap))
    return all_chunks

