"""
Tests for the chunker (ingestion/chunker.py).

All tests are pure unit tests — no files, no model loading, no network.
The chunker only deals with strings and dataclasses, so everything is fast
and deterministic.
"""

import pytest
from ingestion.chunker import chunk_document, chunk_documents
from ingestion.document import Document, Chunk


# ── Shared test fixture ───────────────────────────────────────────────────────

def make_document(text: str) -> Document:
    """
    Helper to create a minimal Document for testing.
    Avoids repeating boilerplate metadata in every test.
    """
    return Document(
        text=text,
        source="Test Source",
        section="page_1",
        url="file:///test/doc.pdf#page_1",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TestChunkDocument
# Tests chunk_document() — splits a single Document into Chunks.
# ══════════════════════════════════════════════════════════════════════════════

class TestChunkDocument:

    def test_short_document_produces_single_chunk(self):
        """
        A document shorter than chunk_size should produce exactly one chunk
        containing the full text. No splitting should occur.
        """
        doc = make_document("This is a short document.")
        chunks = chunk_document(doc, chunk_size=2000, overlap=200)
        assert len(chunks) == 1
        assert chunks[0].text == "This is a short document."

    def test_long_document_produces_multiple_chunks(self):
        """
        A document longer than chunk_size should be split into multiple chunks.
        This verifies the sliding window loop runs more than once.
        """
        long_text = "A" * 5000
        doc = make_document(long_text)
        chunks = chunk_document(doc, chunk_size=2000, overlap=200)
        assert len(chunks) > 1

    def test_chunk_text_does_not_exceed_chunk_size(self):
        """
        No individual chunk should exceed chunk_size characters.
        If any chunk is too large, the downstream embedder may exceed the
        model's token limit, causing truncated or degraded embeddings.
        """
        long_text = "Word " * 1000  # 5000 chars
        doc = make_document(long_text)
        chunks = chunk_document(doc, chunk_size=2000, overlap=200)
        for chunk in chunks:
            assert len(chunk.text) <= 2000, (
                f"Chunk {chunk.chunk_index} is {len(chunk.text)} chars, exceeds limit of 2000"
            )

    def test_overlap_is_present_between_consecutive_chunks(self):
        """
        The end of chunk N and the start of chunk N+1 should share `overlap`
        characters. This is the core guarantee of the sliding window approach.
        Without overlap, ideas split across chunk boundaries are lost.

        Checks that the last `overlap` chars of chunk 0 appear at the start
        of chunk 1.
        """
        text = "ABCDEFGHIJ" * 100   # 1000 chars, predictable content
        doc = make_document(text)
        chunks = chunk_document(doc, chunk_size=200, overlap=50)

        assert len(chunks) >= 2, "Need at least 2 chunks to test overlap"

        # The tail of chunk 0 should appear at the start of chunk 1
        tail_of_first = chunks[0].text[-50:]
        head_of_second = chunks[1].text[:50]
        assert tail_of_first == head_of_second, (
            "Overlap content not found between chunk 0 and chunk 1"
        )

    def test_chunks_inherit_parent_metadata(self):
        """
        Every chunk must carry the same source, section, url, and doc_id as
        the parent Document. This metadata is what the eval harness and the
        retrieval layer use to trace which source a chunk came from.
        """
        doc = make_document("Some text content.")
        chunks = chunk_document(doc, chunk_size=2000, overlap=200)
        for chunk in chunks:
            assert chunk.source  == doc.source
            assert chunk.section == doc.section
            assert chunk.url     == doc.url
            assert chunk.doc_id  == doc.doc_id

    def test_chunk_index_is_sequential_from_zero(self):
        """
        chunk_index must start at 0 and increment by 1 for each chunk.
        This index is used to construct chunk_id (e.g. "abc123_0002") and
        to reconstruct document order during debugging.
        """
        long_text = "X " * 2000
        doc = make_document(long_text)
        chunks = chunk_document(doc, chunk_size=500, overlap=50)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i, (
                f"Expected chunk_index {i} but got {chunk.chunk_index}"
            )

    def test_chunk_ids_are_unique(self):
        """
        Each chunk must have a unique chunk_id. Duplicate IDs would cause
        silent overwrites in the vector store.
        chunk_id is derived from doc_id + chunk_index, so uniqueness depends
        on chunk_index being sequential and non-repeating.
        """
        long_text = "Y " * 2000
        doc = make_document(long_text)
        chunks = chunk_document(doc, chunk_size=500, overlap=50)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids detected"

    def test_empty_document_returns_no_chunks(self):
        """
        A Document with only whitespace text should produce zero chunks.
        The chunker strips each chunk — if nothing remains, it's skipped.
        This prevents empty strings from being sent to the embedder.
        """
        doc = make_document("   \n\n   ")
        chunks = chunk_document(doc, chunk_size=2000, overlap=200)
        assert len(chunks) == 0

    def test_returns_chunk_objects(self):
        """
        Every item in the returned list must be a Chunk instance.
        Ensures the chunker isn't accidentally returning Documents or dicts.
        """
        doc = make_document("Some valid text here.")
        chunks = chunk_document(doc, chunk_size=2000, overlap=200)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_embedding_is_none_before_embedder_runs(self):
        """
        Chunks produced by the chunker should have embedding=None.
        The embedder is a separate step — the chunker must not attempt
        to compute embeddings. This enforces separation of concerns.
        """
        doc = make_document("Some text.")
        chunks = chunk_document(doc, chunk_size=2000, overlap=200)
        for chunk in chunks:
            assert chunk.embedding is None, (
                "Chunker should not set embeddings — that is the embedder's job"
            )


# ══════════════════════════════════════════════════════════════════════════════
# TestChunkDocuments
# Tests the batch convenience wrapper chunk_documents().
# ══════════════════════════════════════════════════════════════════════════════

class TestChunkDocuments:

    def test_processes_multiple_documents(self):
        """
        chunk_documents() should process every Document in the list and
        return a flat combined list of all chunks from all documents.
        """
        docs = [make_document("Short text one."), make_document("Short text two.")]
        chunks = chunk_documents(docs, chunk_size=2000, overlap=200)
        assert len(chunks) == 2   # one chunk per short document

    def test_empty_list_returns_empty(self):
        """
        Passing an empty list should return an empty list without errors.
        The pipeline may call this with an empty source folder.
        """
        chunks = chunk_documents([], chunk_size=2000, overlap=200)
        assert chunks == []

    def test_chunks_from_different_documents_have_different_doc_ids(self):
        """
        Chunks from separate Documents must carry different doc_ids.
        Mixing doc_ids would make it impossible to trace a chunk back to
        its source document in the eval harness or during debugging.
        """
        doc1 = Document(
            text="Content from document one.",
            source="Source A", section="page_1",
            url="file:///doc1.pdf#page_1"
        )
        doc2 = Document(
            text="Content from document two.",
            source="Source B", section="page_1",
            url="file:///doc2.pdf#page_1"
        )
        chunks = chunk_documents([doc1, doc2])
        doc_ids = [c.doc_id for c in chunks]
        assert doc_ids[0] != doc_ids[1], (
            "Chunks from different documents should have different doc_ids"
        )

