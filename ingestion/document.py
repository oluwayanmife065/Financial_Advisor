"""
Core data models for the ingestion pipeline.

All data flowing through the system is typed using these dataclasses.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import hashlib


@dataclass
class Document:
    """
    Represents a single parsed source document (e.g. one PDF page or scraped article).

    Attributes:
        text: The clean extracted text content.
        source: Human-readable source name (e.g. "SEC/Investor.gov", "CFPB").
        section: Sub-section within the source (e.g. "page_3", "article title").
        url: Original URL or absolute file path of the source.
        date_ingested: ISO 8601 timestamp of when this document was ingested.
        doc_id: Deterministic hash ID — same url + date = same ID, prevents duplicates.
    """

    text: str
    source: str
    section: str
    url: str
    date_ingested: str = field(default_factory=lambda: datetime.now().isoformat())
    doc_id: str = field(init=False)

    def __post_init__(self):
        raw = f"{self.url}:{self.date_ingested}"
        self.doc_id = hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class Chunk:
    """
    Represents a single text chunk, ready for embedding and storage.

    A Document is split into many Chunks by the Chunker.

    Attributes:
        text: The chunk text content.
        source: Inherited from the parent Document.
        section: Inherited from the parent Document.
        url: Inherited from the parent Document.
        doc_id: ID of the parent Document.
        chunk_index: Position of this chunk within the parent document (0-indexed).
        embedding: Vector representation — None until set by the Embedder.
    """

    text: str
    source: str
    section: str
    url: str
    doc_id: str
    chunk_index: int
    embedding: Optional[list[float]] = None

    @property
    def chunk_id(self) -> str:
        """Unique ID for this specific chunk: parent doc_id + chunk position."""
        return f"{self.doc_id}_{self.chunk_index:04d}"

