"""
Abstract BaseRetriever interface.

Both the LanceDB and Pinecone retrievers expose the same three operations so
the eval runner, CLI, and Streamlit app can swap backends without changes.

Implementors should import this class and raise NotImplementedError for any
method they intentionally do not support.
"""

from abc import ABC, abstractmethod

from ingestion.document import Chunk


class BaseRetriever(ABC):
    """Abstract base class for all vector-store retriever backends."""

    @abstractmethod
    def write(self, chunks: list[Chunk]) -> int:
        """Persist a list of embedded chunks to the backing store.

        Args:
            chunks: Chunk objects with embedding already set.

        Returns:
            Number of chunks successfully written.

        Raises:
            ValueError: If any chunk is missing its embedding.
        """

    @abstractmethod
    def search(self, query_vector: list[float], k: int) -> list[Chunk]:
        """Return the top-k chunks most semantically similar to query_vector.

        Args:
            query_vector: Dense embedding of the user query (384-dim).
            k: Maximum number of results to return.

        Returns:
            List of Chunk objects ordered by relevance (most similar first).
        """

    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks currently stored.

        Returns:
            Integer count, or 0 if the store is empty / not yet initialised.
        """

