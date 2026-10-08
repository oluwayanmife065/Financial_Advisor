"""
Abstract base class for all scrapers.
"""

from abc import ABC, abstractmethod
from typing import List
from ingestion.document import Document


class BaseScraper(ABC):
    """
    Abstract interface for document scrapers.
    """

    @abstractmethod
    def scrape_all(self) -> List[Document]:
        """
        Scrape content and return a list of standard Document instances.
        """
        pass
