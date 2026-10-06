"""
Unit tests for ingestion/scrapers/intl_student_scraper.py
"""

from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import unittest

from ingestion.document import Document
from ingestion.scrapers.intl_student_scraper import IntlStudentScraper



def test_scraper_text_extraction():
    sample_html = """
    <html>
      <head><title>Test</title></head>
      <body>
        <nav>Skip me</nav>
        <main>
          <h1>CPT Guidelines</h1>
          <p>International students on F-1 visas may participate in Curricular Practical Training.</p>
        </main>
        <footer>Footer noise</footer>
      </body>
    </html>
    """
    text = IntlStudentScraper._extract_text_stdlib(sample_html)
    assert "CPT Guidelines" in text
    assert "Curricular Practical Training" in text
    assert "Skip me" not in text
    assert "Footer noise" not in text


def test_scraper_mocked_fetch():
    scraper = IntlStudentScraper(
        sources=[{"source": "DHS", "section": "CPT", "url": "https://example.com/cpt"}],
        request_delay=0.0
    )
    with patch.object(scraper, "_scrape_one") as mock_scrape:
        mock_scrape.return_value = Document(
            text="F-1 students can work up to 20 hours per week during school session.",
            source="DHS",
            section="CPT",
            url="https://example.com/cpt"
        )
        docs = scraper.scrape_all()
        assert len(docs) == 1
        assert "20 hours" in docs[0].text
        assert docs[0].source == "DHS"


def test_pdf_export_fallback():
    docs = [
        Document(
            text="International students are typically nonresident aliens for tax purposes for their first 5 calendar years.",
            source="IRS",
            section="Nonresident Alien Taxes",
            url="https://www.irs.gov/test"
        )
    ]
    scraper = IntlStudentScraper()
    with tempfile.TemporaryDirectory() as tmpdir:
        target_pdf = Path(tmpdir) / "test.pdf"
        out_path = scraper.export_pdf(docs, output_path=target_pdf)
        assert out_path.exists()
        assert out_path.stat().st_size > 0
