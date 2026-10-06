"""
International Student Finance Scraper
======================================
Scrapes trusted government / university sources that are specifically
relevant to international students in the USA:

  - DHS / Study in the States   (visa, work-authorization, CPT/OPT)
  - IRS                          (tax filing, ITIN, treaty benefits)
  - SSA                          (SSN eligibility for F-1/J-1)
  - CFPB                         (banking, credit building for newcomers)
  - USCIS                        (status changes, work permits)
  - Investor.gov / SEC           (investing basics for non-citizens)
  - FinAid.org-style pages       (scholarships, loans for international)

Each URL is scraped with `requests` + `BeautifulSoup`, then the clean
text is wrapped in the project's `Document` dataclass and optionally
compiled into a PDF via `fpdf2`.

Usage (standalone):
    python -m ingestion.scrapers.intl_student_scraper --pdf out/intl_student_docs.pdf

Usage (as module):
    from ingestion.scrapers.intl_student_scraper import IntlStudentScraper
    scraper = IntlStudentScraper()
    documents = scraper.scrape_all()
    pdf_path  = scraper.export_pdf(documents, "output.pdf")
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

try:
    import requests
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("IntlStudentScraper")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import urllib.request
import html
from html.parser import HTMLParser

from ingestion.document import Document
from ingestion.scrapers.base import BaseScraper


# ---------------------------------------------------------------------------
# Target URL catalogue
# ---------------------------------------------------------------------------

INTL_STUDENT_SOURCES: list[dict] = [
    # --- DHS Study in the States ---
    {
        "source": "DHS/StudyInTheStates",
        "section": "Working in the US (CPT/OPT)",
        "url": "https://studyinthestates.dhs.gov/students/work/working-in-the-united-states",
    },
    {
        "source": "DHS/StudyInTheStates",
        "section": "Maintaining Status",
        "url": "https://studyinthestates.dhs.gov/students/maintain-your-status",
    },
    {
        "source": "DHS/StudyInTheStates",
        "section": "OPT Overview",
        "url": "https://studyinthestates.dhs.gov/students/work/optional-practical-training-opt",
    },
    # --- IRS international students ---
    {
        "source": "IRS",
        "section": "Taxation of International Students",
        "url": "https://www.irs.gov/individuals/international-taxpayers/taxation-of-nonresident-aliens",
    },
    {
        "source": "IRS",
        "section": "ITIN Application",
        "url": "https://www.irs.gov/individuals/individual-taxpayer-identification-number",
    },
    {
        "source": "IRS",
        "section": "Tax Treaties",
        "url": "https://www.irs.gov/individuals/international-taxpayers/tax-treaties",
    },
    {
        "source": "IRS",
        "section": "Foreign Students and Scholars",
        "url": "https://www.irs.gov/individuals/international-taxpayers/foreign-students-and-exchange-visitors",
    },
    # --- Social Security Administration ---
    {
        "source": "SSA",
        "section": "SSN for Non-Citizens",
        "url": "https://www.ssa.gov/ssnumber/ss5doc.htm",
    },
    # --- USCIS ---
    {
        "source": "USCIS",
        "section": "OPT for F-1 Students",
        "url": "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-opt-for-f-1-students",
    },
    {
        "source": "USCIS",
        "section": "STEM OPT Extension",
        "url": "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-extension-for-stem-students-stem-opt",
    },
    {
        "source": "USCIS",
        "section": "Changing Nonimmigrant Status",
        "url": "https://www.uscis.gov/visit-the-united-states/extend-your-stay/change-my-nonimmigrant-status",
    },
    # --- CFPB newcomer banking ---
    {
        "source": "CFPB",
        "section": "Banking Basics",
        "url": "https://www.consumerfinance.gov/consumer-tools/money-as-you-grow/",
    },
    {
        "source": "CFPB",
        "section": "Building Credit",
        "url": "https://www.consumerfinance.gov/ask-cfpb/how-do-i-get-a-credit-card-if-i-dont-have-a-credit-history-en-1175/",
    },
    # --- SEC / Investor.gov ---
    {
        "source": "SEC/Investor.gov",
        "section": "Investing Basics for Everyone",
        "url": "https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/how-0",
    },
    {
        "source": "SEC/Investor.gov",
        "section": "Compound Interest Calculator",
        "url": "https://www.investor.gov/financial-tools-calculators/calculators/compound-interest-calculator",
    },
]


# ---------------------------------------------------------------------------
# Scraper implementation
# ---------------------------------------------------------------------------


class IntlStudentScraper(BaseScraper):
    """
    Scrapes finance and immigration-adjacent web pages that are directly
    useful to international students (F-1, J-1, OPT, CPT, taxes, banking).

    Attributes:
        sources: List of dicts with keys ``source``, ``section``, ``url``.
        request_delay: Seconds to sleep between requests (be polite).
        timeout: HTTP request timeout in seconds.
        headers: HTTP headers sent with every request.
    """

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; IntlStudentRAGBot/1.0; "
            "+https://github.com/your-repo/financial-advisor)"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(
        self,
        sources: Optional[list[dict]] = None,
        request_delay: float = 1.5,
        timeout: int = 15,
    ) -> None:
        self.sources = sources or INTL_STUDENT_SOURCES
        self.request_delay = request_delay
        self.timeout = timeout
        self.headers = self.DEFAULT_HEADERS

    # ------------------------------------------------------------------
    # BaseScraper interface
    # ------------------------------------------------------------------

    def scrape_all(self) -> list[Document]:
        """
        Iterate over every configured URL and return a list of Documents.

        Returns:
            List of Document objects, one per successfully scraped URL.
            Failed URLs are logged and skipped (never crash the pipeline).
        """
        documents: list[Document] = []
        for entry in self.sources:
            doc = self._scrape_one(entry["url"], entry["source"], entry["section"])
            if doc:
                documents.append(doc)
                logger.info(
                    f"✅  Scraped [{entry['source']}] {entry['section']} "
                    f"({len(doc.text)} chars)"
                )
            time.sleep(self.request_delay)
        logger.info(f"Scraping complete — {len(documents)}/{len(self.sources)} pages collected.")
        return documents

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scrape_one(self, url: str, source: str, section: str) -> Optional[Document]:
        """
        Fetch a single URL, extract clean body text, and wrap in a Document.

        Args:
            url: The page URL to fetch.
            source: Human-readable source label (e.g. ``"IRS"``).
            section: Sub-section label (e.g. ``"ITIN Application"``).

        Returns:
            A ``Document`` on success, or ``None`` on any error.
        """
        html_content = None
        if requests is not None:
            try:
                resp = requests.get(url, headers=self.headers, timeout=self.timeout)
                resp.raise_for_status()
                html_content = resp.text
            except Exception as exc:
                logger.warning(f"⚠️  Could not fetch {url} via requests: {exc}")
        
        if html_content is None:
            try:
                req = urllib.request.Request(url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    html_content = response.read().decode(charset, errors="ignore")
            except Exception as exc:
                logger.warning(f"⚠️  Could not fetch {url} via urllib: {exc}")
                return None

        if BeautifulSoup is not None:
            soup = BeautifulSoup(html_content, "html.parser")
            text = self._extract_text(soup)
        else:
            text = self._extract_text_stdlib(html_content)

        if len(text.strip()) < 100:
            logger.warning(f"⚠️  Almost no text extracted from {url} — skipping.")
            return None

        return Document(
            text=text,
            source=source,
            section=section,
            url=url,
        )

    @staticmethod
    def _extract_text(soup) -> str:
        """
        Strip navigation, scripts, and boilerplate; return readable body text.
        """
        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "noscript", "iframe"]):
            tag.decompose()

        # Prefer main content containers
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id=re.compile(r"(main|content|body)", re.I))
            or soup.find(class_=re.compile(r"(main|content|body)", re.I))
            or soup.body
        )
        if main is None:
            main = soup

        lines = [line.strip() for line in main.get_text(separator="\n").splitlines()]
        text = "\n".join(line for line in lines if line)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _extract_text_stdlib(raw_html: str) -> str:
        """Fallback HTML stripper using standard library regex & html unescape."""
        # Strip script and style tags
        cleaned = re.sub(r"<(script|style|nav|footer|header|aside)[^>]*>.*?</\1>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
        # Strip HTML tags
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = html.unescape(cleaned)
        lines = [line.strip() for line in cleaned.splitlines()]
        text = "\n".join(line for line in lines if line)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ------------------------------------------------------------------
    # PDF export
    # ------------------------------------------------------------------

    def export_pdf(
        self,
        documents: list[Document],
        output_path: str | Path = "output/intl_student_sources.pdf",
    ) -> Path:
        """
        Compile all scraped documents into a single PDF.
        Supports fpdf2 if available, or falls back to a robust built-in minimal PDF generator.

        Args:
            documents: List of Document objects to render.
            output_path: Destination path for the generated PDF.

        Returns:
            The resolved Path of the written PDF file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from fpdf import FPDF  # type: ignore

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_margins(left=20, top=20, right=20)

            # Cover page
            pdf.add_page()
            pdf.set_font("Helvetica", style="B", size=22)
            pdf.cell(0, 15, "International Student Finance Guide", new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.set_font("Helvetica", size=12)
            pdf.cell(0, 8, "Auto-compiled from trusted US government & regulatory sources", new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(5)

            # Table of contents
            pdf.set_font("Helvetica", style="B", size=13)
            pdf.cell(0, 10, "Sources included:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
            for doc in documents:
                pdf.cell(0, 7, f"  [{doc.source}]  {doc.section}", new_x="LMARGIN", new_y="NEXT")

            # Document pages
            for doc in documents:
                pdf.add_page()
                pdf.set_font("Helvetica", style="B", size=15)
                header = f"{doc.source}  -  {doc.section}"
                pdf.cell(0, 12, header[:90], new_x="LMARGIN", new_y="NEXT")

                pdf.set_font("Helvetica", style="I", size=9)
                pdf.set_text_color(100, 100, 100)
                pdf.cell(0, 6, f"URL: {doc.url}", new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(0, 0, 0)
                pdf.ln(3)

                pdf.set_font("Helvetica", size=10)
                for para in doc.text.split("\n\n"):
                    para = para.strip()
                    if not para:
                        continue
                    safe = para.encode("latin-1", errors="replace").decode("latin-1")
                    pdf.multi_cell(0, 6, safe)
                    pdf.ln(3)

            pdf.output(str(output_path))
            logger.info(f"PDF written via fpdf2 -> {output_path.resolve()}")
            return output_path.resolve()

        except ImportError:
            # Pure Python standard fallback PDF compiler
            logger.info("fpdf2 not detected. Using built-in minimal PDF writer...")
            return self._export_minimal_pdf(documents, output_path)

    def _export_minimal_pdf(self, documents: list[Document], output_path: Path) -> Path:
        """
        Pure-python standards-compliant minimal PDF writer without external C/pip dependencies.
        """
        def escape_pdf_str(s: str) -> str:
            clean = s.encode("latin-1", errors="replace").decode("latin-1")
            clean = clean.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            return clean

        pages_content = []
        # Page 1: Cover
        cover_stream = [
            "BT",
            "/F1 22 Tf",
            "50 720 Td",
            "(International Student Financial Literacy Guide) Tj",
            "/F1 12 Tf",
            "0 -30 Td",
            "(Curated Regulatory & Advisory Knowledge Base) Tj",
            "0 -25 Td",
            f"(Documents Compiled: {len(documents)}) Tj",
            "0 -40 Td",
            "/F1 14 Tf",
            "(Table of Contents:) Tj",
            "/F1 10 Tf",
        ]
        y_offset = -20
        for i, doc in enumerate(documents[:25], 1):
            title = f"{i}. [{doc.source}] {doc.section}"
            title = escape_pdf_str(title[:75])
            cover_stream.append(f"0 {y_offset} Td")
            cover_stream.append(f"({title}) Tj")
            y_offset = -16
        cover_stream.append("ET")
        pages_content.append("\n".join(cover_stream))

        # Doc pages
        for doc in documents:
            lines = doc.text.splitlines()
            wrapped_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    wrapped_lines.append("")
                    continue
                while len(line) > 85:
                    split_idx = line.rfind(" ", 0, 85)
                    if split_idx == -1:
                        split_idx = 85
                    wrapped_lines.append(line[:split_idx])
                    line = line[split_idx:].strip()
                if line:
                    wrapped_lines.append(line)

            # Split into chunks of 45 lines per PDF page
            page_size = 45
            for page_idx in range(0, max(1, len(wrapped_lines)), page_size):
                sub_lines = wrapped_lines[page_idx:page_idx + page_size]
                stream = [
                    "BT",
                    "/F1 14 Tf",
                    "40 750 Td",
                    f"({escape_pdf_str(doc.source)} - {escape_pdf_str(doc.section[:45])}) Tj",
                    "/F1 8 Tf",
                    "0 -16 Td",
                    f"(URL: {escape_pdf_str(doc.url[:85])}) Tj",
                    "/F1 9 Tf",
                    "0 -20 Td",
                ]
                for l in sub_lines:
                    safe_l = escape_pdf_str(l)
                    stream.append(f"({safe_l}) Tj")
                    stream.append("0 -13 Td")
                stream.append("ET")
                pages_content.append("\n".join(stream))

        # Build PDF object hierarchy
        objects = []
        objects.append("<< /Type /Catalog /Pages 2 0 R >>")  # Obj 1: Catalog
        
        # Obj 2: Pages list (will fill kids later)
        page_obj_ids = [3 + i * 2 for i in range(len(pages_content))]
        kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
        objects.append(f"<< /Type /Pages /Kids [{kids_str}] /Count {len(pages_content)} >>")

        for idx, p_stream in enumerate(pages_content):
            p_obj_id = 3 + idx * 2
            c_obj_id = p_obj_id + 1
            # Page object
            objects.append(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {c_obj_id} 0 R "
                f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>"
            )
            # Content Stream object
            stream_bytes = p_stream.encode("latin-1", errors="replace")
            objects.append(
                f"<< /Length {len(stream_bytes)} >>\nstream\n{p_stream}\nendstream"
            )

        # Assemble PDF file
        pdf_bytes = bytearray(b"%PDF-1.4\n")
        xref_offsets = [0]
        for i, obj in enumerate(objects, 1):
            xref_offsets.append(len(pdf_bytes))
            pdf_bytes.extend(f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1"))

        xref_start = len(pdf_bytes)
        pdf_bytes.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1"))
        for off in xref_offsets[1:]:
            pdf_bytes.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

        pdf_bytes.extend(
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode("latin-1")
        )

        output_path.write_bytes(pdf_bytes)
        logger.info(f"Minimal PDF compiled successfully -> {output_path.resolve()}")
        return output_path.resolve()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape international-student finance sources and export to PDF."
    )
    parser.add_argument(
        "--pdf",
        metavar="PATH",
        default="output/intl_student_sources.pdf",
        help="Output PDF path (default: output/intl_student_sources.pdf)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds between HTTP requests (default: 1.5)",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Scrape only — skip PDF generation (useful for pipeline use)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    scraper = IntlStudentScraper(request_delay=args.delay)
    docs = scraper.scrape_all()

    if not args.no_pdf:
        pdf_path = scraper.export_pdf(docs, output_path=args.pdf)
        print(f"\n✅  PDF saved to: {pdf_path}")
    else:
        print(f"\n✅  Scraped {len(docs)} documents (no PDF requested).")
