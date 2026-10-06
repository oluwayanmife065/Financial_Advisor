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

import requests
from bs4 import BeautifulSoup
from loguru import logger

from ingestion.document import Document
from ingestion.scrapers.base import BaseScraper

# ---------------------------------------------------------------------------
# Target URL catalogue
# ---------------------------------------------------------------------------

INTL_STUDENT_SOURCES: list[dict] = [
    # --- DHS Study in the States (Immigration & Status Boundaries) ---
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
    # --- IRS (Taxation, Pub 519, Nonresident Alien Investing Rules) ---
    {
        "source": "IRS",
        "section": "Taxation of Nonresident Aliens (Pub 519 Overview)",
        "url": "https://www.irs.gov/individuals/international-taxpayers/taxation-of-nonresident-aliens",
    },
    {
        "source": "IRS",
        "section": "ITIN Application & Eligibility",
        "url": "https://www.irs.gov/individuals/individual-taxpayer-identification-number",
    },
    {
        "source": "IRS",
        "section": "Tax Treaties and Nonresident Withholding",
        "url": "https://www.irs.gov/individuals/international-taxpayers/tax-treaties",
    },
    {
        "source": "IRS",
        "section": "Foreign Students and Scholars Exemption",
        "url": "https://www.irs.gov/individuals/international-taxpayers/foreign-students-and-exchange-visitors",
    },
    {
        "source": "IRS",
        "section": "Investment Income of Nonresident Aliens & 30% Withholding",
        "url": "https://www.irs.gov/individuals/international-taxpayers/nontaxable-types-of-interest-income-for-nonresident-aliens",
    },
    {
        "source": "IRS",
        "section": "Capital Gains of Nonresident Aliens (183-Day Rule)",
        "url": "https://www.irs.gov/individuals/international-taxpayers/the-taxation-of-capital-gains-of-nonresident-alien-students-scholars-and-employees-of-foreign-governments",
    },
    # --- Social Security Administration ---
    {
        "source": "SSA",
        "section": "SSN for Non-Citizens & Student Employment",
        "url": "https://www.ssa.gov/ssnumber/ss5doc.htm",
    },
    # --- USCIS (Work Authorization vs Passive Investing) ---
    {
        "source": "USCIS",
        "section": "OPT for F-1 Students",
        "url": "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-opt-for-f-1-students",
    },
    {
        "source": "USCIS",
        "section": "STEM OPT Extension Regulations",
        "url": "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-extension-for-stem-students-stem-opt",
    },
    {
        "source": "USCIS",
        "section": "Changing Nonimmigrant Status",
        "url": "https://www.uscis.gov/visit-the-united-states/extend-your-stay/change-my-nonimmigrant-status",
    },
    # --- CFPB (Banking, High-Yield Savings, CD, Credit) ---
    {
        "source": "CFPB",
        "section": "Banking Basics & Savings Accounts",
        "url": "https://www.consumerfinance.gov/consumer-tools/money-as-you-grow/",
    },
    {
        "source": "CFPB",
        "section": "Building Credit Without US Credit History",
        "url": "https://www.consumerfinance.gov/ask-cfpb/how-do-i-get-a-credit-card-if-i-dont-have-a-credit-history-en-1175/",
    },
    # --- SEC / Investor.gov (Permitted Investments & Risk) ---
    {
        "source": "SEC/Investor.gov",
        "section": "Introduction to Investing: Stocks, Bonds, Mutual Funds",
        "url": "https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/how-0",
    },
    {
        "source": "SEC/Investor.gov",
        "section": "Compound Interest & Long-Term Growth",
        "url": "https://www.investor.gov/financial-tools-calculators/calculators/compound-interest-calculator",
    },
    {
        "source": "SEC/Investor.gov",
        "section": "Opening a Brokerage Account & Non-Citizen W-8BEN",
        "url": "https://www.investor.gov/introduction-investing/investing-basics/how-stock-markets-work",
    },
    # --- FINRA (Investor Protection, Margin & Day Trading Warnings) ---
    {
        "source": "FINRA",
        "section": "Day Trading Margin Requirements and Risk Alerts",
        "url": "https://www.finra.org/investors/learn-to-invest/advanced-investing/day-trading-margin-requirements",
    },
    {
        "source": "FINRA",
        "section": "Mutual Funds, Index Funds and ETFs",
        "url": "https://www.finra.org/investors/investing/investment-products",
    },
    # --- US TreasuryDirect (US Savings Bonds, Treasury Bills) ---
    {
        "source": "US TreasuryDirect",
        "section": "Treasury Bills, Notes and Non-Citizen Eligibility",
        "url": "https://www.treasurydirect.gov/indiv/research/indepth/tbills/res_tbill.htm",
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
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(f"⚠️  Could not fetch {url}: {exc}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        text = self._extract_text(soup)

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
    def _extract_text(soup: BeautifulSoup) -> str:
        """
        Strip navigation, scripts, and boilerplate; return readable body text.

        Args:
            soup: Parsed BeautifulSoup tree.

        Returns:
            Clean multi-line string of body text.
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
        # Collapse excessive blank lines
        text = "\n".join(line for line in lines if line)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ------------------------------------------------------------------
    # PDF export
    # ------------------------------------------------------------------

    def export_pdf(
        self,
        documents: list[Document],
        output_path: str | Path = "intl_student_sources.pdf",
    ) -> Path:
        """
        Compile all scraped documents into a single, bookmarked PDF.

        Requires: ``fpdf2`` (``pip install fpdf2``).

        Args:
            documents: List of Document objects to render.
            output_path: Destination path for the generated PDF.

        Returns:
            The resolved Path of the written PDF file.
        """
        try:
            from fpdf import FPDF  # type: ignore
        except ImportError:
            raise ImportError(
                "fpdf2 is required for PDF export. Install it with:\n"
                "  pip install fpdf2"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_margins(left=20, top=20, right=20)

        # --- Cover page ---
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=22)
        pdf.cell(0, 15, "International Student Finance Guide", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 8, "Auto-compiled from trusted US government sources", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(5)

        # Table of contents stub
        pdf.set_font("Helvetica", style="B", size=13)
        pdf.cell(0, 10, "Sources included:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=11)
        for doc in documents:
            pdf.cell(0, 7, f"  [{doc.source}]  {doc.section}", new_x="LMARGIN", new_y="NEXT")

        # --- One section per document ---
        for doc in documents:
            pdf.add_page()

            # Section header
            pdf.set_font("Helvetica", style="B", size=15)
            header = f"{doc.source}  —  {doc.section}"
            pdf.cell(0, 12, header[:90], new_x="LMARGIN", new_y="NEXT")

            pdf.set_font("Helvetica", style="I", size=9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 6, f"URL: {doc.url}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)

            # Body text — split into paragraphs
            pdf.set_font("Helvetica", size=10)
            for para in doc.text.split("\n\n"):
                para = para.strip()
                if not para:
                    continue
                # Encode to latin-1 safely
                safe = para.encode("latin-1", errors="replace").decode("latin-1")
                pdf.multi_cell(0, 6, safe)
                pdf.ln(3)

        pdf.output(str(output_path))
        logger.info(f"📄  PDF written → {output_path.resolve()}")
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
