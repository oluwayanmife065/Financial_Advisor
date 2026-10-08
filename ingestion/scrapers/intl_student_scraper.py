"""
International Student Finance Scraper
======================================
Scrapes trusted government, regulatory, and university sources specifically
relevant to international students in the USA:

  - DHS / Study in the States   (visa status, work-authorization, CPT/OPT)
  - IRS                          (tax filing, ITIN, 30% dividend withholding,
                                 capital gains 183-day rule, portfolio interest)
  - SSA                          (SSN eligibility for F-1/J-1)
  - CFPB                         (banking, HYSA, credit building without SSN)
  - USCIS                        (OPT, STEM extensions, passive vs active work)
  - SEC / Investor.gov           (investing basics, W-8BEN, compound growth)
  - FINRA                        (day trading warnings, margin rules, ETFs)
  - US TreasuryDirect            (Treasury bills, notes, interest exemptions)

Features:
  - Zero required 3rd-party dependencies: uses standard library urllib/re/html
    with graceful upgrades if requests/BeautifulSoup are installed.
  - Built-in PDF compiler: outputs a formatted PDF with cover page and table
    of contents using standard library binary writer (or fpdf2 if available).

Usage (standalone CLI):
    python3 -m ingestion.scrapers.intl_student_scraper --pdf output/intl_student_financial_guide.pdf

Usage (as module):
    from ingestion.scrapers.intl_student_scraper import IntlStudentScraper
    scraper = IntlStudentScraper()
    documents = scraper.scrape_all()
    pdf_path  = scraper.export_pdf(documents, "output/intl_student_financial_guide.pdf")
"""

from __future__ import annotations

import argparse
import html
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
import urllib.request

# Optional 3rd party imports with graceful standard library fallback
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

from ingestion.document import Document
from ingestion.scrapers.base import BaseScraper


# ---------------------------------------------------------------------------
# Target URL catalogue — Immigration, Taxes, & Financial Investments
# ---------------------------------------------------------------------------

INTL_STUDENT_SOURCES: list[dict] = [
    # ── DHS Study in the States (Immigration, Status & Work Limits) ──
    {
        "source": "DHS/StudyInTheStates",
        "section": "Working in the United States (CPT/OPT Overview)",
        "url": "https://studyinthestates.dhs.gov/students/work/working-in-the-united-states",
    },
    {
        "source": "DHS/StudyInTheStates",
        "section": "Maintaining F-1 Nonimmigrant Status",
        "url": "https://studyinthestates.dhs.gov/students/maintain-your-status",
    },
    {
        "source": "DHS/StudyInTheStates",
        "section": "Training Opportunities in the United States",
        "url": "https://studyinthestates.dhs.gov/students/training-opportunities-in-the-united-states",
    },
    {
        "source": "DHS/StudyInTheStates",
        "section": "Curricular Practical Training (CPT) Regulatory Rules",
        "url": "https://studyinthestates.dhs.gov/sevis-help-hub/student-records/fm-student-employment/f-1-curricular-practical-training-cpt",
    },
    {
        "source": "DHS/StudyInTheStates",
        "section": "STEM OPT Practical Training Extension Rules",
        "url": "https://studyinthestates.dhs.gov/sevis-help-hub/student-records/fm-student-employment/f-1-stem-optional-practical-training-opt-extension",
    },

    # ── IRS (Publication 519, Nonresident Taxes, & Investment Taxation) ──
    {
        "source": "IRS",
        "section": "Publication 519: U.S. Tax Guide for Aliens (Master Guide)",
        "url": "https://www.irs.gov/publications/p519",
    },
    {
        "source": "IRS",
        "section": "Taxation of Nonresident Aliens (Core Rules)",
        "url": "https://www.irs.gov/individuals/international-taxpayers/taxation-of-nonresident-aliens",
    },
    {
        "source": "IRS",
        "section": "Foreign Students, Scholars, Teachers and Exchange Visitors",
        "url": "https://www.irs.gov/individuals/international-taxpayers/foreign-students-scholars-teachers-researchers-and-exchange-visitors",
    },
    {
        "source": "IRS",
        "section": "Substantial Presence Test (5-Year Exemption for F-1)",
        "url": "https://www.irs.gov/individuals/international-taxpayers/substantial-presence-test",
    },
    {
        "source": "IRS",
        "section": "Form 8843 Statement for Exempt Individuals",
        "url": "https://www.irs.gov/forms-pubs/about-form-8843",
    },
    {
        "source": "IRS",
        "section": "Form 1040-NR: U.S. Nonresident Alien Income Tax Return",
        "url": "https://www.irs.gov/forms-pubs/about-form-1040-nr",
    },
    {
        "source": "IRS",
        "section": "Individual Taxpayer Identification Number (ITIN) Guidance",
        "url": "https://www.irs.gov/individuals/individual-taxpayer-identification-number",
    },
    {
        "source": "IRS",
        "section": "United States Income Tax Treaties Overview",
        "url": "https://www.irs.gov/individuals/international-taxpayers/tax-treaties",
    },
    {
        "source": "IRS",
        "section": "Claiming Tax Treaty Benefits for Foreign Students",
        "url": "https://www.irs.gov/individuals/international-taxpayers/claiming-tax-treaty-benefits",
    },
    {
        "source": "IRS",
        "section": "Foreign Student Liability for Social Security & Medicare (FICA)",
        "url": "https://www.irs.gov/individuals/international-taxpayers/foreign-student-liability-for-social-security-and-medicare-taxes",
    },
    {
        "source": "IRS",
        "section": "Nontaxable Types of Interest Income for Nonresident Aliens",
        "url": "https://www.irs.gov/individuals/international-taxpayers/nontaxable-types-of-interest-income-for-nonresident-aliens",
    },
    {
        "source": "IRS",
        "section": "Taxation of Capital Gains for Nonresident Alien Students (183-Day Rule)",
        "url": "https://www.irs.gov/individuals/international-taxpayers/the-taxation-of-capital-gains-of-nonresident-alien-students-scholars-and-employees-of-foreign-governments",
    },

    # ── USCIS (Work Permissions, OPT, & Visa Maintenance) ──
    {
        "source": "USCIS",
        "section": "Optional Practical Training (OPT) for F-1 Students",
        "url": "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-opt-for-f-1-students",
    },
    {
        "source": "USCIS",
        "section": "STEM OPT 24-Month Extension Requirements",
        "url": "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-extension-for-stem-students-stem-opt",
    },
    {
        "source": "USCIS",
        "section": "Form I-765: Application for Employment Authorization (EAD)",
        "url": "https://www.uscis.gov/i-765",
    },
    {
        "source": "USCIS",
        "section": "Changing Nonimmigrant Status Within the US",
        "url": "https://www.uscis.gov/visit-the-united-states/change-my-nonimmigrant-status",
    },
    {
        "source": "USCIS",
        "section": "Students and Exchange Visitors Comprehensive Overview",
        "url": "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors",
    },

    # ── SEC / Investor.gov (Permitted Investments, Stocks, ETFs, Compound Growth) ──
    {
        "source": "SEC/Investor.gov",
        "section": "Stocks: Investment Products, Risks & Returns",
        "url": "https://www.investor.gov/introduction-investing/investing-basics/investment-products/stocks",
    },
    {
        "source": "SEC/Investor.gov",
        "section": "How Stock Markets Work and Execution Basics",
        "url": "https://www.investor.gov/introduction-investing/investing-basics/how-stock-markets-work",
    },
    {
        "source": "SEC/Investor.gov",
        "section": "Mutual Funds and Exchange-Traded Funds (ETFs)",
        "url": "https://www.investor.gov/introduction-investing/investing-basics/investment-products/mutual-funds-and-exchange-traded-funds",
    },
    {
        "source": "SEC/Investor.gov",
        "section": "Bonds and Fixed Income Investment Products",
        "url": "https://www.investor.gov/introduction-investing/investing-basics/investment-products/bonds-or-fixed-income-products",
    },
    {
        "source": "SEC/Investor.gov",
        "section": "Compound Interest and Long-Term Growth Mechanics",
        "url": "https://www.investor.gov/financial-tools-calculators/calculators/compound-interest-calculator",
    },
    {
        "source": "SEC/Investor.gov",
        "section": "The Role of the Securities and Exchange Commission (SEC)",
        "url": "https://www.investor.gov/introduction-investing/investing-basics/role-sec",
    },
    {
        "source": "SEC/Investor.gov",
        "section": "Financial and Investment Terms Comprehensive Glossary",
        "url": "https://www.investor.gov/introduction-investing/investing-basics/glossary",
    },

    # ── US TreasuryDirect (Safe Sovereign Debt, T-Bills, Notes) ──
    {
        "source": "US TreasuryDirect",
        "section": "Treasury Bills (T-Bills): Terms, Maturities, and Exemptions",
        "url": "https://www.treasurydirect.gov/marketable-securities/treasury-bills/",
    },
    {
        "source": "US TreasuryDirect",
        "section": "Treasury Notes: Fixed Principal Investment Mechanics",
        "url": "https://www.treasurydirect.gov/marketable-securities/treasury-notes/",
    },

    # ── FDIC & CFPB (Safe Banking, Deposit Insurance, Credit Building) ──
    {
        "source": "FDIC",
        "section": "Money Smart: Consumer Financial Education",
        "url": "https://www.fdic.gov/resources/consumers/money-smart/index.html",
    },
    {
        "source": "FDIC",
        "section": "Understanding FDIC Deposit Insurance for Bank Accounts",
        "url": "https://www.fdic.gov/resources/deposit-insurance/understanding-deposit-insurance/index.html",
    },
    {
        "source": "CFPB",
        "section": "Money As You Grow: Building Financial Capability",
        "url": "https://www.consumerfinance.gov/consumer-tools/money-as-you-grow/",
    },
    {
        "source": "CFPB",
        "section": "Credit Cards: Key Features, Rates, and How to Apply",
        "url": "https://www.consumerfinance.gov/ask-cfpb/how-do-i-get-a-credit-card-en-42/",
    },
    {
        "source": "CFPB",
        "section": "Building and Maintaining a Strong Credit Score",
        "url": "https://www.consumerfinance.gov/ask-cfpb/how-do-i-get-and-keep-a-good-credit-score-en-318/",
    },
    {
        "source": "CFPB",
        "section": "Consumer Protection and Financial Fraud Prevention",
        "url": "https://www.consumerfinance.gov/consumer-tools/fraud/",
    },
]


# ---------------------------------------------------------------------------
# Scraper implementation
# ---------------------------------------------------------------------------

class IntlStudentScraper(BaseScraper):
    """
    Scrapes finance, investing, tax, and immigration web pages relevant to
    international students (F-1, J-1, OPT, CPT, taxes, banking, investing).

    Attributes:
        sources: List of dicts with keys ``source``, ``section``, ``url``.
        request_delay: Seconds to sleep between requests.
        timeout: HTTP request timeout in seconds.
        headers: HTTP headers sent with every request.
    """

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(
        self,
        sources: Optional[list[dict]] = None,
        request_delay: float = 0.8,
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
        """
        documents: list[Document] = []
        for entry in self.sources:
            doc = self._scrape_one(entry["url"], entry["source"], entry["section"])
            if doc:
                documents.append(doc)
                logger.info(
                    f"✅ Scraped [{entry['source']}] {entry['section']} "
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
        Tries `requests` if available, otherwise falls back to `urllib.request`.
        """
        html_content = None

        if requests is not None:
            try:
                resp = requests.get(url, headers=self.headers, timeout=self.timeout)
                resp.raise_for_status()
                html_content = resp.text
            except Exception as exc:
                logger.warning(f"⚠️ Could not fetch {url} via requests: {exc}")

        if html_content is None:
            import ssl
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

            try:
                req = urllib.request.Request(url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=self.timeout, context=ssl_ctx) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    html_content = response.read().decode(charset, errors="ignore")
            except Exception as exc:
                logger.warning(f"⚠️ Could not fetch {url} via urllib: {exc}")
                return None

        if BeautifulSoup is not None:
            soup = BeautifulSoup(html_content, "html.parser")
            text = self._extract_text(soup)
        else:
            text = self._extract_text_stdlib(html_content)

        if len(text.strip()) < 100:
            logger.warning(f"⚠️ Almost no text extracted from {url} — skipping.")
            return None

        return Document(
            text=text,
            source=source,
            section=section,
            url=url,
        )

    @staticmethod
    def _extract_text(soup) -> str:
        """Strip navigation, scripts, and boilerplate using BeautifulSoup."""
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "noscript", "iframe"]):
            tag.decompose()

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
        cleaned = re.sub(
            r"<(script|style|nav|footer|header|aside)[^>]*>.*?</\1>",
            " ",
            raw_html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = html.unescape(cleaned)
        lines = [line.strip() for line in cleaned.splitlines()]
        text = "\n".join(line for line in lines if line)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ------------------------------------------------------------------
    # PDF export (supports fpdf2 or pure-Python fallback)
    # ------------------------------------------------------------------

    def export_pdf(
        self,
        documents: list[Document],
        output_path: str | Path = "output/intl_student_financial_guide.pdf",
    ) -> Path:
        """
        Compile all scraped documents into a single PDF.
        Supports fpdf2 if available, or falls back to standard library PDF compilation.
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
            pdf.cell(0, 15, "International Student Financial Guide", new_x="LMARGIN", new_y="NEXT", align="C")
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
            logger.info(f"📄 PDF written via fpdf2 -> {output_path.resolve()}")
            return output_path.resolve()

        except ImportError:
            logger.info("fpdf2 not detected. Compiling PDF using built-in minimal PDF generator...")
            return self._export_minimal_pdf(documents, output_path)

    def _export_minimal_pdf(self, documents: list[Document], output_path: Path) -> Path:
        """Pure-python standards-compliant minimal PDF writer without external dependencies."""
        def escape_pdf_str(s: str) -> str:
            clean = s.encode("latin-1", errors="replace").decode("latin-1")
            clean = clean.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            return clean

        pages_content = []

        # Cover page
        cover_stream = [
            "BT",
            "/F1 20 Tf",
            "50 720 Td",
            "(International Student Financial Literacy Guide) Tj",
            "/F1 11 Tf",
            "0 -28 Td",
            "(Regulatory Guidance: Taxes, Banking, & Investments for Non-Citizens) Tj",
            "0 -22 Td",
            f"(Total Documents Compiled: {len(documents)}) Tj",
            "0 -36 Td",
            "/F1 13 Tf",
            "(Table of Contents:) Tj",
            "/F1 10 Tf",
        ]
        y_offset = -18
        for i, doc in enumerate(documents[:25], 1):
            title = f"{i}. [{doc.source}] {doc.section}"
            title = escape_pdf_str(title[:72])
            cover_stream.append(f"0 {y_offset} Td")
            cover_stream.append(f"({title}) Tj")
            y_offset = -15
        cover_stream.append("ET")
        pages_content.append("\n".join(cover_stream))

        # Content pages
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

            page_size = 45
            for page_idx in range(0, max(1, len(wrapped_lines)), page_size):
                sub_lines = wrapped_lines[page_idx:page_idx + page_size]
                stream = [
                    "BT",
                    "/F1 13 Tf",
                    "40 750 Td",
                    f"({escape_pdf_str(doc.source)} - {escape_pdf_str(doc.section[:45])}) Tj",
                    "/F1 8 Tf",
                    "0 -15 Td",
                    f"(URL: {escape_pdf_str(doc.url[:85])}) Tj",
                    "/F1 9 Tf",
                    "0 -18 Td",
                ]
                for l in sub_lines:
                    safe_l = escape_pdf_str(l)
                    stream.append(f"({safe_l}) Tj")
                    stream.append("0 -13 Td")
                stream.append("ET")
                pages_content.append("\n".join(stream))

        # Build PDF object tree
        objects = []
        objects.append("<< /Type /Catalog /Pages 2 0 R >>")
        page_obj_ids = [3 + i * 2 for i in range(len(pages_content))]
        kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
        objects.append(f"<< /Type /Pages /Kids [{kids_str}] /Count {len(pages_content)} >>")

        for idx, p_stream in enumerate(pages_content):
            p_obj_id = 3 + idx * 2
            c_obj_id = p_obj_id + 1
            objects.append(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {c_obj_id} 0 R "
                f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>"
            )
            stream_bytes = p_stream.encode("latin-1", errors="replace")
            objects.append(f"<< /Length {len(stream_bytes)} >>\nstream\n{p_stream}\nendstream")

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
        logger.info(f"📄 Minimal PDF compiled successfully -> {output_path.resolve()}")
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
        default="output/intl_student_financial_guide.pdf",
        help="Output PDF path (default: output/intl_student_financial_guide.pdf)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between HTTP requests (default: 1.0)",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Scrape only — skip PDF generation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    scraper = IntlStudentScraper(request_delay=args.delay)
    docs = scraper.scrape_all()

    if not args.no_pdf and docs:
        pdf_path = scraper.export_pdf(docs, output_path=args.pdf)
        print(f"\n✅ PDF saved to: {pdf_path}")
    else:
        print(f"\n✅ Scraped {len(docs)} documents.")
