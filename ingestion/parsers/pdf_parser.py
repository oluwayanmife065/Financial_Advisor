"""
PDF parser — extracts clean text from PDF files using PyMuPDF.

Each page of a PDF becomes one Document object with full metadata.
"""

import re
import fitz  # PyMuPDF
from pathlib import Path
from datetime import datetime
from ingestion.document import Document


# Maps folder names to human-readable source labels
SOURCE_LABELS = {
    "sec": "SEC/Investor.gov",
    "cfpb": "CFPB",
    "finra": "FINRA",
    "fed": "Federal Reserve",
}


def parse_pdf(pdf_path: str, source: str) -> list[Document]:
    """
    Parse a PDF file into a list of Documents, one per non-empty page.

    Args:
        pdf_path: Path to the PDF file (absolute or relative).
        source: Human-readable source label (e.g. "SEC/Investor.gov").

    Returns:
        List of Document objects with cleaned text and metadata.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    documents = []
    date_ingested = datetime.now().isoformat()

    with fitz.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            raw_text = page.get_text("text")
            clean_text = _clean_text(raw_text)

            # Skip pages with no meaningful content
            if len(clean_text) < 50:
                continue

            documents.append(Document(
                text=clean_text,
                source=source,
                section=f"page_{page_num}",
                url=f"{str(path.resolve())}#page_{page_num}",
                date_ingested=date_ingested,
            ))

    return documents


def parse_pdf_folder(folder_path: str, source: str) -> list[Document]:
    """
    Parse all PDFs inside a folder, returning all Documents combined.

    Args:
        folder_path: Path to the folder containing PDFs.
        source: Human-readable source label applied to all documents.

    Returns:
        Combined list of Document objects from all PDFs in the folder.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    all_documents = []
    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"  [WARN] No PDFs found in {folder_path}")
        return []

    for pdf_file in pdf_files:
        docs = parse_pdf(str(pdf_file), source)
        all_documents.extend(docs)
        print(f"  [OK] {pdf_file.name} → {len(docs)} pages")

    return all_documents


def infer_source_label(pdf_path: str) -> str:
    """
    Infer the source label from the PDF's parent folder name.

    Looks at the folder the PDF lives in and maps it to a label.
    Falls back to the raw folder name if not in SOURCE_LABELS.

    Args:
        pdf_path: Path to a PDF file.

    Returns:
        Human-readable source label string.
    """
    folder_name = Path(pdf_path).parent.name.lower()
    return SOURCE_LABELS.get(folder_name, folder_name.upper())


def _clean_text(text: str) -> str:
    """
    Remove common PDF extraction noise from raw text.

    Cleans:
    - Hyphenated line breaks (e.g. "invest-\\nment" → "investment")
    - Excessive blank lines (3+ → 2)
    - Leading/trailing whitespace

    Args:
        text: Raw text string from PyMuPDF.

    Returns:
        Cleaned text string.
    """
    # Rejoin words split across lines by a hyphen
    text = text.replace("-\n", "")
    # Collapse runs of 3+ newlines into a paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip surrounding whitespace
    return text.strip()
