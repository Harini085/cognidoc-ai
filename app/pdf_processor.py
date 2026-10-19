
from pathlib import Path

import pymupdf


def extract_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF while preserving page numbers.

    Returns a list of dictionaries, one for each page
    containing extractable text.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if not pdf_path.is_file():
        raise ValueError(
            f"The path is not a file: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            "The supplied file must have a .pdf extension."
        )

    pages = []

    with pymupdf.open(pdf_path) as document:
        if document.is_encrypted:
            raise ValueError(
                "The PDF is encrypted and cannot be read."
            )

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text,
                    "source": pdf_path.name,
                })

    if not pages:
        raise ValueError(
            "No extractable text found. "
            "The PDF may be scanned or contain only images."
        )

    return pages
