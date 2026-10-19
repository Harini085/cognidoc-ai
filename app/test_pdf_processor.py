
from pathlib import Path

from pdf_processor import extract_pdf_pages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "sample.pdf"


def test_extract_pdf_pages():
    pages = extract_pdf_pages(PDF_PATH)

    assert len(pages) > 0
    assert pages[0]["page"] >= 1
    assert pages[0]["text"]
    assert pages[0]["source"] == "sample.pdf"


if __name__ == "__main__":
    pages = extract_pdf_pages(PDF_PATH)

    print(f"Extracted {len(pages)} pages.")

    for page in pages[:2]:
        print(f"\n--- Page {page['page']} ---")
        print(page["text"][:500])
