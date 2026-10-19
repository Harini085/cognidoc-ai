
from pathlib import Path

from chunker import chunk_pages
from pdf_processor import extract_pdf_pages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "sample.pdf"


def test_chunking():
    pages = extract_pdf_pages(PDF_PATH)
    chunks = chunk_pages(pages)

    assert len(chunks) > 0

    # Every chunk must contain text and source metadata.
    for chunk in chunks:
        assert chunk["text"].strip()
        assert chunk["source"] == "sample.pdf"
        assert isinstance(chunk["page"], int)
        assert chunk["page"] >= 1
        assert isinstance(chunk["chunk_id"], int)

    # IDs should be sequential and unique.
    assert [c["chunk_id"] for c in chunks] == list(
        range(len(chunks))
    )

    print(f"Extracted pages: {len(pages)}")
    print(f"Created chunks: {len(chunks)}")
    print("\nFirst chunk:")
    print(chunks[0])


if __name__ == "__main__":
    test_chunking()
