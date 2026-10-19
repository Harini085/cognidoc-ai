
import numpy as np
from pathlib import Path

from pdf_processor import extract_pdf_pages
from chunker import chunk_pages
from embeddings import load_embedding_model, embed_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "sample.pdf"


def test_embeddings():
    # Step 1: Extract the PDF.
    pages = extract_pdf_pages(PDF_PATH)

    # Step 2: Split the extracted text into chunks.
    chunks = chunk_pages(pages)

    # Step 3: Load the embedding model.
    model = load_embedding_model()

    # Step 4: Generate the embeddings.
    embeddings = embed_chunks(chunks, model)

    # Step 5: Verify the result.
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == 384
    assert np.isfinite(embeddings).all()

    # Normalized vectors should have approximately unit length.
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    print(f"PDF pages: {len(pages)}")
    print(f"Text chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Vector dimension: {embeddings.shape[1]}")
    print("Embedding verification successful!")


if __name__ == "__main__":
    test_embeddings()
