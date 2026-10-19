
from pathlib import Path

from pdf_processor import extract_pdf_pages
from chunker import chunk_pages
from embeddings import load_embedding_model, embed_chunks
from retriever import build_index, retrieve_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "sample.pdf"


def test_retrieval():
    # 1. Read the PDF.
    pages = extract_pdf_pages(PDF_PATH)

    # 2. Split the document into chunks.
    chunks = chunk_pages(
    pages,
    chunk_size=700,
    overlap=120,
)

    # 3. Generate embeddings.
    model = load_embedding_model()
    embeddings = embed_chunks(chunks, model)

    # 4. Build the FAISS index.
    index = build_index(embeddings)

    assert index.ntotal == len(chunks)

    # 5. Search using a question about the research paper.
    question = "Which seven mental health conditions were included in the dataset?"

    results = retrieve_chunks(
        question=question,
        chunks=chunks,
        index=index,
        model=model,
        top_k=3,
    )

    assert len(results) == 3

    print(f"Indexed chunks: {index.ntotal}")
    print(f"\nQuestion: {question}")
    print("\nRetrieved passages:")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Text: {result['text'][:350]}")

    # Check whether the results retain citation metadata.
    for result in results:
        assert result["source"] == "sample.pdf"
        assert result["page"] >= 1
        assert result["text"].strip()


if __name__ == "__main__":
    test_retrieval()
