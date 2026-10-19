
from pathlib import Path

from chunker import chunk_pages
from pdf_processor import extract_pdf_pages
from embeddings import load_embedding_model, embed_chunks
from retriever import build_index, retrieve_chunks


def test_pdf_to_retrieval_pipeline():
    pdf_path = Path(__file__).resolve().parents[1] / "data" / "sample.pdf"

    assert pdf_path.exists(), f"Sample PDF not found: {pdf_path}"

    pages = extract_pdf_pages(pdf_path)
    assert pages, "PDF extraction returned no pages."

    chunks = chunk_pages(pages, chunk_size=700, overlap=120)
    assert chunks, "Chunking returned no passages."

    model = load_embedding_model()
    embeddings = embed_chunks(chunks, model)
    index = build_index(embeddings)

    results = retrieve_chunks(
        question="What is the dataset size and what are the seven mental health categories?",
        chunks=chunks,
        index=index,
        model=model,
        top_k=5,
    )

    assert results, "Retrieval returned no results."

    retrieved_text = " ".join(
        result["text"].lower() for result in results
    )

    assert "52,681" in retrieved_text, (
        "The dataset size was not found in the retrieved passages."
    )
    assert "depression" in retrieved_text
    assert "anxiety" in retrieved_text
