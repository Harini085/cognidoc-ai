
from pathlib import Path

from chunker import chunk_pages
from pdf_processor import extract_pdf_pages
from embeddings import load_embedding_model, embed_chunks
from retriever import build_index, retrieve_chunks


def test_retrieval_finds_key_accuracy_results():
    pdf_path = Path(__file__).resolve().parents[1] / "data" / "sample.pdf"

    pages = extract_pdf_pages(pdf_path)
    chunks = chunk_pages(pages, chunk_size=700, overlap=120)

    model = load_embedding_model()
    embeddings = embed_chunks(chunks, model)
    index = build_index(embeddings)

    results = retrieve_chunks(
        question=(
            "Compare the accuracy of traditional NLP, "
            "prompt-engineered LLM, and fine-tuned LLM."
        ),
        chunks=chunks,
        index=index,
        model=model,
        top_k=5,
    )

    assert results, "No relevant passages were retrieved."

    retrieved_text = " ".join(
        result["text"].lower() for result in results
    )

    assert "95%" in retrieved_text, "Traditional NLP accuracy not retrieved."
    assert "65%" in retrieved_text, "Prompt-engineered LLM accuracy not retrieved."
    assert "91%" in retrieved_text, "Fine-tuned LLM accuracy not retrieved."
