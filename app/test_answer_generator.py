
from pathlib import Path

from pdf_processor import extract_pdf_pages
from chunker import chunk_pages
from embeddings import load_embedding_model, embed_chunks
from retriever import build_index, retrieve_chunks
from answer_generator import generate_answer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "sample.pdf"


def test_answer_generation():
    # 1. Extract text from the PDF.
    pages = extract_pdf_pages(PDF_PATH)

    # 2. Split the extracted text into chunks.
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

    # 5. Retrieve passages relevant to the question.
    question = (
        "Which seven mental health conditions were included in the dataset?"
    )

    results = retrieve_chunks(
        question=question,
        chunks=chunks,
        index=index,
        model=model,
        top_k=3,
    )

    # 6. Generate an answer using Groq.
    answer = generate_answer(
        question=question,
        retrieved_chunks=results,
    )

    print("\nQuestion:")
    print(question)

    print("\nCognidoc AI answer:")
    print(answer)


if __name__ == "__main__":
    test_answer_generation()
