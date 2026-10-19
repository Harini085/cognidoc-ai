
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Generate an answer using only retrieved PDF passages.
    Ask the model to cite the source and page for its claims.
    """
    if not question.strip():
        raise ValueError("Question must not be empty.")

    if not retrieved_chunks:
        raise ValueError("No retrieved passages were provided.")

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from the .env file.")

    if not model:
        raise ValueError("GROQ_MODEL is missing from the .env file.")

    context_parts = []

    for number, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source", "Unknown source")
        page = chunk.get("page", "Unknown page")
        text = chunk.get("text", "").strip()

        if text:
            context_parts.append(
                f"[Passage {number} | Source: {source} | Page: {page}]\n"
                f"{text}"
            )

    if not context_parts:
        raise ValueError("The retrieved passages contain no usable text.")

    context = "\n\n".join(context_parts)

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
    "role": "system",
    "content": (
        "You are Cognidoc AI, a document question-answering assistant. "
        "Answer using only the supplied document passages. Do not use "
        "outside knowledge or invent facts.\n\n"

        "For factual questions, answer directly using the relevant evidence.\n"
        "For summary questions, combine the relevant findings across all "
        "supplied passages into a coherent summary. Include the main results, "
        "comparisons, and conclusions only when supported by the passages. "
        "Do not require a single passage to contain the entire summary.\n\n"

        "Cite factual claims using the source filename and page number "
        "provided in the passages, for example [sample.pdf, p. 2]. "
        "Place citations near the claims they support. Do not invent citations "
        "or cite a page unless its supplied passage supports the claim.\n\n"

        "If the supplied passages do not contain enough evidence to answer "
        "the question, clearly state that the available document passages "
        "are insufficient. Do not guess or fill gaps with outside knowledge.\n\n"

        "Keep answers clear, concise, and appropriately detailed for the "
        "question."
    ),
},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved document passages:\n{context}"
                ),
            },
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content

    if not answer or not answer.strip():
        raise RuntimeError("Groq returned an empty answer.")

    return answer.strip()
