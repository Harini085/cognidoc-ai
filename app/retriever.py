
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Words that do not add much value to keyword matching.
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on",
    "at", "to", "for", "from", "and", "or", "by", "with", "about",
    "what", "which", "who", "when", "where", "how", "did", "does",
    "do", "this", "that", "these", "those", "it", "its", "their",
    "they", "we", "you", "i", "me", "my", "our", "can", "could",
    "would", "should", "please", "give", "tell", "explain",
}

SUMMARY_TERMS = {
    "summarize", "summary", "findings", "finding", "results", "result",
    "conclusion", "conclusions", "outcomes", "outcome", "overall",
    "performance", "comparison", "compared", "evaluation", "evaluated",
    "accuracy", "precision", "recall", "f1", "limitations",
}


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS inner-product index from normalized embeddings."""
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numerical values.")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Embeddings must be normalized.")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings))
    return index


def _tokens(text: str) -> set[str]:
    """Convert text into a set of normalized, meaningful tokens."""
    words = re.findall(r"\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b",
                       text.lower())
    return {
        word for word in words
        if word not in STOP_WORDS and len(word) > 1
    }


def _keyword_score(question: str, chunk_text: str) -> float:
    """Estimate keyword overlap, including terms useful for study summaries."""
    query_terms = _tokens(question)
    chunk_terms = _tokens(chunk_text)

    if not query_terms or not chunk_terms:
        return 0.0

    # Expand broad study-summary questions with related research terminology.
    question_lower = question.lower()
    if any(term in question_lower for term in SUMMARY_TERMS):
        query_terms |= {
            "findings", "results", "conclusion", "performance",
            "accuracy", "precision", "recall", "f1", "model",
            "models", "classification", "dataset", "compared",
            "comparison", "significant", "outperformed",
        }

    matched_terms = query_terms & chunk_terms
    return len(matched_terms) / len(query_terms)


def retrieve_chunks(
    question: str,
    chunks: list[dict],
    index: faiss.Index,
    model: SentenceTransformer,
    top_k: int = 5,
    min_score: float | None = None,
) -> list[dict]:
    """
    Rank chunks using semantic similarity and keyword relevance.

    The 'score' field remains the original semantic similarity.
    'rerank_score' records the combined ranking score.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if min_score is not None and not -1.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between -1 and 1.")

    if index.ntotal != len(chunks):
        raise ValueError("Index size does not match the chunk count.")

    if index.ntotal == 0:
        return []

    if index.d != model.get_embedding_dimension():
        raise ValueError("Index dimensions do not match the model.")

    query_vector = model.encode(
        [question.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_vector = np.asarray(query_vector, dtype=np.float32)

    if query_vector.shape != (1, index.d):
        raise ValueError("Question embedding has an unexpected shape.")

    if not np.isfinite(query_vector).all():
        raise ValueError("Question embedding contains invalid values.")

    # Retrieve all chunks so potentially useful result passages are not
    # discarded before keyword reranking.
    scores, indices = index.search(
        np.ascontiguousarray(query_vector),
        index.ntotal,
    )

    candidates = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        semantic_score = float(score)

        if min_score is not None and semantic_score < min_score:
            continue

        chunk = dict(chunks[int(idx)])
        text = chunk.get("text", "").strip()

        if not text:
            continue

        keyword_score = _keyword_score(question, text)

        # Semantic relevance remains the main signal.
        combined_score = 0.80 * semantic_score + 0.20 * keyword_score

        chunk["score"] = semantic_score
        chunk["keyword_score"] = keyword_score
        chunk["rerank_score"] = combined_score
        candidates.append(chunk)

    candidates.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return candidates[:top_k]
