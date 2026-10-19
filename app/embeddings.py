
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(
    model_name: str = DEFAULT_MODEL,
) -> SentenceTransformer:
    """Load the sentence embedding model."""
    return SentenceTransformer(model_name)


def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
) -> np.ndarray:
    """Convert text chunks into normalized embedding vectors."""
    if not chunks:
        raise ValueError("Cannot embed an empty list of chunks.")

    texts = [chunk["text"] for chunk in chunks]

    if any(not text.strip() for text in texts):
        raise ValueError("Chunks must not contain empty text.")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError("Expected a two-dimensional embedding array.")

    if embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding count does not match chunk count.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numerical values.")

    return embeddings
