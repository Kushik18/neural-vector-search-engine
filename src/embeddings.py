"""
Phase 2 — Embedding Engine
===========================
Uses sentence-transformers (all-MiniLM-L6-v2) to encode text into
384-dimensional vectors.  CPU-only, batch encoding with progress bars.

All embeddings are L2-normalized so that dot-product == cosine similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model Singleton
# ---------------------------------------------------------------------------

_model = None


def get_model() -> SentenceTransformer:
    """Load model once and reuse."""
    global _model
    if _model is None:
        print("Loading sentence-transformers model (all-MiniLM-L6-v2) ...")
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        print("Model loaded.")
    return _model


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each row so dot-product == cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)          # avoid division by zero
    return (embeddings / norms).astype(np.float32)


def encode_texts(texts: list[str], batch_size: int = 64,
                 show_progress: bool = True) -> np.ndarray:
    """
    Encode a list of texts into L2-normalized embeddings.
    Returns ndarray of shape (N, 384).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    embeddings = normalize_embeddings(embeddings)
    return embeddings


def encode_query(query: str) -> np.ndarray:
    """Encode a single query string and return L2-normalized vector (1, 384)."""
    model = get_model()
    emb = model.encode([query], convert_to_numpy=True)
    emb = normalize_embeddings(emb)
    return emb


if __name__ == "__main__":
    sample = ["hello world", "semantic search engine"]
    vecs = encode_texts(sample, show_progress=False)
    print(f"Shape: {vecs.shape}")
    print(f"Norms (should be ~1): {np.linalg.norm(vecs, axis=1)}")
