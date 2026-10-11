"""
Phase 3 — Vector Indexing
=========================
Builds a separate FAISS IndexFlatIP (inner-product) index for each domain.
Because embeddings are L2-normalized, IP score == cosine similarity ∈ [0, 1].

Indexes are saved/loaded from disk to avoid redundant rebuilds.
"""

import os
import time
import pickle
import numpy as np
import faiss
import pandas as pd

from src.embeddings import encode_texts


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INDEX_DIR = "indexes"


def _index_path(root: str, domain: str) -> str:
    return os.path.join(root, INDEX_DIR, f"{domain}.index")


def _meta_path(root: str, domain: str) -> str:
    return os.path.join(root, INDEX_DIR, f"{domain}_meta.pkl")


# ---------------------------------------------------------------------------
# Build / Load
# ---------------------------------------------------------------------------

def build_index(df: pd.DataFrame, root: str, domain: str,
                batch_size: int = 64, force: bool = False) -> tuple:
    """
    Build (or load) a FAISS index for *domain*.

    Returns (index, metadata_dict, embeddings_array).
    metadata_dict maps integer position → {id, title, domain}.
    """
    os.makedirs(os.path.join(root, INDEX_DIR), exist_ok=True)
    idx_path = _index_path(root, domain)
    meta_path = _meta_path(root, domain)

    # ---- Load existing ----
    if not force and os.path.exists(idx_path) and os.path.exists(meta_path):
        t0 = time.time()
        index = faiss.read_index(idx_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        elapsed = time.time() - t0
        print(f"  [{domain}] Loaded existing index ({index.ntotal} vectors) "
              f"in {elapsed:.2f}s")
        # Try to load embeddings too
        emb_path = os.path.join(root, INDEX_DIR, f"{domain}_embeddings.npy")
        embeddings = None
        if os.path.exists(emb_path):
            embeddings = np.load(emb_path)
        return index, meta, embeddings

    # ---- Build new ----
    print(f"  [{domain}] Building index for {len(df)} items ...")
    t0 = time.time()
    texts = df["text"].tolist()
    embeddings = encode_texts(texts, batch_size=batch_size)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner-product on normalized vecs = cosine
    index.add(embeddings)

    # Metadata
    meta = {}
    for i, row in df.iterrows():
        meta[i] = {
            "id": row["id"],
            "title": row["title"],
            "domain": row["domain"],
        }

    # Save
    faiss.write_index(index, idx_path)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    # Also save raw embeddings so recommend() can retrieve item vectors
    emb_path = os.path.join(root, INDEX_DIR, f"{domain}_embeddings.npy")
    np.save(emb_path, embeddings)

    elapsed = time.time() - t0
    print(f"  [{domain}] Built index ({index.ntotal} vectors) in {elapsed:.2f}s")
    print(f"  [{domain}] Saved -> {idx_path}")

    return index, meta, embeddings


def build_all_indexes(data: dict[str, pd.DataFrame], root: str,
                      batch_size: int = 64,
                      force: bool = False) -> dict:
    """
    Build indexes for every domain.
    Returns dict[domain] → (index, meta, embeddings).
    """
    indexes = {}
    for domain, df in data.items():
        idx, meta, embs = build_index(df, root, domain,
                                      batch_size=batch_size, force=force)
        indexes[domain] = (idx, meta, embs)
    return indexes


if __name__ == "__main__":
    print("Use via main.py or import from src.index_builder")
