"""
Phase 4 & 5 — Search & Recommendation
======================================
search()     — encode query → FAISS lookup → ranked results with cosine scores
recommend()  — retrieve item embedding → find nearest neighbors → exclude self
"""

import numpy as np
from src.embeddings import encode_query


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(query: str, domain: str, indexes: dict, top_k: int = 5) -> list[dict]:
    """
    Semantic search within a single domain.

    Parameters
    ----------
    query   : natural-language query string
    domain  : one of 'jobs', 'courses', 'products'
    indexes : dict returned by build_all_indexes
    top_k   : number of results

    Returns
    -------
    List of dicts with keys: rank, id, title, domain, similarity_score
    Similarity scores are in [0, 1] (cosine similarity).
    """
    if domain not in indexes:
        raise ValueError(f"Unknown domain '{domain}'. Available: {list(indexes.keys())}")

    index, meta, _ = indexes[domain]
    q_emb = encode_query(query)                       # (1, dim), already normalized

    scores, idxs = index.search(q_emb, top_k)         # IP on normalized vecs → cosine

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        if idx == -1:
            continue
        m = meta[idx]
        results.append({
            "rank": rank,
            "id": m["id"],
            "title": m["title"],
            "domain": m["domain"],
            "similarity_score": round(float(np.clip(score, 0, 1)), 4),
        })

    return results


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def recommend(item_id: int, domain: str, indexes: dict,
              top_k: int = 5) -> list[dict]:
    """
    Recommend items similar to *item_id* within the same domain.
    Excludes the item itself from results.

    Parameters
    ----------
    item_id  : the 'id' field from the cleaned CSV
    domain   : 'jobs', 'courses', or 'products'
    indexes  : dict from build_all_indexes
    top_k    : number of recommendations

    Returns
    -------
    List of dicts (same schema as search results).
    """
    if domain not in indexes:
        raise ValueError(f"Unknown domain '{domain}'. Available: {list(indexes.keys())}")

    index, meta, embeddings = indexes[domain]

    if embeddings is None:
        raise RuntimeError(
            f"Embeddings not available for domain '{domain}'. "
            "Rebuild the index with force=True."
        )

    # Find the internal index position for the given item_id
    pos = None
    for k, v in meta.items():
        if v["id"] == item_id:
            pos = k
            break

    if pos is None:
        raise ValueError(f"item_id {item_id} not found in domain '{domain}'")

    # Retrieve embedding and search
    item_emb = embeddings[pos:pos+1]                   # (1, dim)
    scores, idxs = index.search(item_emb, top_k + 1)  # +1 to account for self

    results = []
    rank = 1
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1 or idx == pos:                    # skip self
            continue
        m = meta[idx]
        results.append({
            "rank": rank,
            "id": m["id"],
            "title": m["title"],
            "domain": m["domain"],
            "similarity_score": round(float(np.clip(score, 0, 1)), 4),
        })
        rank += 1
        if rank > top_k:
            break

    return results


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_results(results: list[dict], header: str = "Results") -> None:
    """Print a formatted table of search/recommendation results."""
    print(f"\n{'-'*60}")
    print(f"  {header}")
    print(f"{'-'*60}")
    print(f"  {'Rank':<6}{'Title':<42}{'Score':<8}")
    print(f"  {'-'*5} {'-'*41} {'-'*7}")
    for r in results:
        title = r["title"][:40] + ".." if len(r["title"]) > 40 else r["title"]
        print(f"  {r['rank']:<6}{title:<42}{r['similarity_score']:<8}")
    print()
