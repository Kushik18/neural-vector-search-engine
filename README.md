# Multi-Domain Semantic Search & Recommendation Engine

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![FAISS IndexFlatIP](https://img.shields.io/badge/FAISS-IndexFlatIP-orange)
![Model: all-MiniLM-L6-v2](https://img.shields.io/badge/Model-all--MiniLM--L6--v2-green)
![CPU Only](https://img.shields.io/badge/Hardware-CPU--Only-lightgrey)
![Audit: SAFE](https://img.shields.io/badge/Audit-SAFE-brightgreen)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Model Architecture](#2-model-architecture)
3. [Vector Database & Index](#3-vector-database--index)
4. [Data Pipeline](#4-data-pipeline)
5. [Query Processing Flow](#5-query-processing-flow)
6. [Performance Metrics](#6-performance-metrics)
7. [Deployment Interface](#7-deployment-interface)
8. [Benchmark: Semantic vs TF-IDF](#8-benchmark-semantic-vs-tf-idf)
9. [Project Structure](#9-project-structure)
10. [Installation & Usage](#10-installation--usage)

---

## 1. Problem Statement

### Why Keyword Search Is Insufficient

Traditional keyword (lexical) search relies on exact token overlap between query and document. This fails in three critical ways:

| Failure Mode | Example | Keyword Match | Semantic Match |
|---|---|---|---|
| Vocabulary mismatch | Query: "AI job" vs Document: "Machine Learning Engineer" | ❌ None | ✅ High similarity |
| Synonym blindness | "python course" vs "learn programming" | ❌ None | ✅ Matched |
| Context insensitivity | "wireless earbuds" vs "Bluetooth headphones" | ❌ None | ✅ Matched |

### The Semantic Gap

The **semantic gap** is the disconnect between surface-level token similarity and true meaning similarity. TF-IDF and BM25 rank documents by term frequency, so two semantically identical sentences using different vocabulary score near zero. Semantic search closes this gap by representing text as dense vectors in a shared embedding space where **geometric distance = semantic distance**.

---

## 2. Model Architecture

### Embedding Model

| Property | Value |
|---|---|
| **Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Base Architecture** | BERT (MiniLM-L6 — 6-layer distilled) |
| **Pretrained or Fine-tuned** | Pretrained on 1B+ sentence pairs via contrastive learning |
| **Vector Dimension** | **384** |
| **Parameters** | ~22.7 million |
| **Max Sequence Length** | 256 tokens |
| **Device** | CPU (`device="cpu"` — no GPU dependency) |
| **Batch Size** | 64 texts per forward pass |
| **Output** | Float32 NumPy array, shape `(N, 384)` |

### Post-Processing: L2 Normalization

All embeddings are L2-normalized immediately after encoding:

```python
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = (embeddings / norms).astype(np.float32)
```

The mathematical consequence is that:

```
cosine_similarity(a, b) = dot(a, b)   # when ||a|| = ||b|| = 1
```

This allows FAISS **Inner Product** search to return exact cosine similarity scores in **[0, 1]** with no additional computation.

---

## 3. Vector Database & Index

### FAISS Configuration

| Property | Value |
|---|---|
| **Library** | `faiss-cpu` (Facebook AI Similarity Search) |
| **Index Type** | `faiss.IndexFlatIP` — Flat Inner Product |
| **Search Type** | Exact (brute-force) — no approximation |
| **Similarity Metric** | **Cosine Similarity** (via Inner Product on L2-normalized vecs) |
| **Score Range** | 0.0 (orthogonal) → 1.0 (identical) |
| **Separate Index Per Domain** | Yes — Jobs, Courses, Products are isolated |

### Why `IndexFlatIP` Over Approximate Methods (IVF/HNSW)?

`IndexFlatIP` performs an exhaustive scan over all vectors. For corpus sizes in this project (≤ 5,000 items per domain), this yields:
- **Zero approximation error** (exact nearest neighbors)
- **Sub-millisecond query latency** at this scale (verified: see §6)
- **No training phase** needed (unlike `IndexIVFFlat` which requires `k-means` clustering)

For corpora exceeding ~1M vectors, `IndexIVFFlat` (with `nlist=4096`) or `IndexHNSWFlat` would be preferred.

### Stored Artifacts Per Domain

| File | Content |
|---|---|
| `indexes/{domain}.index` | Serialized FAISS index |
| `indexes/{domain}_meta.pkl` | `dict[int, {id, title, domain}]` — position → metadata |
| `indexes/{domain}_embeddings.npy` | Raw float32 embedding matrix `(N, 384)` used by `recommend()` |

---

## 4. Data Pipeline

### Dataset Summary

| Domain | Source Files | Raw Rows | Processed Records | Notes |
|---|---|---|---|---|
| **Jobs** | `clean_jobs.csv` | ~1,048 | 1,041 | 7 removed (short/dup) |
| **Courses** | `udemy_courses.csv` | 3,678 | 3,672 | 6 removed (short/dup) |
| **Products** | 3 CSVs (review-level) | ~67,992 | 131 | Aggregated by `(name, brand)` |

### Text Preprocessing Rules

```
Input text
   ↓ Fill NaN with ""
   ↓ Detect title + text columns (heuristic scoring — no hardcoded names)
   ↓ Concatenate all meaningful columns with space separator
   ↓ Lowercase
   ↓ Normalize whitespace (regex: \s+ → single space)
   ↓ Filter: len(text) < 20 characters → drop
   ↓ Deduplicate on full text
   ↓ Output: id | title | text | domain
```

**Strict non-operations** (intentionally avoided):
- ❌ No stopword removal (preserves domain context e.g. "for beginners")
- ❌ No stemming / lemmatization (preserves morphological forms)
- ❌ No aggressive punctuation stripping (preserves abbreviations)
- ❌ No number removal (relevant in job/course descriptions)

### Column Detection Heuristic

Columns are scored against keyword lists at runtime — no column names are hardcoded:

```python
TITLE_KEYWORDS = ["title", "name", "course_title", "job_title", "product_name"]
TEXT_KEYWORDS  = ["title", "name", "description", "text", "review", "summary",
                  "subject", "category", "brand", "company", "location", "level", ...]
EXCLUDE_KEYWORDS = ["id", "url", "date", "price", "rating", "num_", "is_", ...]
```

### Products: Review Aggregation

Products CSVs contain one row per customer review. Aggregation logic:

```
Group Key: product_id (if present, >50% filled) else (name + brand)
   ↓ Aggregate: first(title), first(brand), first(categories)
   ↓ Concatenate: up to 10 review texts per product (head(10))
   ↓ Concatenate: up to 10 review titles per product
   ↓ Final text: name + brand + categories + review_titles + review_texts
Output: 131 unique product records from 67,992 review rows
```

### Embedding Generation

```python
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
embeddings = model.encode(texts, batch_size=64, convert_to_numpy=True)
embeddings = L2_normalize(embeddings)   # → shape (N, 384), dtype float32
```

### Index Creation

```python
index = faiss.IndexFlatIP(dim=384)      # Inner Product index
index.add(embeddings)                   # O(N) — flat insert
faiss.write_index(index, path)          # Persist to disk
```

---

## 5. Query Processing Flow

```
User Input (query string)
        │
        ▼
  encode_query(query)
  • Tokenize with WordPiece tokenizer (max 256 tokens)
  • Pass through 6-layer MiniLM transformer
  • Mean-pool token embeddings → (1, 384) vector
  • L2-normalize → cosine-ready unit vector
        │
        ▼
  FAISS IndexFlatIP.search(q_emb, top_k)
  • Compute dot product between query and all N index vectors (brute-force)
  • Return top_k (score, index_position) pairs, sorted descending
  • Time complexity: O(N × d)  where d=384
        │
        ▼
  Metadata Lookup
  • Map index_position → {id, title, domain} via meta dict
  • Clip scores to [0.0, 1.0]  (np.clip)
        │
        ▼
  Ranked Results
  • List of dicts: rank | id | title | domain | similarity_score
  • Sorted by similarity_score descending
  • Strictly domain-isolated (no cross-domain mixing)
```

### Recommendation Flow

```
item_id (integer)
        │
        ▼
  Retrieve embeddings[pos]  from {domain}_embeddings.npy  → (1, 384)
        │
        ▼
  FAISS IndexFlatIP.search(item_emb, top_k + 1)
        │
        ▼
  Filter: idx == pos  → skip self-match
        │
        ▼
  Return top_k nearest neighbors (excluding self)
```

---

## 6. Performance Metrics

### Evaluation Methodology

- **Split**: 80% train / 20% test (stratified by none — random seed 42)
- **Index built on train split ONLY** — no test data leakage
- **Query**: each test item's text is used as the query against the train index
- **Hit criterion**: `cosine_similarity(query, top_result) >= 0.8`
- **Metric**: Precision@1 (Top-1 Accuracy) and Precision@3 (Top-3 Accuracy)

> **Threshold Choice**: Score ≥ 0.8 enforces genuine semantic similarity. A lenient threshold (e.g. 0.5) would inflate accuracy to ~100% because most in-domain items score above 0.5 via cosine similarity — this was verified and rejected during audit.

### Measured Results

| Domain | Train Size | Test Size | Top-1 Acc | Top-3 Acc | Avg Latency |
|---|---|---|---|---|---|
| **Jobs** | 832 | 209 | ~35.9% | ~35.9% | < 1 ms |
| **Courses** | 2,937 | 735 | ~77.8% | ~77.8% | < 1 ms |
| **Products** | 104 | 27 | ~77.8% | ~77.8% | < 1 ms |

> **Note on Jobs accuracy**: The Jobs domain has high lexical diversity (varied job titles, locations) — the 0.8 cosine threshold is strict, and many semantically related jobs score ~0.7–0.79. Lowering to 0.7 increases Top-1 to ~75%+. This is an expected property of the embedding space, not a defect.

### Latency Breakdown

| Operation | Time (CPU, i5-class) |
|---|---|
| Model load (first call) | ~3–5 s (cached after) |
| Encode single query | ~30–80 ms |
| FAISS search (3,672 vectors) | < 0.5 ms |
| Metadata lookup | < 0.1 ms |
| **End-to-end per query** | **~30–80 ms** |

### Dataset & Index Size

| Domain | Vectors | Index File Size | Embedding File Size |
|---|---|---|---|
| Jobs | 1,041 | 1.5 MB | 1.5 MB |
| Courses | 3,672 | 5.4 MB | 5.4 MB |
| Products | 131 | 0.19 MB | 0.19 MB |
| **Total** | **4,844** | **7.09 MB** | **7.09 MB** |

---

## 7. Deployment Interface

This system is deployed as a **local CLI application** — no API server, no web framework, no cloud dependency.

### Modes

| Mode | Trigger | Description |
|---|---|---|
| **Demo Mode** | Select `1` at startup | Preset queries → full pipeline → evaluation table |
| **Interactive CLI** | Select `2` at startup | Free-form search and recommend loop |

### CLI Entrypoint

```bash
python main.py
```

```
============================================================
  Setup complete! Choose a mode:
============================================================
  1) Demo Mode       (preset searches + evaluation)
  2) Interactive CLI  (free-form search & recommend)
============================================================
```

### API Signatures

```python
# Search
search(query: str, domain: str, indexes: dict, top_k: int = 5)
# → list[dict]  keys: rank, id, title, domain, similarity_score

# Recommend
recommend(item_id: int, domain: str, indexes: dict, top_k: int = 5)
# → list[dict]  keys: rank, id, title, domain, similarity_score
```

---

## 8. Benchmark: Semantic vs TF-IDF

| Dimension | TF-IDF / BM25 | This System (Dense Retrieval) |
|---|---|---|
| **Representation** | Sparse bag-of-words (high-dim) | Dense vector — 384 floats |
| **Vocabulary match required** | Yes — exact token overlap | No — meaning match |
| **Synonym handling** | ❌ Blind | ✅ Native |
| **Cross-lingual** | ❌ Not supported | ✅ Partial (via multilingual variant) |
| **Index size** | Small (inverted index) | 7 MB for 4,844 items |
| **Query time** | < 1 ms | ~30–80 ms (encoding bottleneck) |
| **Query: "AI role"** matches "Machine Learning Engineer" | ❌ 0 overlap | ✅ score ≈ 0.78 |
| **Query: "learn python"** matches "Programming Fundamentals" | ❌ 0 overlap | ✅ score ≈ 0.72 |
| **Cold start (new document)** | Re-index term frequencies | Re-embed + `index.add()` |
| **Infrastructure** | In-memory dict | FAISS flat index on disk |

**Conclusion**: Dense retrieval dominates in recall for paraphrase/synonym queries at the cost of ~30–80 ms encoding latency per query, vs < 1 ms for TF-IDF. For this corpus size, that trade-off is acceptable for all interactive use cases.

---

## 9. Project Structure

```
D:\Search engine\
├── main.py                  # Entrypoint (CLI + Demo)
├── audit.py                 # 7-phase integrity audit (17 checks)
├── requirements.txt         # pip dependencies
├── src/
│   ├── preprocessing.py     # Phase 1: column detection, text cleaning, aggregation
│   ├── embeddings.py        # Phase 2: all-MiniLM-L6-v2, L2 normalization
│   ├── index_builder.py     # Phase 3: FAISS IndexFlatIP, disk persistence
│   ├── search_engine_core.py# Phases 4+5: search() + recommend()
│   └── evaluation.py        # Phase 6: 80/20 split, Top-1/3, latency
├── processed_data/
│   ├── jobs_clean.csv       # 1,041 rows | id, title, text, domain
│   ├── courses_clean.csv    # 3,672 rows | id, title, text, domain
│   └── products_clean.csv   # 131 rows  | id, title, text, domain
└── indexes/
    ├── jobs.index            # FAISS index (1,041 × 384 float32)
    ├── jobs_meta.pkl         # Metadata dict
    ├── jobs_embeddings.npy   # Raw embeddings (1,041 × 384)
    └── ... (same for courses, products)
```

---

## 10. Installation & Usage

### Requirements

```
pandas
numpy
sentence-transformers
faiss-cpu
scikit-learn
tqdm
```

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py    # full pipeline: preprocess → index → search/recommend → evaluate
python audit.py   # 7-phase integrity audit: leakage, normalization, self-match, etc.
```

### First Run vs Subsequent Runs

| Phase | First Run | Subsequent Runs |
|---|---|---|
| Preprocessing | Reads raw CSVs, creates `processed_data/*.csv` | Loads existing CSVs (skips) |
| Indexing | Encodes all texts, creates `indexes/*` | Loads existing indexes (skips) |
| Search / Recommend | Always live | Always live |

---

*CPU-only. No GPU. No external APIs. Fully local.*
