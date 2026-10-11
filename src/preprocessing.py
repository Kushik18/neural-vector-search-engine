"""
Phase 1 — Data Standardization
===============================
Automatically detects CSV files, loads/concatenates them, applies strict
preprocessing rules, and outputs standardized clean CSVs.

Key design choices:
  - Robust column detection via heuristics (no hardcoded column names)
  - Products are grouped by (product_id) or (name + brand) to avoid
    review-level duplication
  - Text is lowercased and whitespace-normalized, but NOT stemmed,
    lemmatized, or aggressively stripped
"""

import os
import re
import glob
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Column Detection Heuristics
# ---------------------------------------------------------------------------

# Priority keywords for detecting title-like columns
TITLE_KEYWORDS = ["title", "name", "course_title", "job_title", "product_name"]

# Keywords for columns that carry meaningful text for embedding
TEXT_KEYWORDS = [
    "title", "name", "description", "text", "review", "summary",
    "subject", "category", "categories", "brand", "company",
    "location", "level", "detail", "overview", "body", "content",
]

# Columns that should never be treated as text
EXCLUDE_KEYWORDS = [
    "id", "url", "link", "date", "timestamp", "image", "source",
    "price", "rating", "num_", "is_", "asins", "keys", "manufacturer",
    "did", "helpful", "username", "usercity", "userprovince",
    "province", "city", "paid",
]


def _score_column(col_name: str, keywords: list[str]) -> int:
    """Score a column name against a list of keywords (case-insensitive)."""
    col_lower = col_name.lower().replace(".", "_")
    score = 0
    for kw in keywords:
        if kw in col_lower:
            score += 1
    return score


def _is_excluded(col_name: str) -> bool:
    col_lower = col_name.lower().replace(".", "_")
    for kw in EXCLUDE_KEYWORDS:
        if kw in col_lower:
            return True
    return False


def detect_title_column(columns: list[str]) -> str:
    """Pick the best title column from a list of column names."""
    scored = [(c, _score_column(c, TITLE_KEYWORDS)) for c in columns]
    scored.sort(key=lambda x: x[1], reverse=True)
    if scored and scored[0][1] > 0:
        return scored[0][0]
    # Fallback: first text-ish column
    for c in columns:
        if not _is_excluded(c):
            return c
    return columns[0]


def detect_text_columns(columns: list[str], title_col: str) -> list[str]:
    """Pick all meaningful text columns (including title) for embedding."""
    text_cols = []
    for c in columns:
        if _is_excluded(c):
            continue
        if _score_column(c, TEXT_KEYWORDS) > 0:
            text_cols.append(c)
    # Always include the title column
    if title_col not in text_cols:
        text_cols.insert(0, title_col)
    return text_cols


# ---------------------------------------------------------------------------
# Text Cleaning  (strict: no stemming / lemmatization / stopword removal)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase, normalize whitespace.  Preserves numbers and punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Domain-specific Loaders
# ---------------------------------------------------------------------------

def load_csvs_from_folder(folder: str) -> pd.DataFrame:
    """Load and concatenate all CSV files from *folder*."""
    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    dfs = []
    for f in csv_files:
        print(f"  Loading {os.path.basename(f)} ...")
        df = pd.read_csv(f, encoding="utf-8", encoding_errors="replace",
                         low_memory=False)
        # Strip BOM from first column name if present
        first_col = df.columns[0]
        if first_col.startswith("\ufeff"):
            df.rename(columns={first_col: first_col.lstrip("\ufeff")}, inplace=True)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def _process_generic(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """
    Generic preprocessing: detect columns, combine text, clean, filter.
    Returns a DataFrame with columns [id, title, text, domain].
    """
    cols = list(df.columns)
    title_col = detect_title_column(cols)
    text_cols = detect_text_columns(cols, title_col)

    print(f"  Detected title column : {title_col}")
    print(f"  Detected text columns : {text_cols}")

    # Fill missing values
    df = df.fillna("")

    # Combine text columns
    df["_combined_text"] = df[text_cols].astype(str).agg(" ".join, axis=1)
    df["_combined_text"] = df["_combined_text"].apply(clean_text)

    # Build standardized output
    out = pd.DataFrame()
    out["title"] = df[title_col].astype(str).apply(clean_text)
    out["text"] = df["_combined_text"]
    out["domain"] = domain

    # Remove short text and duplicates
    out = out[out["text"].str.len() >= 20].copy()
    out.drop_duplicates(subset=["text"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    out.insert(0, "id", range(len(out)))

    return out


def process_products(folder: str) -> pd.DataFrame:
    """
    Products require special handling:
      - Each CSV row is a *review*, not a product.
      - Group by product_id (if available) else by (name + brand).
      - Aggregate review text into a single product-level entry.
    """
    df = load_csvs_from_folder(folder)
    original_count = len(df)
    print(f"  Original review-level row count: {original_count}")

    df = df.fillna("")

    # Determine grouping key
    if "product_id" in df.columns and df["product_id"].astype(str).str.strip().ne("").sum() > len(df) * 0.5:
        group_col = "product_id"
        print("  Grouping products by: product_id")
    else:
        # Create a composite key from name + brand
        df["_group_key"] = (df["name"].astype(str).str.strip().str.lower()
                            + " ||| "
                            + df.get("brand", pd.Series([""] * len(df))).astype(str).str.strip().str.lower())
        group_col = "_group_key"
        print("  Grouping products by: name + brand")

    # Detect columns for title and extra text
    cols = list(df.columns)
    title_col = detect_title_column(cols)

    # Aggregate per product
    agg_dict = {
        title_col: "first",
    }
    # Aggregate review text and review title
    review_text_col = None
    review_title_col = None
    for c in cols:
        cl = c.lower()
        if "reviews.text" in cl or "review_text" in cl:
            review_text_col = c
        if "reviews.title" in cl or "review_title" in cl:
            review_title_col = c

    # Add brand/categories if present
    extra_cols = []
    for c in ["brand", "categories", "primaryCategories"]:
        if c in cols:
            agg_dict[c] = "first"
            extra_cols.append(c)

    if review_text_col:
        agg_dict[review_text_col] = lambda x: " ".join(x.astype(str).head(10))  # cap at 10 reviews
    if review_title_col:
        agg_dict[review_title_col] = lambda x: " ".join(x.astype(str).head(10))

    grouped = df.groupby(group_col, sort=False).agg(agg_dict).reset_index(drop=True)
    print(f"  Unique products after grouping: {len(grouped)}")

    # Combine all text columns for embedding
    text_parts = [grouped[title_col].astype(str)]
    for c in extra_cols:
        text_parts.append(grouped[c].astype(str))
    if review_title_col and review_title_col in grouped.columns:
        text_parts.append(grouped[review_title_col].astype(str))
    if review_text_col and review_text_col in grouped.columns:
        text_parts.append(grouped[review_text_col].astype(str))

    combined_text = pd.DataFrame()
    combined_text["title"] = grouped[title_col].astype(str).apply(clean_text)
    combined_text["text"] = (
        pd.concat(text_parts, axis=1)
        .astype(str)
        .agg(" ".join, axis=1)
        .apply(clean_text)
    )
    combined_text["domain"] = "products"

    # Filter short & duplicate text
    combined_text = combined_text[combined_text["text"].str.len() >= 20].copy()
    combined_text.drop_duplicates(subset=["text"], inplace=True)
    combined_text.reset_index(drop=True, inplace=True)
    combined_text.insert(0, "id", range(len(combined_text)))

    print(f"  Final unique product count: {len(combined_text)}")
    return combined_text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DOMAINS = {
    "jobs":     {"folder_name": "Jobs",     "output": "jobs_clean.csv"},
    "courses":  {"folder_name": "Courses",  "output": "courses_clean.csv"},
    "products": {"folder_name": "Products", "output": "products_clean.csv"},
}


def preprocess_all(root_dir: str) -> dict[str, pd.DataFrame]:
    """
    Run full preprocessing pipeline for all domains.
    Returns dict mapping domain → cleaned DataFrame.
    """
    os.makedirs(os.path.join(root_dir, "processed_data"), exist_ok=True)
    results = {}

    for domain, cfg in DOMAINS.items():
        folder = os.path.join(root_dir, cfg["folder_name"])
        print(f"\n{'='*60}")
        print(f"Processing domain: {domain.upper()}")
        print(f"{'='*60}")

        if domain == "products":
            df_clean = process_products(folder)
        else:
            df_raw = load_csvs_from_folder(folder)
            original_count = len(df_raw)
            print(f"  Original row count: {original_count}")
            df_clean = _process_generic(df_raw, domain)

        print(f"  Cleaned row count: {len(df_clean)}")

        # Save
        out_path = os.path.join(root_dir, "processed_data", cfg["output"])
        df_clean.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path}")

        results[domain] = df_clean

    return results


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preprocess_all(root)
