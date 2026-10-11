"""
Multi-Domain Semantic Search & Recommendation Engine
=====================================================
Root-level entrypoint.

Usage:
    python main.py

Modes:
    1) Demo Mode   -- runs preset searches, recommendations, and evaluation
    2) Interactive  -- free-form search & recommend via CLI
"""

import os
import sys
import pandas as pd

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on the path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import preprocess_all
from src.index_builder import build_all_indexes
from src.search_engine_core import search, recommend, print_results
from src.evaluation import evaluate_all


# ======================================================================
# Data helpers
# ======================================================================

def load_processed_data(root: str) -> dict[str, pd.DataFrame]:
    """Load already-processed CSVs if they exist."""
    mapping = {
        "jobs":     "jobs_clean.csv",
        "courses":  "courses_clean.csv",
        "products": "products_clean.csv",
    }
    data = {}
    for domain, fname in mapping.items():
        path = os.path.join(root, "processed_data", fname)
        if os.path.exists(path):
            data[domain] = pd.read_csv(path)
    return data


def setup_pipeline() -> tuple:
    """Run preprocessing + indexing (or load cached). Returns (data, indexes)."""
    print("=" * 60)
    print("  Multi-Domain Semantic Search & Recommendation Engine")
    print("=" * 60)

    # Phase 1 -- Preprocessing
    print("\n>> PHASE 1: Data Standardization")
    processed_dir = os.path.join(ROOT_DIR, "processed_data")
    if (os.path.exists(os.path.join(processed_dir, "jobs_clean.csv"))
            and os.path.exists(os.path.join(processed_dir, "courses_clean.csv"))
            and os.path.exists(os.path.join(processed_dir, "products_clean.csv"))):
        print("  Processed data already exists. Loading ...")
        data = load_processed_data(ROOT_DIR)
        for domain, df in data.items():
            print(f"  [{domain}] {len(df)} rows")
    else:
        data = preprocess_all(ROOT_DIR)

    # Phase 2 & 3 -- Embedding + Indexing
    print("\n>> PHASE 2 & 3: Embedding & Indexing")
    indexes = build_all_indexes(data, ROOT_DIR, batch_size=64)

    return data, indexes


# ======================================================================
# Demo Mode  (original pipeline behaviour)
# ======================================================================

def run_demo(data: dict, indexes: dict) -> None:
    """Run preset demo searches, recommendations, and evaluation."""

    # Demo Search
    print("\n>> PHASE 4: Demo Search")
    demo_queries = {
        "jobs": "data scientist machine learning",
        "courses": "python programming for beginners",
        "products": "wireless bluetooth headphones",
    }
    for domain, query in demo_queries.items():
        results = search(query, domain, indexes, top_k=5)
        print_results(results, header=f"Search: '{query}' in {domain.upper()}")

    # Demo Recommendation
    print("\n>> PHASE 5: Demo Recommendation")
    for domain in ["jobs", "courses", "products"]:
        try:
            results = recommend(item_id=0, domain=domain,
                                indexes=indexes, top_k=5)
            item_title = data[domain].loc[
                data[domain]["id"] == 0, "title"
            ].values[0]
            print_results(
                results,
                header=f"Items similar to '{item_title[:50]}' ({domain})",
            )
        except Exception as e:
            print(f"  [{domain}] Recommendation error: {e}")

    # Evaluation
    print("\n>> PHASE 6: Evaluation (Self-Retrieval)")
    evaluate_all(data, batch_size=64)

    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60)


# ======================================================================
# Interactive CLI
# ======================================================================

def _ask(prompt: str, default: str = "") -> str:
    """Prompt user for input, returning stripped string."""
    try:
        val = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return "exit"
    return val if val else default


def _ask_domain(domains: list[str]) -> str:
    """Prompt for a valid domain name. Returns '' on exit."""
    domain_list = ", ".join(domains)
    while True:
        choice = _ask(f"    Domain ({domain_list}): ")
        if choice.lower() in ("exit", "quit", "q"):
            return ""
        if choice.lower() in domains:
            return choice.lower()
        print(f"    Invalid domain. Choose from: {domain_list}")


def _ask_int(prompt: str, default: int) -> int | None:
    """Prompt for an integer. Returns None on exit."""
    val = _ask(prompt, str(default))
    if val.lower() in ("exit", "quit", "q"):
        return None
    try:
        return int(val)
    except ValueError:
        print(f"    Not a valid number, using default ({default}).")
        return default


def run_cli(data: dict, indexes: dict) -> None:
    """Interactive CLI loop for search and recommendations."""

    domains = list(indexes.keys())

    print("\n" + "=" * 60)
    print("  Multi-Domain Semantic Search Engine (Interactive Mode)")
    print("=" * 60)
    print(f"  Available domains: {', '.join(domains)}")
    print("  Type 'exit' at any prompt to return to menu.\n")

    while True:
        print("-" * 60)
        print("  1) Search")
        print("  2) Recommend by Item ID")
        print("  3) Exit")
        print("-" * 60)
        choice = _ask("  Select option [1/2/3]: ", "1")

        # ----------------------------------------------------------
        # EXIT
        # ----------------------------------------------------------
        if choice in ("3", "exit", "quit", "q"):
            print("\n  Goodbye!\n")
            break

        # ----------------------------------------------------------
        # SEARCH
        # ----------------------------------------------------------
        elif choice == "1":
            print("\n  -- Search --")
            domain = _ask_domain(domains)
            if not domain:
                continue

            query = _ask("    Query: ")
            if not query or query.lower() in ("exit", "quit", "q"):
                continue
            if len(query.strip()) == 0:
                print("    Query cannot be empty.")
                continue

            top_k = _ask_int("    Top K (default 5): ", 5)
            if top_k is None:
                continue

            try:
                results = search(query, domain, indexes, top_k=top_k)
                if results:
                    print_results(results,
                                  header=f"Search: '{query}' in {domain.upper()}")
                else:
                    print("    No results found.")
            except Exception as e:
                print(f"    Error: {e}")

        # ----------------------------------------------------------
        # RECOMMEND
        # ----------------------------------------------------------
        elif choice == "2":
            print("\n  -- Recommend by Item ID --")
            domain = _ask_domain(domains)
            if not domain:
                continue

            item_id = _ask_int("    Item ID: ", 0)
            if item_id is None:
                continue

            # Validate that item_id exists
            _, meta, _ = indexes[domain]
            valid_ids = {v["id"] for v in meta.values()}
            if item_id not in valid_ids:
                print(f"    Item ID {item_id} not found in '{domain}'.")
                print(f"    Valid ID range: {min(valid_ids)} - {max(valid_ids)}")
                continue

            top_k = _ask_int("    Top K (default 5): ", 5)
            if top_k is None:
                continue

            try:
                # Show the source item title
                source_title = data[domain].loc[
                    data[domain]["id"] == item_id, "title"
                ].values[0]
                results = recommend(item_id, domain, indexes, top_k=top_k)
                if results:
                    print_results(
                        results,
                        header=f"Items similar to '{source_title[:50]}' ({domain})",
                    )
                else:
                    print("    No recommendations found.")
            except Exception as e:
                print(f"    Error: {e}")

        else:
            print("    Invalid option. Please choose 1, 2, or 3.")


# ======================================================================
# Entrypoint
# ======================================================================

def main():
    data, indexes = setup_pipeline()

    print("\n" + "=" * 60)
    print("  Setup complete! Choose a mode:")
    print("=" * 60)
    print("  1) Demo Mode       (preset searches + evaluation)")
    print("  2) Interactive CLI  (free-form search & recommend)")
    print("=" * 60)

    choice = _ask("  Select mode [1/2]: ", "2")

    if choice == "1":
        run_demo(data, indexes)
    else:
        run_cli(data, indexes)


if __name__ == "__main__":
    main()
