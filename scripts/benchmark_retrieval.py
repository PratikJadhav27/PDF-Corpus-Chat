#!/usr/bin/env python
"""
scripts/benchmark_retrieval.py

Benchmark five retrieval strategies against the golden dataset and produce:
  - A machine-readable baseline JSON  (evals/baselines/latest.json)
  - A human-readable Markdown table   (printed to stdout)

Strategies benchmarked
----------------------
1. similarity      — dense-only cosine similarity (Qdrant)
2. mmr             — Maximal Marginal Relevance (diversity)
3. bm25            — sparse keyword search only
4. hybrid_norrf    — vector + BM25 concatenated, no fusion (sanity check)
5. hybrid          — vector + BM25 → RRF fusion → cross-encoder rerank

Usage (requires an ingested collection with the fixture PDFs)
-------------------------------------------------------------
    python scripts/benchmark_retrieval.py \\
        --collection ci_collection \\
        --dataset   evals/golden_dataset.jsonl \\
        --output    evals/baselines/latest.json \\
        --top-k     5

Exit codes:
    0  — benchmark completed successfully
    1  — Recall@5 below RECALL_GATE threshold (regression gate)
    2  — setup / config error

Environment variables required (same as run_eval.py):
    GOOGLE_API_KEY
    QDRANT_URL  (or local Qdrant at http://localhost:6333)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.rag.embeddings import get_embeddings
from src.rag.vector_store import load_vector_store
from src.rag.retriever import retrieve_docs, retrieve_hybrid_reranked
from src.rag.bm25_index import bm25_search_collection
from src.eval.retrieval_metrics import compute_run_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGIES = ["similarity", "mmr", "bm25", "hybrid"]
TOP_K_DEFAULT = 5
RECALL_GATE = 0.50   # Recall@5 must be >= this or exit code 1


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_golden_dataset(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Per-strategy retrieval
# ---------------------------------------------------------------------------

def _source_names(docs) -> List[str]:
    """Extract normalised source filenames from retrieved Document objects."""
    return [
        os.path.basename(d.metadata.get("source", "")).lower()
        for d in docs
    ]


def run_strategy(
    strategy: str,
    db,
    collection_name: str,
    question: str,
    top_k: int,
) -> Dict[str, Any]:
    """
    Run a single strategy for one question.
    Returns {"retrieved": [...], "latency_sec": float}.
    """
    t0 = time.perf_counter()

    if strategy == "bm25":
        # bm25_search_collection returns List[Tuple[Dict, float]]
        hits = bm25_search_collection(collection_name, question, k=top_k)
        retrieved = [
            os.path.basename(doc.get("metadata", {}).get("source", "")).lower()
            for doc, _score in hits
        ]
    elif strategy == "hybrid":
        docs = retrieve_hybrid_reranked(
            db=db,
            collection_name=collection_name,
            query=question,
            top_k=top_k,
            candidate_k=max(12, top_k * 3),
        )
        retrieved = _source_names(docs)
    else:
        # similarity or mmr
        docs = retrieve_docs(db, question, mode=strategy, k=top_k)
        retrieved = _source_names(docs)

    latency = round(time.perf_counter() - t0, 4)
    return {"retrieved": retrieved, "latency_sec": latency}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    collection_name: str,
    dataset: List[Dict[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    embeddings = get_embeddings()
    db = load_vector_store(collection_name, embeddings)

    if db is None:
        print(
            f"[ERROR] Collection '{collection_name}' not found. "
            "Run download_fixtures.py and ingest first.",
            file=sys.stderr,
        )
        sys.exit(2)

    results: Dict[str, List[Dict]] = {s: [] for s in STRATEGIES}

    total = len(dataset)
    for i, row in enumerate(dataset, 1):
        question = row["question"]
        relevant = set(
            os.path.basename(s).lower()
            for s in row.get("expected_sources", [])
        )

        print(f"  [{i:2d}/{total}] {question[:70]}", end="\r", flush=True)

        for strategy in STRATEGIES:
            hit = run_strategy(strategy, db, collection_name, question, top_k)
            hit["relevant"] = list(relevant)
            results[strategy].append(hit)

    print()  # clear \r line

    # Compute aggregate metrics per strategy
    strategy_metrics: Dict[str, Dict[str, float]] = {}
    for strategy in STRATEGIES:
        strategy_metrics[strategy] = compute_run_metrics(results[strategy], k=top_k)

    return {
        "generated_at": datetime.now().isoformat(),
        "collection": collection_name,
        "top_k": top_k,
        "n_queries": total,
        "strategies": strategy_metrics,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_markdown_table(benchmark: Dict[str, Any], top_k: int) -> None:
    strategies = benchmark["strategies"]
    k = top_k

    header = (
        f"| Strategy       | Recall@{k} | Precision@{k} | MRR    | nDCG@{k} | "
        f"P50 (s) | P95 (s) |"
    )
    sep = (
        f"|----------------|-----------|-------------|--------|---------|"
        f"---------|---------|"
    )

    print()
    print(f"## Retrieval Benchmark — top-{k} ({benchmark['n_queries']} queries)")
    print()
    print(header)
    print(sep)

    for strategy in STRATEGIES:
        m = strategies[strategy]
        print(
            f"| {strategy:<14} "
            f"| {m[f'recall_at_{k}']:.3f}     "
            f"| {m[f'precision_at_{k}']:.3f}       "
            f"| {m['mrr']:.3f}   "
            f"| {m[f'ndcg_at_{k}']:.3f}    "
            f"| {m['p50_latency_sec']:.3f}    "
            f"| {m['p95_latency_sec']:.3f}    |"
        )
    print()


def save_baseline(benchmark: Dict[str, Any], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)
    print(f"[saved] {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark retrieval strategies")
    p.add_argument("--collection", required=True, help="Qdrant collection name")
    p.add_argument(
        "--dataset",
        default="evals/golden_dataset.jsonl",
        help="Path to golden_dataset.jsonl",
    )
    p.add_argument(
        "--output",
        default="evals/baselines/latest.json",
        help="Where to save the baseline JSON",
    )
    p.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    p.add_argument(
        "--recall-gate",
        type=float,
        default=RECALL_GATE,
        help="Minimum Recall@K for the best strategy (exit code 1 if below)",
    )
    p.add_argument(
        "--strategy",
        choices=STRATEGIES + ["all"],
        default="all",
        help="Run a single strategy or all (default: all)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global STRATEGIES
    if args.strategy != "all":
        STRATEGIES = [args.strategy]

    dataset = load_golden_dataset(args.dataset)
    print(f"[benchmark] collection={args.collection}  queries={len(dataset)}  top_k={args.top_k}")
    print(f"[benchmark] strategies: {STRATEGIES}")

    benchmark = run_benchmark(
        collection_name=args.collection,
        dataset=dataset,
        top_k=args.top_k,
    )

    print_markdown_table(benchmark, args.top_k)
    save_baseline(benchmark, args.output)

    # Regression gate: best strategy must clear Recall@K
    k = args.top_k
    best_recall = max(
        benchmark["strategies"][s].get(f"recall_at_{k}", 0.0)
        for s in benchmark["strategies"]
    )
    gate = args.recall_gate
    if best_recall < gate:
        print(
            f"[GATE FAIL] Best Recall@{k} = {best_recall:.3f} < threshold {gate:.2f}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[GATE PASS] Best Recall@{k} = {best_recall:.3f} >= {gate:.2f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
