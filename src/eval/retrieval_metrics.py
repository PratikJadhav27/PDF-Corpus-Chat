"""
src/eval/retrieval_metrics.py

Pure-Python retrieval quality metrics.  No external dependencies.

Metrics
-------
recall_at_k       — fraction of relevant docs found in top-K results
precision_at_k    — fraction of top-K results that are relevant
mrr               — Mean Reciprocal Rank (rank of first relevant hit)
ndcg_at_k         — Normalised Discounted Cumulative Gain
latency_stats     — P50 / P95 / mean from a list of latency samples

All functions follow a consistent signature:
    retrieved : list[str]   — ordered list of doc identifiers returned
    relevant  : set[str]    — set of ground-truth relevant identifiers
    k         : int         — cut-off depth

A "doc identifier" is whatever string uniquely identifies a chunk source.
For this project we use the PDF filename (e.g. "attention_is_all_you_need.pdf").
"""

import math
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Fraction of *relevant* documents that appear in the top-K retrieved.

    recall@K = |relevant ∩ retrieved[:K]| / |relevant|

    Returns 0.0 when |relevant| == 0 to avoid division by zero.
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Fraction of top-K retrieved documents that are relevant.

    precision@K = |relevant ∩ retrieved[:K]| / K

    Returns 0.0 when k == 0.
    """
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Reciprocal rank of the first relevant document in the ranked list.

    MRR = 1 / rank_of_first_hit   (0.0 if no hit)
    """
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at depth K.

    Binary relevance: gain = 1 if doc is relevant, else 0.
    DCG  = Σ gain_i / log2(i+1)  for i in 1..K
    IDCG = DCG of ideal (perfect) ranking
    nDCG = DCG / IDCG
    """
    def _dcg(hits: List[int]) -> float:
        return sum(h / math.log2(i + 2) for i, h in enumerate(hits))

    gains = [1 if r in relevant else 0 for r in retrieved[:k]]
    ideal = sorted(gains, reverse=True)

    dcg_score = _dcg(gains)
    idcg_score = _dcg(ideal)

    return dcg_score / idcg_score if idcg_score > 0 else 0.0


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Average Precision (AP) — area under the precision-recall curve.

    AP = Σ precision@k × rel(k)  /  |relevant|
    where rel(k) = 1 if retrieved[k] is relevant.

    Returns 0.0 if no relevant docs or retrieved is empty.
    """
    if not relevant or not retrieved:
        return 0.0
    hits = 0
    score = 0.0
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            score += hits / rank
    return score / len(relevant)


# ---------------------------------------------------------------------------
# Aggregate over a run
# ---------------------------------------------------------------------------

def compute_run_metrics(
    run: List[Dict],
    k: int = 5,
) -> Dict[str, float]:
    """
    Aggregate per-query metrics across a full benchmark run.

    Args:
        run: list of dicts, each containing:
            - "retrieved" : List[str]  — ordered retrieved source filenames
            - "relevant"  : List[str]  — ground-truth relevant source filenames
            - "latency_sec": float     — wall-clock retrieval latency
        k: cut-off for Recall@K, Precision@K, nDCG@K

    Returns a dict with keys:
        recall_at_k, precision_at_k, mrr, ndcg_at_k, map,
        p50_latency_sec, p95_latency_sec, mean_latency_sec, n_queries
    """
    recalls, precisions, mrrs, ndcgs, aps, latencies = [], [], [], [], [], []

    for row in run:
        retrieved = row.get("retrieved", [])
        relevant = set(row.get("relevant", []))
        lat = row.get("latency_sec", 0.0)

        recalls.append(recall_at_k(retrieved, relevant, k))
        precisions.append(precision_at_k(retrieved, relevant, k))
        mrrs.append(mrr(retrieved, relevant))
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))
        aps.append(average_precision(retrieved, relevant))
        latencies.append(lat)

    def _mean(lst: List[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    def _percentile(lst: List[float], pct: float) -> float:
        if not lst:
            return 0.0
        sorted_lst = sorted(lst)
        idx = int(math.ceil(pct / 100.0 * len(sorted_lst))) - 1
        return round(sorted_lst[max(0, idx)], 4)

    return {
        f"recall_at_{k}": _mean(recalls),
        f"precision_at_{k}": _mean(precisions),
        "mrr": _mean(mrrs),
        f"ndcg_at_{k}": _mean(ndcgs),
        "map": _mean(aps),
        "p50_latency_sec": _percentile(latencies, 50),
        "p95_latency_sec": _percentile(latencies, 95),
        "mean_latency_sec": _mean(latencies),
        "n_queries": len(run),
    }
