#!/usr/bin/env python
"""
scripts/run_eval.py

CLI runner for deterministic batch evaluation against a golden dataset.

Usage:
    python scripts/run_eval.py \
        --collection my_collection \
        --mode hybrid \
        --top-k 4 \
        --dataset evals/golden_dataset.jsonl \
        --output evals/reports/latest.json \
        --faithfulness-threshold 0.85 \
        --citation-validity-threshold 0.90

Exit codes:
    0 — all metrics meet thresholds
    1 — one or more metrics fell below threshold (CI gate fail)
    2 — configuration / setup error
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path so src.* imports resolve
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

# Late imports after path fix
from langchain_google_genai import ChatGoogleGenerativeAI
from src.rag.vector_store import load_vector_store, get_embeddings
from src.rag.retriever import retrieve_docs, retrieve_hybrid_reranked
from src.rag.prompting import build_context_with_citations, build_grounded_prompt
from src.rag.validators import guard_answer_with_citation_check, citation_validity
from src.rag.prompt_registry import DEFAULT_PROMPT_VERSION
from src.eval.evaluator import evaluate_one, build_report

LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "gemini-2.5-flash")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_golden_dataset(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping invalid JSON on line {line_no}: {exc}", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# Single inference call
# ---------------------------------------------------------------------------

def run_inference(
    db,
    collection_name: str,
    question: str,
    mode: str,
    top_k: int,
    candidate_k: int,
    prompt_version: str,
    temperature: float,
) -> tuple[str, list, float]:
    """Run retrieval + generation for one question. Returns (answer, docs, latency_sec)."""
    t0 = time.time()

    if mode == "hybrid":
        docs = retrieve_hybrid_reranked(
            db=db,
            collection_name=collection_name,
            query=question,
            top_k=top_k,
            candidate_k=candidate_k,
        )
    else:
        docs = retrieve_docs(db, question, mode=mode, k=top_k)

    if not docs:
        return "I don't know based on the provided documents.", [], round(time.time() - t0, 3)

    context = build_context_with_citations(docs)
    prompt = build_grounded_prompt(context=context, question=question, version=prompt_version)

    model = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=temperature)
    answer_raw = model.invoke(prompt).content
    answer = guard_answer_with_citation_check(answer_raw, docs, min_validity_ratio=0.8)

    return answer, docs, round(time.time() - t0, 3)


# ---------------------------------------------------------------------------
# Threshold checks
# ---------------------------------------------------------------------------

def check_thresholds(
    summary: Dict[str, Any],
    faithfulness_threshold: float,
    citation_validity_threshold: float,
    composite_score_threshold: float,
) -> List[str]:
    """Return a list of violation messages; empty list means all pass."""
    violations = []

    avg_faith = summary.get("avg_faithfulness")
    if avg_faith is not None and avg_faith < faithfulness_threshold:
        violations.append(
            f"avg_faithfulness={avg_faith:.4f} < threshold={faithfulness_threshold}"
        )

    avg_cit = summary.get("avg_citation_validity_ratio", 0.0)
    if avg_cit < citation_validity_threshold:
        violations.append(
            f"avg_citation_validity_ratio={avg_cit:.4f} < threshold={citation_validity_threshold}"
        )

    avg_comp = summary.get("avg_composite_score")
    if avg_comp is not None and avg_comp < composite_score_threshold:
        violations.append(
            f"avg_composite_score={avg_comp:.4f} < threshold={composite_score_threshold}"
        )

    return violations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run golden-dataset evaluation against a collection."
    )
    p.add_argument("--collection", required=True, help="Vector store collection name.")
    p.add_argument(
        "--mode",
        default="hybrid",
        choices=["similarity", "mmr", "hybrid"],
        help="Retrieval mode.",
    )
    p.add_argument("--top-k", type=int, default=4, help="Final top-K chunks.")
    p.add_argument(
        "--candidate-k",
        type=int,
        default=12,
        help="Candidate K for hybrid retrieval before rerank.",
    )
    p.add_argument(
        "--dataset",
        default="evals/golden_dataset.jsonl.template",
        help="Path to golden Q&A dataset.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output path for the JSON report. Defaults to evals/reports/<timestamp>.json.",
    )
    p.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        help="Prompt version to use.",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--llm-metrics",
        action="store_true",
        default=True,
        help="Enable LLM-as-judge metrics (faithfulness, context relevance, completeness).",
    )
    p.add_argument(
        "--no-llm-metrics",
        dest="llm_metrics",
        action="store_false",
        help="Skip LLM-judge metrics (faster, heuristic only).",
    )
    p.add_argument(
        "--faithfulness-threshold",
        type=float,
        default=0.85,
        help="Minimum acceptable avg faithfulness score.",
    )
    p.add_argument(
        "--citation-validity-threshold",
        type=float,
        default=0.90,
        help="Minimum acceptable avg citation validity ratio.",
    )
    p.add_argument(
        "--composite-score-threshold",
        type=float,
        default=0.85,
        help="Minimum acceptable avg composite score.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # --- Load dataset ---
    if not os.path.exists(args.dataset):
        print(f"[ERROR] Dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    records = load_golden_dataset(args.dataset)
    if not records:
        print("[ERROR] Golden dataset is empty.", file=sys.stderr)
        return 2

    print(f"[INFO] Loaded {len(records)} golden Q&A records from '{args.dataset}'")

    # --- Load vector store ---
    embeddings = get_embeddings()
    db = load_vector_store(args.collection, embeddings)
    if db is None:
        print(
            f"[ERROR] Collection '{args.collection}' not found. "
            "Please ingest PDFs into this collection first.",
            file=sys.stderr,
        )
        return 2

    print(
        f"[INFO] Loaded collection '{args.collection}' | mode={args.mode} | top_k={args.top_k}"
    )

    # --- Run inference + evaluate ---
    results = []
    for idx, record in enumerate(records, 1):
        question = record.get("question", "")
        if not question:
            print(f"[WARN] Skipping record {idx}: missing 'question' field.", file=sys.stderr)
            continue

        print(f"  [{idx}/{len(records)}] Q: {question[:80]}...")

        try:
            answer, docs, latency = run_inference(
                db=db,
                collection_name=args.collection,
                question=question,
                mode=args.mode,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                prompt_version=args.prompt_version,
                temperature=args.temperature,
            )
        except Exception as exc:
            print(f"  [ERROR] Inference failed: {exc}", file=sys.stderr)
            answer = ""
            docs = []
            latency = 0.0

        row = evaluate_one(
            question=question,
            answer=answer,
            docs=docs,
            include_llm_metrics=args.llm_metrics,
        )
        row["latency_sec"] = latency
        row["expected_answer"] = record.get("expected_answer", "")
        row["expected_sources"] = record.get("expected_sources", [])
        results.append(row)

    # --- Build report ---
    report = build_report(
        collection_name=args.collection,
        retrieval_mode=args.mode,
        top_k=args.top_k,
        results=results,
    )
    report["prompt_version"] = args.prompt_version
    report["llm_metrics_enabled"] = args.llm_metrics

    # --- Print summary ---
    summary = report["summary"]
    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"  Total Questions       : {summary['total_questions']}")
    print(f"  Citation Rate         : {summary['citation_rate']}%")
    print(f"  IDK Rate              : {summary['idk_rate']}%")
    print(f"  Avg Latency           : {summary['avg_latency_sec']}s")
    print(f"  Avg Citation Validity : {summary['avg_citation_validity_ratio']}")
    if args.llm_metrics:
        print(f"  Avg Faithfulness      : {summary.get('avg_faithfulness')}")
        print(f"  Avg Context Relevance : {summary.get('avg_context_relevance')}")
        print(f"  Avg Completeness      : {summary.get('avg_answer_completeness')}")
        print(f"  Avg Composite Score   : {summary.get('avg_composite_score')}")
    print("=" * 60)

    # --- Write report to disk ---
    output_path = args.output
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"evals/reports/{ts}_{args.collection}_{args.mode}.json"

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[INFO] Report saved to: {output_path}")

    # --- Threshold gate ---
    violations = check_thresholds(
        summary,
        faithfulness_threshold=args.faithfulness_threshold,
        citation_validity_threshold=args.citation_validity_threshold,
        composite_score_threshold=args.composite_score_threshold,
    )

    if violations:
        print("\n[FAIL] Metric thresholds not met:", file=sys.stderr)
        for v in violations:
            print(f"  ✗ {v}", file=sys.stderr)
        return 1

    print("\n[PASS] All metric thresholds met. ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
