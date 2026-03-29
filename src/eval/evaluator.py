import time
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.rag.validators import citation_validity


CITATION_PATTERN = re.compile(r"\[[^\]]+\|\s*page\s*\d+\]", re.IGNORECASE)


def has_citations(answer: str) -> bool:
    if not answer:
        return False
    return bool(CITATION_PATTERN.search(answer))


def is_idk_answer(answer: str) -> bool:
    if not answer:
        return True
    return "i don't know based on the provided documents" in answer.lower()


def evaluate_one(
    question: str,
    answer: str,
    docs,
    include_llm_metrics: bool = False,
) -> Dict[str, Any]:
    """
    Heuristic evaluation (always runs):
    - checks citations presence
    - checks IDK fallback
    - stores retrieved sources
    - citation validity ratio

    Optional LLM-judge metrics (set include_llm_metrics=True for CLI/CI runs):
    - faithfulness
    - context_relevance
    - answer_completeness
    - composite_score
    """
    sources = []
    for d in docs or []:
        sources.append(
            {
                "source": d.metadata.get("source", "Unknown"),
                "page": d.metadata.get("page_number", "?"),
                "chunk_index": d.metadata.get("chunk_index", "?"),
            }
        )

    validity = citation_validity(answer, docs or [])

    result: Dict[str, Any] = {
        "question": question,
        "answer": answer,
        "has_citations": has_citations(answer),
        "is_idk": is_idk_answer(answer),
        "citation_validity_ratio": validity.get("validity_ratio", 0.0),
        "citation_count": validity.get("citation_count", 0),
        "valid_citation_count": validity.get("valid_citation_count", 0),
        "sources": sources,
        "llm_metrics": None,
    }

    if include_llm_metrics and not is_idk_answer(answer) and docs:
        try:
            from src.eval.metrics import evaluate_all_metrics
            llm_scores = evaluate_all_metrics(question=question, answer=answer, docs=docs)
            result["llm_metrics"] = llm_scores
        except Exception as exc:
            result["llm_metrics"] = {"error": str(exc)}

    return result


def summarize_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "total_questions": 0,
            "citation_rate": 0.0,
            "idk_rate": 0.0,
            "avg_latency_sec": 0.0,
            "avg_citation_validity_ratio": 0.0,
            "avg_faithfulness": None,
            "avg_context_relevance": None,
            "avg_answer_completeness": None,
            "avg_composite_score": None,
        }

    total = len(results)
    citation_count = sum(1 for r in results if r.get("has_citations"))
    idk_count = sum(1 for r in results if r.get("is_idk"))
    latencies = [r.get("latency_sec", 0.0) for r in results]
    validity_scores = [r.get("citation_validity_ratio", 0.0) for r in results]

    # Aggregate LLM judge scores if present
    faithfulness_scores = []
    context_relevance_scores = []
    completeness_scores = []
    composite_scores = []

    for r in results:
        llm = r.get("llm_metrics")
        if llm and isinstance(llm, dict) and "error" not in llm:
            if "faithfulness" in llm:
                faithfulness_scores.append(llm["faithfulness"].get("score", 0.0))
            if "context_relevance" in llm:
                context_relevance_scores.append(llm["context_relevance"].get("score", 0.0))
            if "answer_completeness" in llm:
                completeness_scores.append(llm["answer_completeness"].get("score", 0.0))
            if "composite_score" in llm:
                composite_scores.append(llm["composite_score"])

    def _avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "total_questions": total,
        "citation_rate": round((citation_count / total) * 100, 2),
        "idk_rate": round((idk_count / total) * 100, 2),
        "avg_latency_sec": round(sum(latencies) / total, 3),
        "avg_citation_validity_ratio": round(sum(validity_scores) / total, 4),
        "avg_faithfulness": _avg(faithfulness_scores),
        "avg_context_relevance": _avg(context_relevance_scores),
        "avg_answer_completeness": _avg(completeness_scores),
        "avg_composite_score": _avg(composite_scores),
    }


def build_report(
    collection_name: str,
    retrieval_mode: str,
    top_k: int,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(),
        "collection_name": collection_name,
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "summary": summarize_report(results),
        "results": results,
    }
