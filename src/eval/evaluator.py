import time
import re
from typing import Dict, Any, List
from datetime import datetime


CITATION_PATTERN = re.compile(r"\[[^\]]+\|\s*page\s*\d+\]", re.IGNORECASE)


def has_citations(answer: str) -> bool:
    if not answer:
        return False
    return bool(CITATION_PATTERN.search(answer))


def is_idk_answer(answer: str) -> bool:
    if not answer:
        return True
    return "i don't know based on the provided documents" in answer.lower()


def evaluate_one(question: str, answer: str, docs) -> Dict[str, Any]:
    """
    Heuristic evaluation:
    - checks citations presence
    - checks IDK fallback
    - stores retrieved sources
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

    return {
        "question": question,
        "answer": answer,
        "has_citations": has_citations(answer),
        "is_idk": is_idk_answer(answer),
        "sources": sources,
    }


def summarize_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "total_questions": 0,
            "citation_rate": 0.0,
            "idk_rate": 0.0,
            "avg_latency_sec": 0.0,
        }

    total = len(results)
    citation_count = sum(1 for r in results if r.get("has_citations"))
    idk_count = sum(1 for r in results if r.get("is_idk"))
    latencies = [r.get("latency_sec", 0.0) for r in results]

    return {
        "total_questions": total,
        "citation_rate": round((citation_count / total) * 100, 2),
        "idk_rate": round((idk_count / total) * 100, 2),
        "avg_latency_sec": round(sum(latencies) / total, 3),
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
