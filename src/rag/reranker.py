import re
from typing import Any, Dict, List


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def _overlap_score(query: str, text: str) -> float:
    q = set(_tokenize(query))
    t = set(_tokenize(text))
    if not q or not t:
        return 0.0
    return len(q.intersection(t)) / len(q)


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = 4,
) -> List[Dict[str, Any]]:
    """
    Lightweight reranker:
    - computes lexical overlap score
    - combines with RRF score for stable ranking
    """
    scored: List[Dict[str, Any]] = []
    for c in candidates:
        text = c.get("text", "")
        overlap = _overlap_score(query, text)
        rrf = float(c.get("rrf_score", 0.0))
        combined = (0.7 * overlap) + (0.3 * rrf)

        row = dict(c)
        row["rerank_score"] = round(combined, 6)
        row["overlap_score"] = round(overlap, 6)
        scored.append(row)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_n]
