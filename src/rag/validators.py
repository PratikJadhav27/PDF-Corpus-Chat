import re
from typing import Dict, List, Set, Tuple
from langchain_core.documents import Document


CITATION_PATTERN = re.compile(r"\[([^\]]+?)\|\s*page\s*([0-9]+)\]", re.IGNORECASE)
FALLBACK = "I don't know based on the provided documents."


def build_allowed_citations(docs: List[Document]) -> Set[Tuple[str, str]]:
    allowed: Set[Tuple[str, str]] = set()
    for d in docs:
        source = str(d.metadata.get("source", "Unknown")).strip()
        page = str(d.metadata.get("page_number", "?")).strip()
        allowed.add((source, page))
    return allowed


def parse_citations(answer: str) -> List[Tuple[str, str]]:
    matches = CITATION_PATTERN.findall(answer or "")
    return [(src.strip(), page.strip()) for src, page in matches]


def citation_validity(answer: str, docs: List[Document]) -> Dict[str, float]:
    found = parse_citations(answer)
    if not found:
        return {"citation_count": 0, "valid_citation_count": 0, "validity_ratio": 0.0}

    allowed = build_allowed_citations(docs)
    valid = sum(1 for item in found if item in allowed)
    ratio = valid / len(found)
    return {
        "citation_count": len(found),
        "valid_citation_count": valid,
        "validity_ratio": round(ratio, 4),
    }


def guard_answer_with_citation_check(
    answer: str,
    docs: List[Document],
    min_validity_ratio: float = 0.8,
) -> str:
    stats = citation_validity(answer, docs)
    if stats["citation_count"] == 0:
        return FALLBACK
    if stats["validity_ratio"] < min_validity_ratio:
        return FALLBACK
    return answer
