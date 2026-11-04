import re
from typing import Dict, List, Set, Tuple
from langchain_core.documents import Document


# A citation names one page or several: "page 2", "page 2, 3", "pages 2-3",
# "page 2 & 3". The model produces these even though the prompt asks for one
# page per citation, so the guard has to read them (ISSUE-017).
PAGE_SPEC = r"page[s]?\s*[0-9]+(?:\s*(?:,|&|and|-|–|—|to)\s*[0-9]+)*"
CITATION_PATTERN = re.compile(r"\[([^\]]+?)\|\s*(" + PAGE_SPEC + r")\]", re.IGNORECASE)
FALLBACK = "I don't know based on the provided documents."


def build_allowed_citations(docs: List[Document]) -> Set[Tuple[str, str]]:
    allowed: Set[Tuple[str, str]] = set()
    for d in docs:
        source = str(d.metadata.get("source", "Unknown")).strip()
        page = str(d.metadata.get("page_number", "?")).strip()
        allowed.add((source, page))
    return allowed


def parse_citations(answer: str) -> List[Tuple[str, str]]:
    """Return ``(source, pages)`` for every citation in ``answer``.

    ``pages`` is the cited page list normalised to digits separated by ", ": a
    single page stays ``"3"`` and a range such as ``"pages 2-3"`` becomes
    ``"2, 3"``. A citation bracket is one entry whatever its page count, so
    the validity ratio still describes citations rather than pages.
    """
    found: List[Tuple[str, str]] = []
    for src, page_spec in CITATION_PATTERN.findall(answer or ""):
        pages = re.findall(r"[0-9]+", page_spec)
        if not pages:
            continue
        found.append((src.strip(), ", ".join(pages)))
    return found


def citation_validity(answer: str, docs: List[Document]) -> Dict[str, float]:
    found = parse_citations(answer)
    if not found:
        return {"citation_count": 0, "valid_citation_count": 0, "validity_ratio": 0.0}

    allowed = build_allowed_citations(docs)
    valid = 0
    for src, pages in found:
        # Every page named must have been retrieved: a range that adds a page
        # the model never saw is as fabricated as an invented filename.
        cited_pages = [p.strip() for p in pages.split(",") if p.strip()]
        if cited_pages and all((src, page) in allowed for page in cited_pages):
            valid += 1
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
