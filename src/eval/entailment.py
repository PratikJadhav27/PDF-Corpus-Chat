"""
src/eval/entailment.py

Citation Entailment Evaluation — DSPy-powered.

Goal
----
Measure whether each inline citation in the RAG answer is *actually entailed*
by the chunk it references.  This is stricter than citation validity (which
only checks that the source/page pair exists in the retrieved set) — it asks
the LLM judge whether the cited passage could logically support the claim made
in the sentence that contains the citation.

Relation to existing metrics
-----------------------------
citation_validity  (src/rag/validators.py)
  — Structural check: does [source.pdf | page N] refer to a source that was
    actually retrieved?  Pure regex, no LLM, runs in the hot path.

citation_entailment  (this file)
  — Semantic check: does the cited chunk *support* the claim? LLM-judge,
    eval-only, never runs in the live query path.

Usage
-----
from src.eval.entailment import evaluate_citation_entailment

result = evaluate_citation_entailment(
    answer=answer_text,
    docs=retrieved_docs,
)
# result["entailment_rate"]  — float in [0.0, 1.0], avg over all citations
# result["citations"]        — per-citation breakdown
# result["error"]            — str | None

Design decisions
----------------
- Uses `dspy.ChainOfThought` so the model reasons before scoring.
- Hard-caps chunk text at 1 200 chars and claim text at 400 chars to keep
  latency and cost reasonable.
- Gracefully degrades: if no citations found, returns entailment_rate=1.0
  (vacuously true — no hallucination possible if no claims made).
- Isolated from the live query path — only called from scripts/run_eval.py
  when --entailment flag is passed.
"""

import os
import re
from typing import Any, Dict, List, Optional

import dspy
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# DSPy configuration (shared model with metrics.py)
# ---------------------------------------------------------------------------

_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gemini-2.5-flash")

# Lazy singleton — only configure DSPy once.  metrics.py may have already
# called dspy.configure(), which is idempotent.
_dspy_configured = False


def _ensure_dspy_configured() -> None:
    global _dspy_configured
    if not _dspy_configured:
        gemini_lm = dspy.LM(f"gemini/{_JUDGE_MODEL}")
        dspy.configure(lm=gemini_lm)
        _dspy_configured = True


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Matches: [filename.pdf | page 3]  or  [Attention_Is_All_You_Need.pdf | page 12]
_CITATION_RE = re.compile(
    r"\[([^\|\]]+)\|\s*page\s*(\d+)\]",
    re.IGNORECASE,
)

# Splits answer into sentences (rough but good enough for eval purposes)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _parse_citations(answer: str) -> List[Dict[str, str]]:
    """
    Extract all inline citations from the answer text.

    Returns a list of dicts with keys:
      - source   : normalised filename (e.g. "attention_is_all_you_need.pdf")
      - page     : page number as string
      - sentence : the sentence in the answer that contains this citation
    """
    if not answer:
        return []

    # Split into sentences so we can associate each citation with its claim
    sentences = _SENTENCE_RE.split(answer.strip())

    citations = []
    for sentence in sentences:
        for match in _CITATION_RE.finditer(sentence):
            source = os.path.basename(match.group(1).strip()).lower()
            page = match.group(2).strip()
            # Strip the citation bracket itself to get just the claim text
            claim = _CITATION_RE.sub("", sentence).strip()
            citations.append({"source": source, "page": page, "claim": claim, "sentence": sentence})

    return citations


def _find_cited_chunk(
    source: str,
    page: str,
    docs: List[Document],
) -> Optional[str]:
    """
    Return the text of the chunk whose source filename and page match.
    Returns None if no matching chunk is found.
    """
    page_int = int(page) if page.isdigit() else -1
    for doc in docs:
        doc_source = os.path.basename(doc.metadata.get("source", "")).lower()
        doc_page = doc.metadata.get("page_number", -1)
        try:
            doc_page = int(doc_page)
        except (TypeError, ValueError):
            doc_page = -1

        if doc_source == source and doc_page == page_int:
            return doc.page_content

    return None


# ---------------------------------------------------------------------------
# DSPy signature
# ---------------------------------------------------------------------------

class CitationEntailment(dspy.Signature):
    """
    You are an expert fact-checker.

    Determine whether the CLAIM made in the answer is *entailed by* (i.e.,
    logically supported by) the CITED PASSAGE.

    Strict entailment rules:
    - Score 1.0  — the cited passage contains information that directly
                   supports the claim. No hallucination.
    - Score 0.5  — the cited passage partially supports the claim but is
                   missing some details, or the claim is a reasonable
                   paraphrase/inference from the passage.
    - Score 0.0  — the cited passage does NOT support the claim, or the
                   claim introduces information not in the passage.

    Be strict. Partial credit (0.5) is only for reasonable paraphrases.
    """
    claim = dspy.InputField(
        desc="The specific factual claim made in the answer sentence (citation bracket removed)"
    )
    cited_passage = dspy.InputField(
        desc="The text of the chunk referenced by the citation"
    )
    explanation = dspy.OutputField(
        desc="One sentence explaining why the claim is or is not supported"
    )
    entailment_score = dspy.OutputField(
        desc="A float: 1.0 (fully supported), 0.5 (partially supported), or 0.0 (not supported)"
    )


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

_SCORE_RE = re.compile(r"\b(1\.0|0\.5|0\.0|1|0)\b")


def _parse_entailment_score(text: str) -> float:
    if not text:
        return 0.0
    match = _SCORE_RE.search(str(text))
    if match:
        return round(float(match.group(1)), 1)
    # Fallback: look for any float in [0,1]
    general = re.search(r"\b([01](?:\.\d+)?)\b", str(text))
    if general:
        return round(min(max(float(general.group(1)), 0.0), 1.0), 4)
    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_citation_entailment(
    answer: str,
    docs: List[Document],
) -> Dict[str, Any]:
    """
    Evaluate citation entailment for all inline citations in an answer.

    Args:
        answer: The RAG-generated answer text (may contain inline citations).
        docs:   The retrieved Document objects used to generate the answer.

    Returns a dict with:
        entailment_rate   — float [0, 1], mean score across all citations
                            (1.0 vacuously if no citations found)
        citation_count    — total number of inline citations found
        entailed_count    — citations scoring >= 0.5
        not_found_count   — citations where the cited chunk was not in docs
        citations         — list of per-citation dicts with:
                              source, page, claim, score, explanation, error
        error             — str | None  (top-level error, not per-citation)
    """
    _ensure_dspy_configured()

    if not answer:
        return {
            "entailment_rate": 0.0,
            "citation_count": 0,
            "entailed_count": 0,
            "not_found_count": 0,
            "citations": [],
            "error": "Empty answer",
        }

    parsed = _parse_citations(answer)

    if not parsed:
        # Vacuously true — no inline citations means no hallucinated citations
        return {
            "entailment_rate": 1.0,
            "citation_count": 0,
            "entailed_count": 0,
            "not_found_count": 0,
            "citations": [],
            "error": None,
        }

    judge = dspy.ChainOfThought(CitationEntailment)
    citation_results = []
    scores = []
    not_found_count = 0

    for cit in parsed:
        chunk_text = _find_cited_chunk(cit["source"], cit["page"], docs)

        if chunk_text is None:
            # Citation refers to a chunk that wasn't retrieved — not entailed
            not_found_count += 1
            citation_results.append({
                "source": cit["source"],
                "page": cit["page"],
                "claim": cit["claim"][:200],
                "score": 0.0,
                "explanation": "Cited chunk not found in retrieved documents.",
                "error": None,
            })
            scores.append(0.0)
            continue

        try:
            res = judge(
                claim=cit["claim"][:400],
                cited_passage=chunk_text[:1200],
            )
            score = _parse_entailment_score(str(res.entailment_score))
            citation_results.append({
                "source": cit["source"],
                "page": cit["page"],
                "claim": cit["claim"][:200],
                "score": score,
                "explanation": str(res.explanation)[:400],
                "error": None,
            })
            scores.append(score)
        except Exception as exc:
            citation_results.append({
                "source": cit["source"],
                "page": cit["page"],
                "claim": cit["claim"][:200],
                "score": 0.0,
                "explanation": "",
                "error": str(exc),
            })
            scores.append(0.0)

    entailment_rate = round(sum(scores) / len(scores), 4) if scores else 0.0
    entailed_count = sum(1 for s in scores if s >= 0.5)

    return {
        "entailment_rate": entailment_rate,
        "citation_count": len(parsed),
        "entailed_count": entailed_count,
        "not_found_count": not_found_count,
        "citations": citation_results,
        "error": None,
    }
