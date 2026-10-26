"""
Regression tests for ISSUE-007 — the source-card relevance badge never rendered.

The card read flat ``metadata["rerank_score"]`` / ``metadata["score"]``, but
hybrid retrieval stores its scores under ``metadata["retrieval"]``, so the
"▲ N% relevance" affordance was dead in the default mode. The reranked score is
also an unbounded cross-encoder logit: formatting it with ``:.1%`` would print
nonsense (e.g. ``-320.0%``).

The fix normalises the score where the scorer's semantics are known — the
reranker writes a 0-1 ``relevance`` alongside the raw logits, the retriever
carries it in ``metadata["retrieval"]``, and the card formats only that.
"""

import math
from pathlib import Path

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from app.views.chat import _relevance_label

APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


# ---------------------------------------------------------------------------
# Reranker — normalisation happens next to the scorer
# ---------------------------------------------------------------------------

class _FakeCrossEncoder:
    """Minimal stand-in for sentence_transformers.CrossEncoder."""

    def __init__(self, logits):
        self._logits = list(logits)

    def predict(self, pairs, show_progress_bar=False):  # noqa: ARG002 - API shape
        return list(self._logits)


def _candidate(text: str) -> dict:
    return {"text": text, "rrf_score": 0.01, "metadata": {}}


def test_cross_encoder_logits_become_normalised_relevance(monkeypatch):
    import src.rag.reranker as reranker

    monkeypatch.setattr(reranker, "_CE_MODEL", _FakeCrossEncoder([8.0, 0.0, -8.0]))

    results = reranker.rerank_candidates(
        "query",
        [_candidate("relevant"), _candidate("neutral"), _candidate("irrelevant")],
        top_n=3,
    )

    by_text = {r["text"]: r for r in results}
    assert by_text["relevant"]["relevance"] == round(1 / (1 + math.exp(-8.0)), 4)
    assert by_text["neutral"]["relevance"] == 0.5
    assert by_text["irrelevant"]["relevance"] == round(1 / (1 + math.exp(8.0)), 4)

    # Raw logits are untouched and still drive ordering.
    assert [r["rerank_score"] for r in results] == [8.0, 0.0, -8.0]
    assert [r["relevance"] for r in results] == sorted(
        (r["relevance"] for r in results), reverse=True
    )
    assert all(0.0 <= r["relevance"] <= 1.0 for r in results)


def test_lexical_fallback_reports_its_overlap_fraction(monkeypatch):
    import src.rag.reranker as reranker

    monkeypatch.setattr(reranker, "_CE_MODEL", None)

    results = reranker.rerank_candidates(
        "revenue quarter",
        [
            _candidate("Revenue grew 15% this quarter."),
            _candidate("The sky is blue and the weather is fine."),
        ],
        top_n=2,
    )

    by_text = {r["text"]: r for r in results}
    assert by_text["Revenue grew 15% this quarter."]["relevance"] == 1.0
    assert by_text["The sky is blue and the weather is fine."]["relevance"] == 0.0


# ---------------------------------------------------------------------------
# Retriever — relevance travels with the chunk
# ---------------------------------------------------------------------------

def _fake_rerank(query, candidates, top_n):  # noqa: ARG001 - matches signature
    rows = []
    for candidate in candidates[:top_n]:
        row = dict(candidate)
        row["rerank_score"] = 7.5
        row["ce_score"] = 7.5
        row["relevance"] = 0.8734
        rows.append(row)
    return rows


def _vector_only_db():
    from unittest.mock import MagicMock

    doc = Document(
        page_content="alpha text",
        metadata={"source": "a.pdf", "page_number": 1, "chunk_index": 0},
    )
    db = MagicMock()
    db.similarity_search_with_score.return_value = [(doc, 0.82)]
    return db


def test_retrieval_metadata_carries_both_raw_and_normalised_scores(monkeypatch):
    import src.rag.retriever as retriever

    monkeypatch.setattr(retriever, "bm25_search_collection", lambda *a, **k: [])
    monkeypatch.setattr(retriever, "rerank_candidates", _fake_rerank)

    docs = retriever.retrieve_hybrid_reranked(
        _vector_only_db(), "coll", "query", top_k=1, candidate_k=4
    )

    retrieval = docs[0].metadata["retrieval"]
    assert retrieval["rerank_score"] == 7.5      # raw, noisy for display
    assert retrieval["relevance"] == 0.8734      # normalised, display-safe


# ---------------------------------------------------------------------------
# Card label
# ---------------------------------------------------------------------------

def test_badge_formats_the_normalised_fraction():
    label = _relevance_label({"retrieval": {"relevance": 0.8734, "rerank_score": 8.1}})
    assert label == "▲ 87.3% relevance"


def test_raw_logit_alone_does_not_become_a_percentage():
    """A CE logit is unbounded — never format it as a relevance percentage."""
    assert _relevance_label({"retrieval": {"rerank_score": -3.2}}) == ""
    assert _relevance_label({"retrieval": {"ce_score": 7.5}}) == ""
    assert _relevance_label({"rerank_score": 0.9}) == ""
    assert _relevance_label({"score": 0.9}) == ""


def test_missing_or_bad_scores_render_nothing():
    assert _relevance_label({}) == ""
    assert _relevance_label(None) == ""
    assert _relevance_label({"retrieval": None}) == ""
    assert _relevance_label({"retrieval": {"relevance": None}}) == ""
    assert _relevance_label({"retrieval": {"relevance": "n/a"}}) == ""


def test_badge_is_clamped_to_zero_one():
    assert _relevance_label({"retrieval": {"relevance": 1.4}}) == "▲ 100.0% relevance"
    assert _relevance_label({"retrieval": {"relevance": -0.2}}) == "▲ 0.0% relevance"


# ---------------------------------------------------------------------------
# Real entrypoint
# ---------------------------------------------------------------------------

def _doc(source, page, metadata_extra=None):
    metadata = {"source": source, "page_number": page, "chunk_index": 0}
    metadata.update(metadata_extra or {})
    return Document(page_content="Alpha applies to bravo.", metadata=metadata)


def test_chat_history_renders_badge_from_retrieval_metadata():
    hybrid_doc = _doc(
        "a.pdf",
        1,
        {
            "retrieval": {
                "rrf_score": 0.031,
                "rerank_score": 7.5,
                "ce_score": 7.5,
                "relevance": 0.8734,
                "channels": ["vector", "bm25"],
            }
        },
    )
    legacy_doc = _doc("b.pdf", 2)

    at = AppTest.from_file(str(APP_MAIN), default_timeout=120)
    at.session_state["chat_history"] = [
        {
            "timestamp": "2026-10-26 10:00:00",
            "question": "What is alpha?",
            "answer": "Alpha applies to bravo [a.pdf | page 1].",
            "latency": {"total_sec": 0.4, "retrieval_sec": 0.1, "llm_sec": 0.3},
            "docs": [hybrid_doc, legacy_doc],
            "prompt_version": "v1_strict",
            "collection": "default",
        }
    ]
    at.run()

    assert not at.exception, at.exception

    body = "\n".join(m.value for m in at.markdown)
    assert body.count("▲ 87.3% relevance") == 1   # only the scored chunk
    # The raw logit must not leak into the badge.
    assert "750.0%" not in body
    assert "-" not in body.split("▲")[-1].split("relevance")[0]
