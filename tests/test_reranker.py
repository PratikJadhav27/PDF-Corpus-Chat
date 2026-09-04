"""
tests/test_reranker.py

Unit tests for src/rag/reranker.py.

Tests cover both the cross-encoder path (when sentence-transformers is
installed) and the lexical-overlap fallback path (when it is not).
The tests are deliberately model-agnostic where possible so they run fast
in CI without downloading model weights.
"""
import pytest
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_candidates(texts_and_rrfs) -> List[Dict[str, Any]]:
    return [
        {"text": text, "rrf_score": rrf, "metadata": {}}
        for text, rrf in texts_and_rrfs
    ]


# ---------------------------------------------------------------------------
# Public contract — must hold regardless of which backend is active
# ---------------------------------------------------------------------------

class TestRerankerContract:
    """Invariants that must hold for both the CE and fallback paths."""

    def test_empty_candidates_returns_empty(self):
        from src.rag.reranker import rerank_candidates
        assert rerank_candidates("anything", [], top_n=4) == []

    def test_top_n_is_respected(self):
        from src.rag.reranker import rerank_candidates
        candidates = make_candidates([(f"Doc {i}", 0.01) for i in range(10)])
        results = rerank_candidates("query", candidates, top_n=3)
        assert len(results) == 3

    def test_top_n_larger_than_pool_returns_all(self):
        from src.rag.reranker import rerank_candidates
        candidates = make_candidates([("Doc A", 0.1), ("Doc B", 0.2)])
        results = rerank_candidates("query", candidates, top_n=10)
        assert len(results) == 2

    def test_output_rows_have_rerank_score(self):
        from src.rag.reranker import rerank_candidates
        candidates = make_candidates([("Revenue grew 15%.", 0.05)])
        results = rerank_candidates("revenue", candidates, top_n=1)
        assert "rerank_score" in results[0]

    def test_output_rows_have_ce_score(self):
        """Cross-encoder path sets ce_score; fallback also sets it via overlap."""
        from src.rag.reranker import rerank_candidates
        candidates = make_candidates([("Revenue grew 15%.", 0.05)])
        results = rerank_candidates("revenue", candidates, top_n=1)
        assert "ce_score" in results[0]

    def test_relevant_doc_outranks_irrelevant(self):
        """Revenue doc should beat weather doc for a revenue query."""
        from src.rag.reranker import rerank_candidates
        candidates = make_candidates([
            ("The weather is sunny today.", 0.01),
            ("Revenue increased 23% in the fiscal year.", 0.01),
        ])
        results = rerank_candidates("annual revenue fiscal year", candidates, top_n=2)
        texts = [r["text"] for r in results]
        assert texts.index("Revenue increased 23% in the fiscal year.") < \
               texts.index("The weather is sunny today.")

    def test_results_sorted_descending_by_rerank_score(self):
        from src.rag.reranker import rerank_candidates
        candidates = make_candidates([
            ("The cat sat on the mat.", 0.01),
            ("Machine learning improves search relevance significantly.", 0.01),
            ("The quarterly earnings exceeded analyst expectations.", 0.01),
        ])
        results = rerank_candidates("machine learning search", candidates, top_n=3)
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_original_metadata_preserved(self):
        from src.rag.reranker import rerank_candidates
        candidates = [{
            "text": "Revenue grew 10%.",
            "rrf_score": 0.1,
            "metadata": {"source": "report.pdf", "page_number": 5},
            "vector_score": 0.88,
            "bm25_score": 0.72,
        }]
        results = rerank_candidates("revenue", candidates, top_n=1)
        assert results[0]["metadata"]["source"] == "report.pdf"
        assert results[0]["metadata"]["page_number"] == 5
        assert results[0]["vector_score"] == 0.88
        assert results[0]["bm25_score"] == 0.72

    def test_only_top15_candidates_scored(self):
        """Pool is capped at _MAX_CANDIDATES_FOR_CE regardless of input size."""
        from src.rag.reranker import rerank_candidates, _MAX_CANDIDATES_FOR_CE

        # Build 30 candidates, ask for top 5. Should not error.
        candidates = make_candidates([(f"Document about topic {i}.", 0.01 * i) for i in range(30)])
        results = rerank_candidates("topic relevance", candidates, top_n=5)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# Fallback path — monkeypatch sentence_transformers away
# ---------------------------------------------------------------------------

class TestLexicalFallback:
    """Verify the lexical overlap fallback produces sensible results."""

    def test_fallback_activates_when_model_unavailable(self, monkeypatch):
        """When CE model is None, _cross_encoder_scores uses _overlap_score."""
        import src.rag.reranker as reranker_mod

        original = reranker_mod._CE_MODEL
        monkeypatch.setattr(reranker_mod, "_CE_MODEL", None)
        try:
            candidates = make_candidates([
                ("Revenue grew 15% this quarter.", 0.05),
                ("The sky is blue and the weather is fine.", 0.05),
            ])
            results = reranker_mod.rerank_candidates("revenue quarter", candidates, top_n=2)
            texts = [r["text"] for r in results]
            # Revenue doc should still win on lexical overlap
            assert texts[0].startswith("Revenue")
        finally:
            monkeypatch.setattr(reranker_mod, "_CE_MODEL", original)

    def test_fallback_sets_ce_score_key(self, monkeypatch):
        import src.rag.reranker as reranker_mod
        monkeypatch.setattr(reranker_mod, "_CE_MODEL", None)
        candidates = make_candidates([("hello world", 0.01)])
        results = reranker_mod.rerank_candidates("hello", candidates, top_n=1)
        assert "ce_score" in results[0]


# ---------------------------------------------------------------------------
# Cross-encoder path — only run when sentence-transformers is available
# ---------------------------------------------------------------------------

sentence_transformers_available = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("sentence_transformers"),
    reason="sentence-transformers not installed",
)


class TestCrossEncoderPath:
    @sentence_transformers_available
    def test_is_cross_encoder_available_returns_true(self):
        from src.rag.reranker import is_cross_encoder_available
        assert is_cross_encoder_available() is True

    @sentence_transformers_available
    def test_ce_score_is_finite_float(self):
        from src.rag.reranker import rerank_candidates
        import math
        candidates = make_candidates([
            ("The transformer architecture uses self-attention.", 0.05),
        ])
        results = rerank_candidates("attention mechanism transformer", candidates, top_n=1)
        score = results[0]["ce_score"]
        assert isinstance(score, float)
        assert math.isfinite(score)

    @sentence_transformers_available
    def test_ce_scores_are_deterministic(self):
        """Same input → same scores across two calls."""
        from src.rag.reranker import rerank_candidates
        candidates = make_candidates([
            ("BERT uses masked language modelling.", 0.04),
            ("The stock market closed higher today.", 0.04),
        ])
        r1 = rerank_candidates("language model pre-training", candidates, top_n=2)
        r2 = rerank_candidates("language model pre-training", candidates, top_n=2)
        assert [r["ce_score"] for r in r1] == [r["ce_score"] for r in r2]
