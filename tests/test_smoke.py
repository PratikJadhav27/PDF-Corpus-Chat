"""
Smoke tests — verify the core data pipeline works end-to-end without
requiring an API key, Qdrant server, or any external service.

These tests use only deterministic, pure-Python components (BM25, RRF,
reranker, citation validators) so they always run in CI with zero cost.
"""

import json
import pathlib
import tempfile
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# BM25 smoke
# ---------------------------------------------------------------------------

class TestBM25Smoke:
    def test_index_build_and_search_returns_results(self):
        from src.rag.bm25_index import SimpleBM25

        docs = [
            "The quarterly revenue increased by 23 percent year over year.",
            "Operating expenses remained flat compared to the previous quarter.",
            "The company expanded its workforce by hiring 500 new engineers.",
        ]
        bm25 = SimpleBM25(docs)
        hits = bm25.search("revenue quarterly", k=3)

        assert len(hits) > 0, "BM25 should return at least one hit"
        # The revenue doc should rank first
        assert hits[0].index == 0, "Revenue document should be top hit"

    def test_empty_corpus_returns_no_results(self):
        from src.rag.bm25_index import SimpleBM25

        bm25 = SimpleBM25([])
        hits = bm25.search("revenue", k=5)
        assert hits == []

    def test_no_matching_terms_returns_empty(self):
        from src.rag.bm25_index import SimpleBM25

        bm25 = SimpleBM25(["Hello world", "Python programming"])
        hits = bm25.search("zzzyyyxxx", k=5)
        assert hits == []


# ---------------------------------------------------------------------------
# BM25 persistence smoke
# ---------------------------------------------------------------------------

class TestBM25Persistence:
    def test_save_and_load_chunk_corpus(self, tmp_path, monkeypatch):
        """save_chunk_corpus writes JSONL; load_chunk_corpus reads it back."""
        from src.rag import bm25_index

        # Redirect vector_stores/ to a temp directory
        monkeypatch.chdir(tmp_path)

        collection = "smoke_test_collection"
        chunks = [
            {"text": "Revenue grew 18% in Q4.", "metadata": {"source": "report.pdf", "page": 1}},
            {"text": "Operating costs declined.", "metadata": {"source": "report.pdf", "page": 2}},
        ]

        bm25_index.save_chunk_corpus(collection, chunks)

        loaded = bm25_index.load_chunk_corpus(collection)
        assert len(loaded) == 2
        assert loaded[0]["text"] == chunks[0]["text"]
        assert loaded[1]["metadata"]["page"] == 2

    def test_load_nonexistent_corpus_returns_empty(self, tmp_path, monkeypatch):
        from src.rag import bm25_index

        monkeypatch.chdir(tmp_path)
        result = bm25_index.load_chunk_corpus("does_not_exist")
        assert result == []


# ---------------------------------------------------------------------------
# RRF smoke
# ---------------------------------------------------------------------------

class TestRRFFusion:
    def test_fused_results_contain_both_sources(self):
        from src.rag.fusion import rrf_fuse

        # rrf_fuse takes lists of (doc_dict, raw_score) tuples
        dense_hits = [
            ({"text": "Revenue grew 18%.", "metadata": {"source": "a.pdf", "page_number": 1, "chunk_index": 0}}, 0.92),
            ({"text": "Costs declined.",   "metadata": {"source": "a.pdf", "page_number": 2, "chunk_index": 1}}, 0.85),
        ]
        bm25_hits = [
            ({"text": "Costs declined.",   "metadata": {"source": "a.pdf", "page_number": 2, "chunk_index": 1}}, 14.2),
            ({"text": "New product launch.","metadata": {"source": "b.pdf", "page_number": 5, "chunk_index": 0}}, 11.8),
        ]

        fused = rrf_fuse(dense_hits, bm25_hits)
        assert len(fused) >= 1

        # All fused results should have an rrf_score
        for item in fused:
            assert "rrf_score" in item, "Each fused result must carry rrf_score"

    def test_chunk_in_both_channels_ranks_first(self):
        from src.rag.fusion import rrf_fuse

        shared_doc = {"text": "Shared chunk.", "metadata": {"source": "x.pdf", "page_number": 1, "chunk_index": 0}}
        unique_doc = {"text": "Unique chunk.", "metadata": {"source": "x.pdf", "page_number": 2, "chunk_index": 1}}

        dense_hits = [(shared_doc, 0.9), (unique_doc, 0.7)]
        bm25_hits  = [(shared_doc, 10.0)]

        fused = rrf_fuse(dense_hits, bm25_hits)
        # Shared chunk appears in both channels — should have higher RRF
        assert fused[0]["text"] == "Shared chunk."


# ---------------------------------------------------------------------------
# Reranker smoke
# ---------------------------------------------------------------------------

class TestRerankerSmoke:
    def test_reranker_sorts_by_relevance(self):
        from src.rag.reranker import rerank_candidates

        candidates = [
            {"text": "The weather is sunny today.", "rrf_score": 0.01},
            {"text": "Revenue increased by 23 percent in the fiscal year.", "rrf_score": 0.02},
            {"text": "The board approved the annual budget.", "rrf_score": 0.015},
        ]

        results = rerank_candidates("annual revenue fiscal", candidates, top_n=3)

        assert len(results) == 3
        # Revenue doc should outrank weather doc
        texts = [r["text"] for r in results]
        assert texts.index("Revenue increased by 23 percent in the fiscal year.") < \
               texts.index("The weather is sunny today.")

    def test_reranker_respects_top_n(self):
        from src.rag.reranker import rerank_candidates

        candidates = [{"text": f"Doc {i}", "rrf_score": 0.01} for i in range(10)]
        results = rerank_candidates("query", candidates, top_n=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Citation validator smoke
# ---------------------------------------------------------------------------

class TestCitationValidatorSmoke:
    def test_valid_citation_passes(self):
        from src.rag.validators import citation_validity
        from langchain_core.documents import Document

        answer = "The revenue grew [report.pdf | page 3]."
        docs = [
            Document(page_content="Revenue grew.", metadata={"source": "report.pdf", "page_number": 3}),
        ]
        result = citation_validity(answer, docs)
        assert result["valid_citation_count"] >= 1
        assert result["validity_ratio"] == 1.0

    def test_hallucinated_citation_is_caught(self):
        from src.rag.validators import citation_validity
        from langchain_core.documents import Document

        answer = "The revenue grew [phantom.pdf | page 99]."
        docs = [
            Document(page_content="Revenue grew.", metadata={"source": "report.pdf", "page_number": 3}),
        ]
        result = citation_validity(answer, docs)
        assert result["valid_citation_count"] == 0
        assert result["validity_ratio"] == 0.0
