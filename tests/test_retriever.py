"""
tests/test_retriever.py

Unit tests for src/rag/retriever.py and related retrieval plumbing.

These tests use mocking to avoid real FAISS / API calls.

Tests cover:
- retrieve_docs: similarity and MMR modes
- _vector_hits_with_scores: correct wrapping of FAISS results
- retrieve_hybrid_reranked: end-to-end pipeline (vector + BM25 + fuse + rerank)
- retrieve_hybrid_reranked: graceful handling of empty BM25 / empty vector results
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.rag.retriever import retrieve_docs, _vector_hits_with_scores, retrieve_hybrid_reranked


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_doc(source: str = "test.pdf", page: int = 1, chunk_index: int = 0, text: str = "hello"):
    return Document(
        page_content=text,
        metadata={"source": source, "page_number": page, "chunk_index": chunk_index},
    )


def _make_fake_db(docs=None, scored_docs=None):
    """Build a minimal mock FAISS-like object."""
    db = MagicMock()
    db.similarity_search.return_value = docs or [_fake_doc()]
    db.max_marginal_relevance_search.return_value = docs or [_fake_doc()]
    db.similarity_search_with_score.return_value = [
        (d, 0.85) for d in (scored_docs or [_fake_doc()])
    ]
    return db


# ---------------------------------------------------------------------------
# retrieve_docs
# ---------------------------------------------------------------------------

class TestRetrieveDocs:
    def test_similarity_mode_calls_similarity_search(self):
        db = _make_fake_db()
        result = retrieve_docs(db, "test query", mode="similarity", k=3)
        db.similarity_search.assert_called_once_with("test query", k=3)
        assert result == db.similarity_search.return_value

    def test_mmr_mode_calls_mmr_search(self):
        db = _make_fake_db()
        result = retrieve_docs(db, "test query", mode="mmr", k=4)
        db.max_marginal_relevance_search.assert_called_once()
        call_kwargs = db.max_marginal_relevance_search.call_args
        assert call_kwargs[1]["k"] == 4 or call_kwargs[0][1] == 4

    def test_default_mode_is_similarity(self):
        db = _make_fake_db()
        with patch("src.rag.retriever.RETRIEVAL") as mock_cfg:
            mock_cfg.mode = "similarity"
            mock_cfg.top_k = 4
            retrieve_docs(db, "query")
        db.similarity_search.assert_called()

    def test_returns_documents(self):
        docs = [_fake_doc("a.pdf", 1, 0), _fake_doc("b.pdf", 2, 0)]
        db = _make_fake_db(docs=docs)
        result = retrieve_docs(db, "query", mode="similarity", k=2)
        assert len(result) == 2

    def test_unknown_mode_falls_back_to_similarity(self):
        db = _make_fake_db()
        # Any mode that's not "mmr" falls through to similarity search
        result = retrieve_docs(db, "query", mode="cosine", k=2)
        db.similarity_search.assert_called()


# ---------------------------------------------------------------------------
# _vector_hits_with_scores
# ---------------------------------------------------------------------------

class TestVectorHitsWithScores:
    def test_returns_list_of_tuples(self):
        doc = _fake_doc()
        db = _make_fake_db(scored_docs=[doc])
        result = _vector_hits_with_scores(db, "query", k=1)
        assert isinstance(result, list)
        assert len(result) == 1
        row, score = result[0]
        assert isinstance(row, dict)
        assert isinstance(score, float)

    def test_text_and_metadata_extracted(self):
        doc = _fake_doc(source="myfile.pdf", page=5, text="important text")
        db = MagicMock()
        db.similarity_search_with_score.return_value = [(doc, 0.92)]
        result = _vector_hits_with_scores(db, "query", k=1)
        row, score = result[0]
        assert row["text"] == "important text"
        assert row["metadata"]["source"] == "myfile.pdf"
        assert score == pytest.approx(0.92)

    def test_empty_results(self):
        db = MagicMock()
        db.similarity_search_with_score.return_value = []
        result = _vector_hits_with_scores(db, "query", k=5)
        assert result == []


# ---------------------------------------------------------------------------
# retrieve_hybrid_reranked
# ---------------------------------------------------------------------------

class TestRetrieveHybridReranked:
    def _make_hybrid_db(self, docs=None):
        doc = docs or [_fake_doc("a.pdf", 1, 0, "machine learning overview")]
        db = MagicMock()
        db.similarity_search_with_score.return_value = [(d, 0.9) for d in doc]
        return db

    @patch("src.rag.retriever.bm25_search_collection")
    def test_returns_list_of_documents(self, mock_bm25):
        mock_bm25.return_value = []
        db = self._make_hybrid_db()
        result = retrieve_hybrid_reranked(db, "test_col", "what is ML?", top_k=2, candidate_k=4)
        assert isinstance(result, list)
        assert all(isinstance(d, Document) for d in result)

    @patch("src.rag.retriever.bm25_search_collection")
    def test_retrieval_metadata_attached(self, mock_bm25):
        mock_bm25.return_value = []
        db = self._make_hybrid_db()
        result = retrieve_hybrid_reranked(db, "test_col", "ML query", top_k=1, candidate_k=4)
        if result:
            assert "retrieval" in result[0].metadata
            retrieval = result[0].metadata["retrieval"]
            assert "rrf_score" in retrieval
            assert "rerank_score" in retrieval
            assert "channels" in retrieval

    @patch("src.rag.retriever.bm25_search_collection")
    def test_bm25_called_with_correct_args(self, mock_bm25):
        mock_bm25.return_value = []
        db = self._make_hybrid_db()
        retrieve_hybrid_reranked(db, "my_coll", "test query", top_k=2, candidate_k=8)
        mock_bm25.assert_called_once_with("my_coll", "test query", k=8)

    @patch("src.rag.retriever.bm25_search_collection")
    def test_empty_vector_still_returns_bm25_results(self, mock_bm25):
        doc = _fake_doc("bm25_only.pdf", 2, 0, "bm25 result text")
        mock_bm25.return_value = [({"text": doc.page_content, "metadata": doc.metadata}, 5.5)]
        db = MagicMock()
        db.similarity_search_with_score.return_value = []
        result = retrieve_hybrid_reranked(db, "coll", "query", top_k=2, candidate_k=4)
        assert len(result) >= 1

    @patch("src.rag.retriever.bm25_search_collection")
    def test_top_k_limits_output(self, mock_bm25):
        docs = [_fake_doc("a.pdf", i, 0, f"text {i}") for i in range(10)]
        db = MagicMock()
        db.similarity_search_with_score.return_value = [(d, 0.8) for d in docs]
        mock_bm25.return_value = [
            ({"text": d.page_content, "metadata": d.metadata}, float(10 - i))
            for i, d in enumerate(docs)
        ]
        result = retrieve_hybrid_reranked(db, "coll", "query", top_k=3, candidate_k=10)
        assert len(result) <= 3

    @patch("src.rag.retriever.bm25_search_collection")
    def test_no_duplicates_in_output(self, mock_bm25):
        doc = _fake_doc("shared.pdf", 1, 0, "shared chunk text")
        db = MagicMock()
        db.similarity_search_with_score.return_value = [(doc, 0.9)]
        mock_bm25.return_value = [({"text": doc.page_content, "metadata": doc.metadata}, 5.5)]
        result = retrieve_hybrid_reranked(db, "coll", "query", top_k=4, candidate_k=4)
        keys = [
            f"{d.metadata.get('source')}::{d.metadata.get('page_number')}::{d.metadata.get('chunk_index')}"
            for d in result
        ]
        assert len(keys) == len(set(keys)), "Duplicate chunks found in output"
