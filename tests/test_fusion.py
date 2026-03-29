"""
tests/test_fusion.py

Unit tests for src/rag/fusion.py

Tests cover:
- RRF score computation
- Deduplication by (source, page, chunk_index)
- Ranking stability
- Single-channel (vector-only or BM25-only) behavior
- Empty input handling
"""

import pytest
from src.rag.fusion import chunk_key, rrf_fuse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hit(source: str, page, chunk_index, text: str = "some text"):
    """Create a (row, raw_score) tuple as returned by retrievers."""
    return (
        {
            "text": text,
            "metadata": {
                "source": source,
                "page_number": page,
                "chunk_index": chunk_index,
            },
        },
        1.0,  # raw score doesn't matter for fusion tests
    )


# ---------------------------------------------------------------------------
# chunk_key
# ---------------------------------------------------------------------------

class TestChunkKey:
    def test_basic(self):
        meta = {"source": "doc.pdf", "page_number": 3, "chunk_index": 0}
        assert chunk_key(meta) == "doc.pdf::3::0"

    def test_missing_fields_use_defaults(self):
        key = chunk_key({})
        assert key == "unknown::?::?"

    def test_different_pages_produce_different_keys(self):
        m1 = {"source": "a.pdf", "page_number": 1, "chunk_index": 0}
        m2 = {"source": "a.pdf", "page_number": 2, "chunk_index": 0}
        assert chunk_key(m1) != chunk_key(m2)

    def test_different_chunks_produce_different_keys(self):
        m1 = {"source": "a.pdf", "page_number": 1, "chunk_index": 0}
        m2 = {"source": "a.pdf", "page_number": 1, "chunk_index": 1}
        assert chunk_key(m1) != chunk_key(m2)


# ---------------------------------------------------------------------------
# rrf_fuse — empty inputs
# ---------------------------------------------------------------------------

class TestRrfFuseEmpty:
    def test_both_empty(self):
        result = rrf_fuse([], [])
        assert result == []

    def test_vector_only_empty(self):
        hit = _make_hit("x.pdf", 1, 0)
        result = rrf_fuse([], [hit])
        assert len(result) == 1
        assert result[0]["channels"] == ["bm25"]

    def test_bm25_only_empty(self):
        hit = _make_hit("x.pdf", 1, 0)
        result = rrf_fuse([hit], [])
        assert len(result) == 1
        assert result[0]["channels"] == ["vector"]


# ---------------------------------------------------------------------------
# rrf_fuse — deduplication
# ---------------------------------------------------------------------------

class TestRrfFuseDedupe:
    def test_same_chunk_in_both_channels_appears_once(self):
        hit = _make_hit("doc.pdf", 1, 0)
        result = rrf_fuse([hit], [hit])
        assert len(result) == 1

    def test_same_chunk_accumulates_rrf_score(self):
        hit = _make_hit("doc.pdf", 1, 0)
        result = rrf_fuse([hit], [hit])
        # Both channels contribute rrf = 1/(60+1) each; total ≈ 0.03278
        expected = (1.0 / 61) + (1.0 / 61)
        assert abs(result[0]["rrf_score"] - expected) < 1e-9

    def test_same_chunk_has_both_channels_listed(self):
        hit = _make_hit("doc.pdf", 1, 0)
        result = rrf_fuse([hit], [hit])
        assert set(result[0]["channels"]) == {"vector", "bm25"}

    def test_different_chunks_are_distinct(self):
        v_hit = _make_hit("doc.pdf", 1, 0)
        b_hit = _make_hit("doc.pdf", 1, 1)
        result = rrf_fuse([v_hit], [b_hit])
        assert len(result) == 2

    def test_different_sources_are_distinct(self):
        v_hit = _make_hit("a.pdf", 1, 0)
        b_hit = _make_hit("b.pdf", 1, 0)
        result = rrf_fuse([v_hit], [b_hit])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# rrf_fuse — ranking
# ---------------------------------------------------------------------------

class TestRrfFuseRanking:
    def test_chunk_in_both_channels_ranks_higher_than_single(self):
        shared = _make_hit("shared.pdf", 1, 0)
        only_bm25 = _make_hit("other.pdf", 2, 0)
        result = rrf_fuse([shared], [shared, only_bm25])
        # shared appears rank-1 in vector and rank-1 in bm25 → higher total rrf
        assert result[0]["key"].startswith("shared.pdf")

    def test_results_are_sorted_descending(self):
        hits = [
            _make_hit("a.pdf", 1, 0),
            _make_hit("b.pdf", 1, 0),
            _make_hit("c.pdf", 1, 0),
        ]
        result = rrf_fuse(hits, hits)
        scores = [r["rrf_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_rank_has_highest_rrf(self):
        # rank-1 doc in both lists should beat rank-2 doc in one list
        v_hits = [_make_hit("top.pdf", 1, 0), _make_hit("mid.pdf", 2, 0)]
        b_hits = [_make_hit("top.pdf", 1, 0)]
        result = rrf_fuse(v_hits, b_hits)
        assert result[0]["key"].startswith("top.pdf")


# ---------------------------------------------------------------------------
# rrf_fuse — score values
# ---------------------------------------------------------------------------

class TestRrfFuseScoreValues:
    def test_single_vector_hit_rrf_score(self):
        hit = _make_hit("a.pdf", 1, 0)
        result = rrf_fuse([hit], [])
        expected = 1.0 / (60 + 1)
        assert abs(result[0]["rrf_score"] - expected) < 1e-10

    def test_vector_score_stored(self):
        hit = _make_hit("a.pdf", 1, 0)
        result = rrf_fuse([hit], [])
        assert result[0]["vector_score"] == 1.0
        assert result[0]["bm25_score"] is None

    def test_bm25_score_stored(self):
        hit = _make_hit("a.pdf", 1, 0)
        result = rrf_fuse([], [hit])
        assert result[0]["bm25_score"] == 1.0
        assert result[0]["vector_score"] is None

    def test_custom_k_constant(self):
        hit = _make_hit("a.pdf", 1, 0)
        result = rrf_fuse([hit], [], k_constant=0)
        expected = 1.0 / (0 + 1)
        assert abs(result[0]["rrf_score"] - expected) < 1e-10
