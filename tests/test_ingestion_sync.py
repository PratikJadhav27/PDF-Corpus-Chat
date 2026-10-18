"""
Regression tests for ISSUE-001 — re-ingestion must keep the vector store, the
BM25/JSONL corpus and metadata.json in sync with per-file replace semantics.

All vector work uses a deterministic fake embedding model (no network, no API
key) and runs inside a pytest tmp directory, so these tests are fast and hermetic.
"""
import hashlib
import json

import pytest
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

from app import helpers
from src.rag.bm25_index import (
    bm25_search_collection,
    get_chunk_corpus_path,
    load_chunk_corpus,
)
from src.rag.vector_store import get_collection_dir, sanitize_collection_name


class FakeEmbeddings(Embeddings):
    """Deterministic 16-dimension embeddings derived from a SHA-256 digest.

    Subclasses langchain's Embeddings ABC because langchain-qdrant validates
    the concrete type when constructing a store over an existing collection.
    """

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(text):
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:16]]


def _chunks(source, texts):
    return [
        {
            "text": t,
            "metadata": {"source": source, "page_number": i + 1, "chunk_index": i},
        }
        for i, t in enumerate(texts)
    ]


def _stats(source, pages, extracted=None, empty=0, failed=0):
    extracted = pages if extracted is None else extracted
    return {
        "file_name": source,
        "total_pages": pages,
        "extracted_pages": extracted,
        "empty_pages": empty,
        "failed_pages": failed,
        "extraction_success_rate": round((extracted / pages) * 100, 2) if pages else 0.0,
    }


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run each test inside an isolated tmp working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _qdrant_state(collection):
    """Return (set of metadata.source values, point count) for a collection."""
    client = QdrantClient(path=get_collection_dir(collection))
    try:
        safe = sanitize_collection_name(collection)
        sources = set()
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=safe,
                limit=256,
                offset=offset,
                with_payload=["metadata.source"],
            )
            for p in points:
                sources.add(((p.payload or {}).get("metadata") or {}).get("source"))
            if offset is None:
                break
        count = client.count(collection_name=safe).count
    finally:
        client.close()
    return sources, count


# ---------------------------------------------------------------------------
# Fresh ingestion
# ---------------------------------------------------------------------------

def test_first_ingest_creates_both_indexes(workdir):
    emb = FakeEmbeddings()
    res = helpers.ingest_chunks_with_stats(
        _chunks("a.pdf", ["alpha one", "alpha two"]), [_stats("a.pdf", 2)], "col", emb
    )

    assert res["ok"] is True
    assert res["store_mode"] == "created"

    rows = load_chunk_corpus("col")
    assert len(rows) == 2

    sources, count = _qdrant_state("col")
    assert count == 2
    assert sources == {"a.pdf"}


# ---------------------------------------------------------------------------
# Re-ingestion of the SAME file replaces on both indexes
# ---------------------------------------------------------------------------

def test_reingest_same_file_replaces_everywhere(workdir):
    emb = FakeEmbeddings()
    helpers.ingest_chunks_with_stats(
        _chunks("a.pdf", ["old alpha one", "old alpha two"]), [_stats("a.pdf", 2)], "col", emb
    )
    res = helpers.ingest_chunks_with_stats(
        _chunks("a.pdf", ["new alpha one", "new alpha two", "new alpha three"]),
        [_stats("a.pdf", 3)],
        "col",
        emb,
    )

    assert res["store_mode"] == "appended"

    # Corpus: no duplicates, old text gone
    rows = load_chunk_corpus("col")
    assert len(rows) == 3
    assert {r["text"] for r in rows} == {"new alpha one", "new alpha two", "new alpha three"}

    # Vector store: same count, same sources (no duplicated points)
    sources, count = _qdrant_state("col")
    assert count == 3
    assert sources == {"a.pdf"}

    # BM25: old content is no longer retrievable ("old" only exists in the
    # replaced texts), new content is
    assert bm25_search_collection("col", "old", k=10) == []
    hits = bm25_search_collection("col", "new alpha", k=10)
    assert {h[0]["text"] for h in hits} == {"new alpha one", "new alpha two", "new alpha three"}


# ---------------------------------------------------------------------------
# Adding a NEW file keeps previous files
# ---------------------------------------------------------------------------

def test_new_file_appends_and_keeps_previous(workdir):
    emb = FakeEmbeddings()
    helpers.ingest_chunks_with_stats(
        _chunks("a.pdf", ["alpha"]), [_stats("a.pdf", 1)], "col", emb
    )
    helpers.ingest_chunks_with_stats(
        _chunks("b.pdf", ["bravo"]), [_stats("b.pdf", 1)], "col", emb
    )

    rows = load_chunk_corpus("col")
    assert len(rows) == 2

    sources, count = _qdrant_state("col")
    assert count == 2
    assert sources == {"a.pdf", "b.pdf"}

    hits = bm25_search_collection("col", "alpha", k=10)
    assert {h[0]["text"] for h in hits} == {"alpha"}


# ---------------------------------------------------------------------------
# Cross-index consistency after mixed batches
# ---------------------------------------------------------------------------

def test_indexes_stay_consistent_across_mixed_batches(workdir):
    emb = FakeEmbeddings()
    helpers.ingest_chunks_with_stats(
        _chunks("a.pdf", ["alpha", "another alpha"]), [_stats("a.pdf", 1)], "col", emb
    )
    helpers.ingest_chunks_with_stats(
        _chunks("b.pdf", ["bravo"]), [_stats("b.pdf", 1)], "col", emb
    )
    # Replace a.pdf with a single new chunk
    helpers.ingest_chunks_with_stats(
        _chunks("a.pdf", ["alpha v2"]), [_stats("a.pdf", 1)], "col", emb
    )

    rows = load_chunk_corpus("col")
    corpus_sources = {r["metadata"].get("source") for r in rows}
    qdrant_sources, count = _qdrant_state("col")

    # Both indexes agree on counts and on which sources exist
    assert len(rows) == 2
    assert count == 2
    assert corpus_sources == qdrant_sources == {"a.pdf", "b.pdf"}


# ---------------------------------------------------------------------------
# metadata.json merges per file
# ---------------------------------------------------------------------------

def test_metadata_merges_per_file(workdir):
    emb = FakeEmbeddings()
    helpers.ingest_chunks_with_stats(
        _chunks("a.pdf", ["alpha"]), [_stats("a.pdf", 1)], "col", emb
    )
    res = helpers.ingest_chunks_with_stats(
        _chunks("b.pdf", ["bravo", "bravo two"]), [_stats("b.pdf", 2)], "col", emb
    )

    meta = res["metadata"]
    assert meta["files"] == ["a.pdf", "b.pdf"]
    assert meta["total_pages"] == 3
    assert meta["total_chunks"] == 3
    assert meta["extraction_stats"]["overall"]["extracted_pages"] == 3

    # Re-ingesting a.pdf refreshes it in place and keeps b.pdf
    res2 = helpers.ingest_chunks_with_stats(
        _chunks("a.pdf", ["alpha v2"]), [_stats("a.pdf", 1)], "col", emb
    )
    meta2 = res2["metadata"]
    assert meta2["files"] == ["a.pdf", "b.pdf"]
    assert meta2["total_pages"] == 3
    assert meta2["total_chunks"] == 3

    # The persisted file matches the returned metadata
    persisted = helpers.load_collection_metadata("col")
    assert persisted["files"] == ["a.pdf", "b.pdf"]


# ---------------------------------------------------------------------------
# Legacy corpus rows without a source are never dropped
# ---------------------------------------------------------------------------

def test_legacy_corpus_rows_without_source_are_retained(workdir):
    emb = FakeEmbeddings()
    helpers.ingest_chunks_with_stats(
        _chunks("a.pdf", ["alpha"]), [_stats("a.pdf", 1)], "col", emb
    )

    # Simulate a legacy row written by an older version (no metadata.source)
    with open(get_chunk_corpus_path("col"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": "legacy row", "metadata": {"page_number": 9}}) + "\n")

    helpers.ingest_chunks_with_stats(
        _chunks("b.pdf", ["bravo"]), [_stats("b.pdf", 1)], "col", emb
    )

    texts = {r["text"] for r in load_chunk_corpus("col")}
    assert texts == {"alpha", "legacy row", "bravo"}

    # And the rebuilt BM25 index covers all merged rows, including the legacy one
    hits = bm25_search_collection("col", "legacy", k=10)
    assert {h[0]["text"] for h in hits} == {"legacy row"}
