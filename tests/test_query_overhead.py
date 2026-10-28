"""
Regression tests for ISSUE-011 — per-query overheads.

Three costs were paid on **every** query:

1. ``get_embeddings()`` built a new embeddings object and performed a live HTTPS
   model-discovery request each time it was called (chat query, semantic search,
   ingestion).
2. ``load_vector_store()`` opened a new local ``QdrantClient`` per call. Local
   mode takes an exclusive lock on the storage folder, released only by GC.
3. ``bm25_search_collection()`` re-parsed the entire JSONL corpus per query,
   only to map hit indexes back to metadata.

The tests below assert the *work performed* (call counts), which is
deterministic, instead of wall-clock timings. Wall-clock measurements for the
same change are recorded in PROJECT_AUDIT.md.
"""
import json
import os

import pytest
from langchain_core.embeddings import Embeddings

import src.rag.bm25_index as bm25
import src.rag.embeddings as embeddings_mod
import src.rag.vector_store as vector_store
from src.rag.bm25_index import (
    bm25_search_collection,
    load_chunk_corpus,
    merge_chunk_corpus,
    save_chunk_corpus,
)
from src.rag.embeddings import clear_embeddings_cache, get_embeddings
from src.rag.vector_store import load_vector_store


class FakeEmbeddings(Embeddings):
    """Deterministic 3-dimension embeddings; subclasses langchain's ABC so the
    store validates it like the real ones."""

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


@pytest.fixture(autouse=True)
def _tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    clear_embeddings_cache()
    bm25.forget_corpus_cache()
    yield tmp_path
    clear_embeddings_cache()
    bm25.forget_corpus_cache()


def _chunks(source, texts):
    return [
        {"text": t, "metadata": {"source": source, "page_number": i + 1, "chunk_index": i}}
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# 1. Embeddings factory
# ---------------------------------------------------------------------------

def _patch_google(monkeypatch, calls):
    """Replace the Google embeddings class and the discovery HTTP call."""
    class FakeGoogle:
        def __init__(self, model):
            self.model = model
            calls["constructed"] += 1

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(
                {"models": [{"name": "models/text-embedding-005",
                            "supportedGenerationMethods": ["embedContent"]}]}
            ).encode()

    def fake_urlopen(request, timeout=None):
        calls["discovery"] += 1
        return FakeResponse()

    import urllib.request
    monkeypatch.setattr("langchain_google_genai.GoogleGenerativeAIEmbeddings", FakeGoogle)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_EMBEDDING_MODEL", raising=False)


def test_embeddings_are_built_and_discovered_once(monkeypatch):
    calls = {"constructed": 0, "discovery": 0}
    _patch_google(monkeypatch, calls)

    first = get_embeddings()
    for _ in range(4):
        assert get_embeddings() is first

    assert calls["constructed"] == 1
    assert calls["discovery"] == 1
    assert first.model == "models/text-embedding-005"


def test_failed_discovery_is_not_retried(monkeypatch):
    calls = {"constructed": 0}

    class FakeGoogle:
        def __init__(self, model):
            self.model = model
            calls["constructed"] += 1

    attempts = {"n": 0}

    def boom(request, timeout=None):
        attempts["n"] += 1
        raise OSError("network down")

    import urllib.request
    monkeypatch.setattr("langchain_google_genai.GoogleGenerativeAIEmbeddings", FakeGoogle)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_EMBEDDING_MODEL", raising=False)

    first = get_embeddings()
    for _ in range(3):
        get_embeddings()

    assert attempts["n"] == 1
    assert calls["constructed"] == 1
    assert first.model == "models/text-embedding-004"


def test_pinned_model_skips_discovery(monkeypatch):
    calls = {"constructed": 0, "discovery": 0}
    _patch_google(monkeypatch, calls)
    monkeypatch.setenv("GOOGLE_EMBEDDING_MODEL", "models/pinned-embedding")

    assert get_embeddings().model == "models/pinned-embedding"
    assert calls["discovery"] == 0


# ---------------------------------------------------------------------------
# 2. Qdrant client
# ---------------------------------------------------------------------------

def test_vector_store_calls_share_one_client():
    save_chunk_corpus("col", _chunks("doc.pdf", ["alpha bravo", "charlie delta"]))
    vector_store.build_vector_store_from_chunks(
        _chunks("doc.pdf", ["alpha bravo", "charlie delta"]),
        collection_name="col",
        embeddings=FakeEmbeddings(),
    )

    first = load_vector_store("col", FakeEmbeddings())
    second = load_vector_store("col", FakeEmbeddings())

    assert first is not None and second is not None
    assert first.client is second.client
    assert len(vector_store._CLIENTS) == 1


def test_delete_collection_closes_the_cached_client(tmp_path):
    from app import helpers

    closed = []

    class FakeClient:
        def close(self):
            closed.append(True)

    key = os.path.abspath(vector_store.get_collection_dir("gone"))
    vector_store._CLIENTS[key] = FakeClient()
    os.makedirs(os.path.join("vector_stores", "gone"), exist_ok=True)

    helpers.delete_collection("gone")

    assert closed == [True], "the cached client must be closed before rmtree"
    assert not os.path.exists(os.path.join("vector_stores", "gone"))


# ---------------------------------------------------------------------------
# 3. BM25 corpus
# ---------------------------------------------------------------------------

def test_corpus_is_parsed_once_across_queries(monkeypatch):
    rows = _chunks("doc.pdf", [f"alpha bravo chunk {i}" for i in range(6)])
    save_chunk_corpus("col", rows)
    bm25.forget_corpus_cache()

    parses = {"n": 0}
    real_loads = json.loads

    def counting_loads(*args, **kwargs):
        parses["n"] += 1
        return real_loads(*args, **kwargs)

    monkeypatch.setattr(bm25.json, "loads", counting_loads)

    for _ in range(3):
        hits = bm25_search_collection("col", "alpha bravo", k=3)
        assert hits

    assert parses["n"] == len(rows), "corpus must be parsed once, not per query"


def test_corpus_cache_picks_up_an_external_rewrite():
    save_chunk_corpus("col", _chunks("doc.pdf", ["alpha bravo"]))
    first = load_chunk_corpus("col")
    assert len(first) == 1

    # Another process rewrites the corpus and the file's mtime changes.
    path = bm25.get_chunk_corpus_path("col")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": "zulu yankee", "metadata": {"source": "new.pdf"}}) + "\n")
    stat = os.stat(path)
    os.utime(path, (stat.st_atime + 10, stat.st_mtime + 10))

    assert len(load_chunk_corpus("col")) == 2


def test_reingest_refreshes_the_cached_corpus():
    save_chunk_corpus("col", _chunks("doc.pdf", ["alpha bravo"]))
    assert bm25_search_collection("col", "alpha", k=5)

    merge_chunk_corpus("col", _chunks("doc.pdf", ["zulu yankee"]))
    bm25.forget_corpus_cache()

    assert not bm25_search_collection("col", "alpha", k=5)
    hits = bm25_search_collection("col", "zulu", k=5)
    assert hits and hits[0][0]["text"] == "zulu yankee"


def test_search_without_corpus_returns_empty():
    assert load_chunk_corpus("missing") == []
    assert bm25_search_collection("missing", "anything") == []
