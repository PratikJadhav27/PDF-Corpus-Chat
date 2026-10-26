"""
Regression tests for ISSUE-008 — Semantic Search listed chat-history collections.

The History tab built its search-collection dropdown from ``chat_history``
(`collections = sorted(set(c["collection"] for c in chat_history)) or ["default"]`)
instead of from the collections that exist on disk. Consequences:

* a fresh session (no chat yet) could only ever search ``default``, even though
  other indexed collections existed;
* worse, the History panel's empty state did a bare ``return`` from the whole
  ``render_history_tab()``, so with no chat history the Semantic Search tab was
  never rendered at all — no collection picker, no query box.

These tests drive the real entrypoint (``app/main.py``) through Streamlit's
``AppTest`` harness in a temporary working directory, so the "collections on
disk" are directories the test creates. No API key, vector store or network is
needed: the vector-store calls are faked.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


def _indexed(tmp_path, *names: str) -> None:
    """Create empty collection directories under ``vector_stores/``."""
    for name in names:
        (tmp_path / "vector_stores" / name).mkdir(parents=True, exist_ok=True)


def _history_entry(question: str, collection: str = "default") -> dict:
    return {
        "timestamp": "2026-10-25 10:00:00",
        "question": question,
        "answer": "Alpha applies to bravo [a.pdf | page 1].",
        "latency": {"total_sec": 0.4, "retrieval_sec": 0.1, "llm_sec": 0.3},
        "docs": [],
        "prompt_version": "v1_strict",
        "collection": collection,
    }


def _run(monkeypatch, tmp_path, history=None, current=None) -> AppTest:
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(APP_MAIN), default_timeout=120)
    if history is not None:
        at.session_state["chat_history"] = list(history)
    if current is not None:
        at.session_state["current_collection"] = current
    at.run()
    return at


def test_search_lists_indexed_collections_with_no_chat_history(monkeypatch, tmp_path):
    _indexed(tmp_path, "papers", "research")

    at = _run(monkeypatch, tmp_path, history=[])

    assert not at.exception, at.exception
    picker = at.selectbox(key="search_collection")
    assert list(picker.options) == ["papers", "research"]
    assert picker.value == "papers"

    # The History panel still shows its own empty state.
    body = "\n".join(m.value for m in at.markdown)
    assert "No conversation history yet" in body


def test_search_ignores_collections_that_only_appear_in_history(monkeypatch, tmp_path):
    _indexed(tmp_path, "papers")

    at = _run(
        monkeypatch,
        tmp_path,
        history=[_history_entry("What is alpha?", collection="deleted_collection")],
    )

    assert not at.exception, at.exception
    picker = at.selectbox(key="search_collection")
    assert list(picker.options) == ["papers"]
    assert "deleted_collection" not in list(picker.options)


def test_search_preselects_the_active_collection(monkeypatch, tmp_path):
    _indexed(tmp_path, "papers", "research")

    at = _run(monkeypatch, tmp_path, history=[], current="research")

    assert not at.exception, at.exception
    assert at.selectbox(key="search_collection").value == "research"


def test_search_shows_empty_state_when_nothing_is_indexed(monkeypatch, tmp_path):
    at = _run(monkeypatch, tmp_path, history=[])

    assert not at.exception, at.exception
    with pytest.raises(KeyError):
        at.selectbox(key="search_collection")

    infos = [i.value for i in at.info]
    assert any(
        "upload PDFs in the Collections tab to build one" in text for text in infos
    ), infos


def test_search_queries_the_collection_picked_from_disk(monkeypatch, tmp_path):
    """The dropdown value must reach the vector store, not just the widget."""
    from langchain_core.documents import Document

    import app.views.history as history_view

    calls = {}

    class _FakeStore:
        def similarity_search(self, query, k):
            calls["query"] = query
            calls["k"] = k
            return [Document(page_content="Revenue grew.", metadata={"source": "a.pdf", "page_number": 3})]

    def _fake_load(collection_name, embeddings):
        calls["collection"] = collection_name
        return _FakeStore()

    monkeypatch.setattr(history_view, "load_vector_store", _fake_load)
    monkeypatch.setattr(history_view, "get_embeddings", lambda: object())

    _indexed(tmp_path, "papers", "research")
    at = _run(monkeypatch, tmp_path, history=[], current="research")

    at.text_input(key="search_query").set_value("revenue")
    search = next(b for b in at.button if b.label == "🔍 Search")
    search.click().run()

    assert not at.exception, at.exception
    assert calls["collection"] == "research"
    assert calls["query"] == "revenue"
    body = "\n".join(m.value for m in at.markdown)
    assert "Result 1 · a.pdf" in body
    assert "Page 3" in body
