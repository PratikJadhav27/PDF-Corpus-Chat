"""
Regression tests for ISSUE-004 — the Prompt Style selector was a no-op.

The sidebar wrote ``session_state["prompt_type"]`` while ``run_rag_query``
built prompts (and logged traces) from ``session_state["prompt_version"]``.
Nothing in the app ever wrote ``prompt_version`` after start-up, so selecting
a style changed neither the prompt nor the answer, and every trace recorded a
``prompt_style`` that was never sent to the model.

The contract under test: the value the sidebar selects is the value that
reaches the model, the trace and the persisted chat entry — one key
(``prompt_version``) end to end.
"""

from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"

DOCS = [
    Document(
        page_content="Alpha applies to bravo.",
        metadata={"source": "a.pdf", "page_number": 1, "chunk_index": 0},
    )
]
ANSWER = "Alpha applies to bravo [a.pdf | page 1]."

# Distinctive markers from src/rag/prompt_registry.py
V1_MARKER = "strict document-grounded assistant"
V2_MARKER = "1) Use only context."


# ---------------------------------------------------------------------------
# Unit tests — the pipeline itself
# ---------------------------------------------------------------------------

def _install_pipeline_fakes(monkeypatch, captured: dict, records: list) -> None:
    """Stub retrieval, the LLM and tracing so ``run_rag_query`` is hermetic."""

    class _Store:
        def similarity_search(self, query, k):
            return list(DOCS)

    class _Model:
        def __init__(self, model=None, temperature=None, **kwargs):
            pass

        def invoke(self, prompt, config=None):
            captured["prompt"] = prompt
            return SimpleNamespace(content=ANSWER)

    monkeypatch.setattr(
        "src.rag.inference.load_vector_store",
        lambda collection, embeddings: _Store(),
    )
    monkeypatch.setattr("src.rag.inference.get_embeddings", lambda: None)
    monkeypatch.setattr(
        "src.rag.inference.retrieve_docs",
        lambda db, query, mode, k: list(DOCS),
    )
    monkeypatch.setattr("src.rag.inference.ChatGoogleGenerativeAI", _Model)
    monkeypatch.setattr("src.rag.inference.append_rag_trace", records.append)
    monkeypatch.setattr("src.rag.inference.langfuse_flush", lambda: None)


def _session_state(monkeypatch, prompt_version: str = "v1_strict"):
    """Stale session: the OLD selector key points at the selected style while
    ``prompt_version`` still holds the start-up default."""
    import streamlit as st

    monkeypatch.setattr(st, "session_state", SimpleNamespace())
    st.session_state.retrieval_mode = "similarity"
    st.session_state.prompt_version = prompt_version
    st.session_state.prompt_type = "v2_concise"
    st.session_state.top_k = 4
    st.session_state.candidate_k = 12
    st.session_state.rerank_top_n = 4
    monkeypatch.setattr("streamlit.error", lambda msg: None)


def test_selected_prompt_version_reaches_the_model(monkeypatch):
    captured: dict = {}
    records: list = []
    _session_state(monkeypatch)
    _install_pipeline_fakes(monkeypatch, captured, records)

    from src.rag.inference import run_rag_query

    # Same positional call the chat view makes.
    run_rag_query("What is alpha?", "v2_concise", 0.0, "test_collection")

    assert "prompt" in captured, "the model was never invoked"
    assert V2_MARKER in captured["prompt"], captured["prompt"]
    assert V1_MARKER not in captured["prompt"], captured["prompt"]


def test_trace_records_the_version_that_was_actually_used(monkeypatch):
    captured: dict = {}
    records: list = []
    _session_state(monkeypatch)
    _install_pipeline_fakes(monkeypatch, captured, records)

    from src.rag.inference import run_rag_query

    run_rag_query("What is alpha?", "v2_concise", 0.0, "test_collection")

    assert records, "no trace was written"
    record = records[-1]
    assert record["prompt_version"] == "v2_concise"
    # The old code logged a second, unused style — nothing may contradict the
    # version that actually shaped the prompt.
    assert "prompt_style" not in record


def test_legacy_version_is_logged_as_the_one_that_ran(monkeypatch):
    """``standard`` no longer exists; the fallback must be logged honestly."""
    captured: dict = {}
    records: list = []
    _session_state(monkeypatch)
    _install_pipeline_fakes(monkeypatch, captured, records)

    from src.rag.inference import run_rag_query

    run_rag_query("What is alpha?", "standard", 0.0, "test_collection")

    assert V1_MARKER in captured["prompt"]
    assert records[-1]["prompt_version"] == "v1_strict"
    assert "prompt_style" not in records[-1]


# ---------------------------------------------------------------------------
# App-level tests — the sidebar → pipeline wiring
# ---------------------------------------------------------------------------

def _fake_rag_query(seen: dict):
    def _run(question, prompt_version, temperature, collection_name):
        seen["prompt_version"] = prompt_version
        return ANSWER, list(DOCS), {
            "total_sec": 0.4, "retrieval_sec": 0.1, "llm_sec": 0.3,
        }

    return _run


def test_sidebar_selection_reaches_the_pipeline(monkeypatch):
    import app.views.chat as chat_view

    seen: dict = {}
    monkeypatch.setattr(chat_view, "run_rag_query", _fake_rag_query(seen))

    at = AppTest.from_file(str(APP_MAIN), default_timeout=120)
    at.run()
    assert not at.exception, at.exception

    # One key only: the duplicate selector key must be gone.
    assert "prompt_type" not in at.session_state

    at.selectbox(key="sidebar_prompt_version").set_value("v2_concise").run()
    assert not at.exception, at.exception

    at.chat_input[0].set_value("What is alpha?").run()
    assert not at.exception, at.exception

    assert seen["prompt_version"] == "v2_concise"
    assert at.session_state["prompt_version"] == "v2_concise"

    history = at.session_state["chat_history"]
    assert len(history) == 1
    assert history[0]["prompt_version"] == "v2_concise"
    assert "prompt_type" not in history[0]


def test_ui_shows_the_active_prompt_version(monkeypatch):
    import app.views.chat as chat_view

    monkeypatch.setattr(chat_view, "run_rag_query", _fake_rag_query({}))

    at = AppTest.from_file(str(APP_MAIN), default_timeout=120)
    at.run()
    at.selectbox(key="sidebar_prompt_version").set_value("v2_concise").run()
    at.chat_input[0].set_value("What is alpha?").run()
    assert not at.exception, at.exception

    captions = [c.value for c in at.caption]
    assert any("Active prompt: v2_concise" in c for c in captions), captions

    badges = [m.value for m in at.markdown if "dm-badge" in m.value]
    assert any("v2_concise" in b for b in badges), badges

    status_bar = [m.value for m in at.markdown if "dm-status-bar" in m.value]
    assert any("v2_concise" in s for s in status_bar), status_bar


def test_history_tab_shows_the_prompt_version_used(monkeypatch):
    """History lives in the official record: a new-style entry must render its
    version even though old entries only carry ``prompt_type``."""
    import app.views.chat as chat_view

    monkeypatch.setattr(chat_view, "run_rag_query", _fake_rag_query({}))

    at = AppTest.from_file(str(APP_MAIN), default_timeout=120)
    at.session_state["chat_history"] = [{
        "timestamp": "2026-10-19 10:00:00",
        "question": "What is alpha?",
        "answer": ANSWER,
        "latency": {"total_sec": 0.4, "retrieval_sec": 0.1, "llm_sec": 0.3},
        "docs": [],
        "prompt_version": "v2_concise",
        "collection": "default",
    }]
    at.run()
    assert not at.exception, at.exception

    rendered = "\n".join(m.value for m in at.markdown)
    assert "v2_concise" in rendered
    assert "standard" not in rendered
