"""Regression tests for ISSUE-013 — the Usage & Cost panel always showed zero.

``app/state.py`` initialised ``session_state["token_count"] = 0`` and the
Collections tab turned it into two cards ("Total Tokens", "Est. Cost (USD)"),
but no code path ever wrote the key: every session showed ``0`` and
``$0.000000`` however many questions were asked. The pipeline also discarded
everything except ``.content``, so the provider's own accounting
(``AIMessage.usage_metadata``) never reached the app.

The contract under test: one real model call adds its reported tokens to the
session total, an unreported call adds nothing, and the cards show what was
actually spent.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from src.rag.usage import TokenUsage, accumulate_usage, usage_from_message

APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"

DOCS = [
    Document(
        page_content="Alpha applies to bravo.",
        metadata={"source": "a.pdf", "page_number": 1, "chunk_index": 0},
    )
]
ANSWER = "Alpha applies to bravo [a.pdf | page 1]."


# ---------------------------------------------------------------------------
# Unit tests — usage extraction
# ---------------------------------------------------------------------------

def test_usage_is_read_from_usage_metadata():
    reply = SimpleNamespace(
        usage_metadata={
            "input_tokens": 1200,
            "output_tokens": 300,
            "total_tokens": 1500,
        }
    )

    usage = usage_from_message(reply)

    assert usage == TokenUsage(
        prompt_tokens=1200, completion_tokens=300, total_tokens=1500
    )


def test_usage_falls_back_to_provider_response_metadata():
    """Integrations that do not populate ``usage_metadata`` still get counted."""
    reply = SimpleNamespace(
        usage_metadata=None,
        response_metadata={
            "token_usage": {"prompt_tokens": 700, "completion_tokens": 100}
        },
    )

    usage = usage_from_message(reply)

    assert usage == TokenUsage(
        prompt_tokens=700, completion_tokens=100, total_tokens=800
    )


def test_missing_usage_returns_none_not_zero():
    """A silent provider must be distinguishable from a free call."""
    assert usage_from_message(SimpleNamespace(content="hi")) is None
    assert usage_from_message(SimpleNamespace(usage_metadata={})) is None
    assert usage_from_message("plain string") is None


def test_total_is_derived_when_the_provider_omits_it():
    reply = SimpleNamespace(
        usage_metadata={"input_tokens": 40, "output_tokens": 2}
    )

    assert usage_from_message(reply).total_tokens == 42


def test_accumulate_adds_every_call():
    state = {}

    assert accumulate_usage(state, TokenUsage(1000, 200, 1200)) == 1200
    assert accumulate_usage(state, TokenUsage(300, 100, 400)) == 1600
    assert state["token_count"] == 1600


def test_unreported_usage_leaves_the_total_alone():
    state = {"token_count": 900}

    assert accumulate_usage(state, None) == 900
    assert accumulate_usage(state, TokenUsage(0, 0, 0)) == 900
    assert state["token_count"] == 900


def test_accumulate_accepts_an_attribute_style_session():
    """Streamlit's proxy is exercised as a namespace by the other tests."""
    state = SimpleNamespace(token_count=250)

    assert accumulate_usage(state, TokenUsage(50, 0, 50)) == 300
    assert state.token_count == 300


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------

def _install_pipeline_fakes(monkeypatch, reply):
    """Stub retrieval and the model so ``run_rag_query`` is hermetic."""

    class _Store:
        def similarity_search(self, query, k):
            return list(DOCS)

    class _Model:
        def __init__(self, model=None, temperature=None, **kwargs):
            pass

        def invoke(self, prompt, config=None):
            return reply

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
    monkeypatch.setattr("src.rag.inference.append_rag_trace", lambda record: None)
    monkeypatch.setattr("src.rag.inference.langfuse_flush", lambda: None)


def _session(monkeypatch):
    import streamlit as st

    monkeypatch.setattr(st, "session_state", SimpleNamespace())
    st.session_state.retrieval_mode = "similarity"
    st.session_state.prompt_version = "v1_strict"
    st.session_state.top_k = 4
    st.session_state.candidate_k = 12
    st.session_state.rerank_top_n = 4
    monkeypatch.setattr("streamlit.error", lambda msg: None)


def test_pipeline_records_the_tokens_of_a_successful_answer(monkeypatch):
    reply = SimpleNamespace(
        content=ANSWER,
        usage_metadata={
            "input_tokens": 1200,
            "output_tokens": 300,
            "total_tokens": 1500,
        },
    )
    _session(monkeypatch)
    _install_pipeline_fakes(monkeypatch, reply)

    import streamlit as st
    from src.rag.inference import run_rag_query

    answer, docs, latency = run_rag_query("What is alpha?", "v1_strict", 0.0, "c")

    assert answer
    assert st.session_state.token_count == 1500
    # The answer still travels as text; timing keys are unchanged.
    assert set(latency) == {"total_sec", "retrieval_sec", "llm_sec"}


def test_pipeline_accumulates_across_questions(monkeypatch):
    reply = SimpleNamespace(
        content=ANSWER,
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    )
    _session(monkeypatch)
    _install_pipeline_fakes(monkeypatch, reply)

    import streamlit as st
    from src.rag.inference import run_rag_query

    run_rag_query("What is alpha?", "v1_strict", 0.0, "c")
    run_rag_query("And bravo?", "v1_strict", 0.0, "c")

    assert st.session_state.token_count == 240


# ---------------------------------------------------------------------------
# App-level test — the cards the user reads
# ---------------------------------------------------------------------------

def test_usage_panel_shows_the_tokens_spent_in_the_session(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAT_LOG_DIR", str(tmp_path / "chat_log"))
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "0")
    monkeypatch.setenv("COST_PER_MILLION", "0.15")

    reply = SimpleNamespace(
        content=ANSWER,
        usage_metadata={
            "input_tokens": 1200,
            "output_tokens": 300,
            "total_tokens": 1500,
        },
    )
    _install_pipeline_fakes(monkeypatch, reply)

    at = AppTest.from_file(str(APP_MAIN), default_timeout=120)
    at.session_state["retrieval_mode"] = "similarity"
    at.run()
    assert not at.exception, at.exception

    # Nothing has been asked yet: the panel is honestly empty, not broken.
    before = {m.label: m.value for m in at.metric}
    assert before["Total Tokens"] == "0"

    at.chat_input[0].set_value("What is alpha?").run()
    assert not at.exception, at.exception

    assert at.session_state["token_count"] == 1500

    cards = {m.label: m.value for m in at.metric}
    assert cards["Total Tokens"] == "1,500"
    assert cards["Est. Cost (USD)"] == f"${1500 / 1_000_000 * 0.15:.6f}"
    assert cards["Est. Cost (USD)"] != "$0.000000"
