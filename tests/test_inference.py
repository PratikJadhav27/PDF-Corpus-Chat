"""
Smoke test for the core RAG inference pipeline (run_rag_query).

This test verifies that when the vector store returns no documents,
the function gracefully returns the expected "I don't know..." response
and an empty docs list. It mocks out all external dependencies
(Qdrant, embeddings, Streamlit) so it runs in CI without any API keys.
"""

import pytest
from types import SimpleNamespace


def test_run_rag_query_no_docs(monkeypatch):
    """run_rag_query should return the fallback answer when no docs are retrieved."""
    import streamlit as st

    # Prepare minimal Streamlit session_state using SimpleNamespace for attribute access
    st.session_state = SimpleNamespace()
    st.session_state.retrieval_mode = "similarity"
    st.session_state.prompt_version = "v1_strict"
    st.session_state.top_k = 4
    st.session_state.candidate_k = 12
    st.session_state.rerank_top_n = 4

    # Mock the vector store to return an empty list of docs
    class MockStore:
        def similarity_search(self, query, k):
            return []

    # Patch the vector store loader to return our mock
    monkeypatch.setattr(
        "src.rag.vector_store.load_vector_store",
        lambda collection, embeddings: MockStore()
    )

    # Embeddings are not used when the store is mocked, but the function calls it.
    monkeypatch.setattr(
        "src.rag.inference.get_embeddings",
        lambda: None
    )

    # Silence Streamlit's error calls (they would raise in a headless test)
    monkeypatch.setattr("streamlit.error", lambda msg: None)

    # Now import and run the function under test
    from src.rag.inference import run_rag_query

    answer, docs, latency = run_rag_query(
        user_question="What is the revenue?",
        prompt_type="standard",
        temperature=0.0,
        collection_name="test_collection"
    )

    assert answer == "I don't know based on the provided documents."
    assert docs == []
    assert isinstance(latency, dict)
    assert "total_sec" in latency
