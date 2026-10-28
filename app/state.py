"""
app/state.py

Centralised session state initialisation.
Call init_session_state() once at the top of main() before rendering any page.
"""
import streamlit as st
from app.history_store import load_entries
from src.rag.prompt_registry import DEFAULT_PROMPT_VERSION


def init_session_state() -> None:
    defaults = {
        "chat_history": [],
        "document_metadata": None,
        "current_collection": "default",
        "token_count": 0,
        "eval_results": [],
        "retrieval_mode": "hybrid",
        "top_k": 4,
        "candidate_k": 12,
        "rerank_top_n": 4,
        "prompt_version": DEFAULT_PROMPT_VERSION,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ISSUE-009: a fresh session starts with an empty log — restore the turns
    # that were saved to disk. A non-empty history (e.g. seeded by a test, or
    # kept across a rerun) is never overwritten. Feedback lives entirely on
    # disk and is read where it is rendered, so it needs no session default.
    if not st.session_state.get("chat_history"):
        st.session_state["chat_history"] = load_entries()
