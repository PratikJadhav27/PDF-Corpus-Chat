"""
app/state.py

Centralised session state initialisation.
Call init_session_state() once at the top of main() before rendering any page.
"""
import streamlit as st
from src.rag.prompt_registry import DEFAULT_PROMPT_VERSION


def init_session_state() -> None:
    defaults = {
        "chat_history": [],
        "document_metadata": None,
        "current_collection": "default",
        "token_count": 0,
        "feedback_data": [],
        "eval_results": [],
        "retrieval_mode": "hybrid",
        "top_k": 4,
        "candidate_k": 12,
        "rerank_top_n": 4,
        "prompt_version": DEFAULT_PROMPT_VERSION,
        "prompt_type": DEFAULT_PROMPT_VERSION,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
