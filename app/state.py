"""
app/state.py

Centralised session state initialisation.
Call init_session_state() once at the top of main() before rendering any page.

Active collection
-----------------
``current_collection`` is the single source of truth: every upload, query and
statistics panel reads it. Selector widgets only *display* it and write it
through their ``on_change`` callback.

A code-driven switch (creating or deleting a collection) happens in an
``on_click`` callback, because Streamlit only allows widget state to be written
before the widget is created. Writing ``current_collection`` alone is not
enough: a selectbox ignores ``index=`` once its key holds a value, so the stale
selection would win on the next run and every later upload would land in the
wrong collection.
"""
import streamlit as st
from app.history_store import load_entries
from src.rag.prompt_registry import DEFAULT_PROMPT_VERSION


_ACTIVE_COLLECTION_KEY = "current_collection"

# Selectors whose displayed value must follow the active collection.
_DISPLAY_WIDGET_KEYS = (
    "sidebar_collection_select",
    "upload_target_col",
    "search_collection",
)

# Pickers whose option list can change when a collection appears/disappears.
_LIST_WIDGET_KEYS = (
    "stats_collection",
    "del_collection",
)


def current_collection() -> str:
    """The active collection name (never empty)."""
    return st.session_state.get(_ACTIVE_COLLECTION_KEY) or "default"


def set_current_collection(name: str) -> None:
    """Set the active collection without touching widget state."""
    st.session_state[_ACTIVE_COLLECTION_KEY] = name or "default"


def choose_collection(name: str) -> None:
    """Make ``name`` active and re-seed every collection widget to it.

    Only valid from a callback (``on_click``), i.e. before any widget has been
    created in the run.
    """
    set_current_collection(name)
    for key in _DISPLAY_WIDGET_KEYS:
        st.session_state[key] = current_collection()
    for key in _LIST_WIDGET_KEYS:
        st.session_state.pop(key, None)


def forget_collection_pickers() -> None:
    """Drop the inspect/delete pickers so they re-seed from the current list.

    Only valid from a callback (before those widgets are created in the run).
    """
    for key in _LIST_WIDGET_KEYS:
        st.session_state.pop(key, None)


def sync_collection_from_widget(widget_key: str) -> None:
    """``on_change`` callback: the widget value becomes the active collection."""
    value = st.session_state.get(widget_key)
    if isinstance(value, str) and value:
        st.session_state[_ACTIVE_COLLECTION_KEY] = value


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
