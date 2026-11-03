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

For the same reason the selectors are created through ``collection_selector()``:
handing a default (``index=``) to a widget that already carries a value set by a
callback makes Streamlit warn on screen, and leaves two things claiming to be
the selection.
"""
import streamlit as st
from app.history_store import load_entries
from src.rag.prompt_registry import DEFAULT_PROMPT_VERSION


_ACTIVE_COLLECTION_KEY = "current_collection"

# Selectors whose displayed value must follow the active collection. They are
# rendered through ``collection_selector()`` so the widget's default (``index=``)
# never competes with the value a callback has already set.
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
    _seed_widgets(_DISPLAY_WIDGET_KEYS + _LIST_WIDGET_KEYS)


def forget_collection_pickers() -> None:
    """Point the inspect/delete pickers at the live active collection.

    Only valid from a callback (before those widgets are created in the run).
    """
    _seed_widgets(_LIST_WIDGET_KEYS)


def _seed_widgets(keys) -> None:
    """Give each widget key a live collection name.

    Assigning (not removing) is what keeps the browser in step: dropping a
    widget's state leaves the rendered element still showing the old value
    while the server has already re-created the widget from ``index=0`` — the
    picker then displayed a collection that had just been deleted. A valid
    value means the element and the value never disagree.
    """
    active = current_collection()
    for key in keys:
        st.session_state[key] = active


def sync_collection_from_widget(widget_key: str) -> None:
    """``on_change`` callback: the widget value becomes the active collection."""
    value = st.session_state.get(widget_key)
    if isinstance(value, str) and value:
        st.session_state[_ACTIVE_COLLECTION_KEY] = value


def collection_selector(label: str, options, key: str, **kwargs) -> str:
    """Render a collection selectbox whose stored value is the source of truth.

    ``index=`` is a *default*, not a binding: passing it on a run where the key
    already holds a value set through the Session State API — which is how
    ``choose_collection()`` switches collections from a callback — makes
    Streamlit render a warning box ("The widget with key … was created with a
    default value but also had its value set via the Session State API"). It
    shows once per server process, on the create that follows an earlier
    create or switch. A default and an explicit value cannot both be
    authoritative, so the default is supplied only on the render that has no
    stored value to honour.

    A stored value that is no longer among ``options`` (the collection was
    removed in another session) is dropped and re-seeded from the active
    collection, rather than leaving a name behind that nothing can select.
    """
    choices = list(options) or ["default"]
    if st.session_state.get(key) not in choices:
        st.session_state.pop(key, None)
        active = current_collection()
        kwargs["index"] = choices.index(active) if active in choices else 0
    return st.selectbox(label, choices, key=key, **kwargs)


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
