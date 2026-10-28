"""
Regression tests for the active-collection selection.

Before the fix three widgets wrote ``session_state.current_collection`` — the
sidebar selectbox, the Collections tab's "Target Collection" selectbox and the
create/delete buttons — and Streamlit's widgets ignore ``index=`` once their key
exists. Two consequences, both reproduced in a real browser session:

1. Creating a collection set ``current_collection`` and immediately reran, but
   the sidebar selectbox still held the previous value, so the switch was
   reverted on the next run. The UI showed the new collection in the badge while
   the very next upload was written into the *old* collection.
2. ``st.tabs`` renders every tab, so the Collections tab (last) always ran and
   its stale target value overwrote whatever the sidebar had just selected.
   The active collection therefore depended on render order, not on the user.

These tests drive the real entrypoint through Streamlit's AppTest harness inside
a tmp working directory, so no API key, Qdrant store or network is needed.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


def _app(tmp_path, monkeypatch, collections=()):
    for name in collections:
        (tmp_path / "vector_stores" / name).mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(APP_MAIN), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _widget(elements, key):
    for element in elements:
        if getattr(element, "key", None) == key:
            return element
    raise AssertionError(f"no widget with key {key!r}")


def _run(at):
    at.run()
    assert not at.exception, at.exception
    return at


def test_creating_a_collection_activates_it(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, collections=("default",))

    _widget(at.text_input, "new_collection_input").set_value("alpha")
    _run(at)
    _widget(at.button, "create_collection_btn").click()
    _run(at)

    assert at.session_state["current_collection"] == "alpha"
    assert _widget(at.selectbox, "sidebar_collection_select").value == "alpha"
    # the semantic-search picker follows the active collection instead of
    # falling back to whatever its widget state happened to hold
    assert _widget(at.selectbox, "search_collection").value == "alpha"


def test_collections_tab_does_not_clobber_the_active_collection(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, collections=("default", "alpha", "beta"))

    _widget(at.selectbox, "sidebar_collection_select").set_value("alpha")
    _run(at)

    # The Collections tab renders on every run (st.tabs builds all children);
    # seed the value an earlier render of its target selector would leave
    # behind. Before the fix that stale value became the active collection.
    at.session_state["upload_target_col"] = "beta"
    _run(at)

    assert at.session_state["current_collection"] == "alpha"
    assert _widget(at.selectbox, "sidebar_collection_select").value == "alpha"


def test_sidebar_selection_updates_the_active_collection(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, collections=("default", "alpha", "beta"))

    _widget(at.selectbox, "sidebar_collection_select").set_value("beta")
    _run(at)

    assert at.session_state["current_collection"] == "beta"


def test_deleting_the_active_collection_falls_back_to_a_survivor(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, collections=("default", "alpha", "beta"))

    _widget(at.selectbox, "sidebar_collection_select").set_value("alpha")
    _run(at)
    # Both the inspect/delete pickers and the upload target still point at the
    # collection that is about to disappear.
    at.session_state["del_collection"] = "alpha"
    at.session_state["upload_target_col"] = "alpha"
    _run(at)

    _widget(at.button, "delete_col_btn").click()
    _run(at)

    assert not (tmp_path / "vector_stores" / "alpha").exists()
    active = at.session_state["current_collection"]
    assert active != "alpha"
    assert (tmp_path / "vector_stores" / active).is_dir(), active
    assert _widget(at.selectbox, "sidebar_collection_select").value == active


def test_active_collection_survives_a_noop_rerun(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, collections=("default", "alpha"))

    _widget(at.selectbox, "sidebar_collection_select").set_value("alpha")
    _run(at)
    _run(at)
    _run(at)

    assert at.session_state["current_collection"] == "alpha"
