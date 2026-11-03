"""
Regression tests for ISSUE-016: the collection selectors fought their own state.

The three display selectors (sidebar, upload target, semantic search) passed
``index=`` derived from the active collection *and* carried a key that
``choose_collection()`` seeds from an ``on_click`` callback. Streamlit treats
``index=`` as a default, so once a run both seeded the key and handed the widget
a default, the app rendered a warning box that reads like an error:

    The widget with key "sidebar_collection_select" was created with a default
    value but also had its value set via the Session State API.

It appeared the moment a second collection was created (or after switching, then
creating) and it advertised two competing sources of truth for one widget.
``collection_selector()`` supplies the default only on the render that has no
stored value to honour.

These tests drive the real entrypoint through Streamlit's AppTest harness.
"""

import shutil
from pathlib import Path

import pytest
from streamlit.elements.lib import policies
from streamlit.testing.v1 import AppTest

APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"

DISPLAY_SELECTORS = (
    "sidebar_collection_select",
    "upload_target_col",
    "search_collection",
)


@pytest.fixture(autouse=True)
def _warning_visible():
    """Streamlit shows the default-value warning once per process.

    Resetting the flag makes the first offender in every run visible, so a
    selector that is still wrong cannot hide behind the one-shot suppression.
    """
    previous = policies._shown_default_value_warning
    policies._shown_default_value_warning = False
    yield
    policies._shown_default_value_warning = previous


def _app(tmp_path, monkeypatch, collections=()):
    for name in collections:
        (tmp_path / "vector_stores" / name).mkdir(parents=True, exist_ok=True)
        (tmp_path / "metadata" / name).mkdir(parents=True, exist_ok=True)
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


def _default_value_warnings(at):
    return [
        w.value for w in at.warning if "created with a default value" in str(w.value)
    ]


def _create_from_sidebar(at, name):
    _widget(at.text_input, "new_collection_input").set_value(name)
    _run(at)
    _widget(at.button, "create_collection_btn").click()
    return _run(at)


def _create_from_tab(at, name):
    _widget(at.text_input, "new_col_name_main").set_value(name)
    _run(at)
    _widget(at.button, "create_col_main").click()
    return _run(at)


def test_creating_a_second_collection_does_not_warn(tmp_path, monkeypatch):
    """The reported symptom: the create that follows a create warns."""
    at = _app(tmp_path, monkeypatch, collections=("default",))

    _create_from_sidebar(at, "alpha")
    assert _default_value_warnings(at) == []

    policies._shown_default_value_warning = False
    _create_from_sidebar(at, "beta")

    assert _default_value_warnings(at) == []
    # the fix must not undo ISSUE-014: the created collection is selected
    assert at.session_state["current_collection"] == "beta"
    for key in DISPLAY_SELECTORS:
        assert _widget(at.selectbox, key).value == "beta", key


def test_switching_then_creating_does_not_warn(tmp_path, monkeypatch):
    """A user-driven switch followed by a code-driven one, same session."""
    at = _app(tmp_path, monkeypatch, collections=("default", "alpha"))

    _widget(at.selectbox, "sidebar_collection_select").set_value("alpha")
    _run(at)
    _create_from_sidebar(at, "gamma")

    assert _default_value_warnings(at) == []
    assert _widget(at.selectbox, "sidebar_collection_select").value == "gamma"


def test_creating_from_the_collections_tab_does_not_warn(tmp_path, monkeypatch):
    """Covers the upload-target selector, end to end."""
    at = _app(tmp_path, monkeypatch, collections=("default",))

    _create_from_sidebar(at, "alpha")
    policies._shown_default_value_warning = False
    _create_from_tab(at, "beta")

    assert _default_value_warnings(at) == []
    assert at.session_state["current_collection"] == "beta"
    assert _widget(at.selectbox, "upload_target_col").value == "beta"


def test_seeded_selector_is_not_given_a_default(tmp_path):
    """The contract: a key that already holds a value gets no ``index=``.

    This is the deterministic form of the defect — the script seeds the widget
    key before the widget exists, which is exactly what a callback does, and
    Streamlit reports the conflict only when a default is supplied as well.
    """
    script = """
import streamlit as st
from app.state import collection_selector

st.session_state.setdefault("current_collection", "alpha")
st.session_state["probe_select"] = "alpha"   # what choose_collection() writes
collection_selector("Pick", ["default", "alpha"], "probe_select")
"""
    at = AppTest.from_string(script, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert _default_value_warnings(at) == []
    assert _widget(at.selectbox, "probe_select").value == "alpha"


def test_first_render_still_follows_the_active_collection(tmp_path):
    """With no stored value the selector seeds from ``current_collection``."""
    script = """
import streamlit as st
from app.state import collection_selector

st.session_state.setdefault("current_collection", "beta")
rendered = collection_selector("Pick", ["default", "alpha", "beta"], "probe_select")
st.write(f"picked={rendered}")
"""
    at = AppTest.from_string(script, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert _widget(at.selectbox, "probe_select").value == "beta"


def test_switching_collection_points_the_pickers_at_it(tmp_path):
    """The pickers must be *assigned* a live name, never dropped.

    Removing a widget's key leaves the browser still showing the old value
    while the server re-creates the widget from ``index=0``: in a real session
    the delete picker went on displaying the collection that had just been
    deleted, next to a warning naming a different one.
    """
    script = """
import streamlit as st
from app.state import choose_collection

st.session_state.setdefault("current_collection", "default")
if "switched" not in st.session_state:
    st.session_state["switched"] = True
    st.session_state["del_collection"] = "beta"
    st.session_state["stats_collection"] = "beta"
    choose_collection("alpha")      # what the create/delete callbacks do

st.write("del=" + str(st.session_state.get("del_collection")))
st.write("stats=" + str(st.session_state.get("stats_collection")))
"""
    at = AppTest.from_string(script, default_timeout=60)
    at.run()
    assert not at.exception, at.exception

    rendered = "\n".join(str(m.value) for m in at.markdown)
    assert "del=alpha" in rendered, rendered
    assert "stats=alpha" in rendered, rendered


def test_selectors_reseed_when_their_collection_vanishes(tmp_path, monkeypatch):
    """A collection deleted outside this session must not stay in a selector."""
    at = _app(tmp_path, monkeypatch, collections=("default", "alpha", "beta"))

    _widget(at.selectbox, "sidebar_collection_select").set_value("alpha")
    _run(at)
    for key in ("search_collection", "upload_target_col"):
        at.session_state[key] = "alpha"
    _run(at)

    shutil.rmtree(tmp_path / "vector_stores" / "alpha")
    shutil.rmtree(tmp_path / "metadata" / "alpha")
    live = {"default", "beta"}
    _run(at)

    for key in DISPLAY_SELECTORS:
        assert at.session_state[key] in live, key
        assert _widget(at.selectbox, key).value in live, key


def test_delete_leaves_every_picker_on_a_live_collection(tmp_path, monkeypatch):
    """Guard for the pickers that never pass ``index`` at all."""
    at = _app(tmp_path, monkeypatch, collections=("default", "alpha", "beta"))

    _widget(at.selectbox, "sidebar_collection_select").set_value("alpha")
    _run(at)
    at.session_state["del_collection"] = "alpha"
    _run(at)

    _widget(at.button, "delete_col_btn").click()
    _run(at)

    assert not (tmp_path / "vector_stores" / "alpha").exists()
    for key in (
        "sidebar_collection_select",
        "upload_target_col",
        "search_collection",
        "stats_collection",
        "del_collection",
    ):
        picker = _widget(at.selectbox, key)
        assert picker.value in set(picker.options), key
        assert picker.value != "alpha", key
