"""
Regression tests for ISSUE-010 — collection names and path traversal.

A collection name used to be sanitised with ``strip().replace(" ", "_")`` only,
yet it flowed straight into ``os.makedirs``, ``shutil.rmtree`` and the Qdrant
storage path. A name like ``../../elsewhere`` therefore escaped
``vector_stores/`` and could create or delete arbitrary directories.

These tests pin the rules:
  - only ``[A-Za-z0-9_-]`` survives in a new name;
  - a name that already exists on disk keeps working verbatim (legacy dirs);
  - no path produced for a collection ever leaves its base directory.
"""
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.helpers import create_collection, delete_collection, load_collection_metadata
from src.rag.collection_names import (
    collection_path,
    resolve_collection_name,
    sanitize_collection_name,
)
from src.rag.vector_store import get_collection_dir

APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run inside an isolated tmp working directory (all paths are relative)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("default", "default"),
        ("my collection", "my_collection"),          # unchanged legacy behaviour
        ("my-research_2024", "my-research_2024"),
        ("../etc", "etc"),
        ("..\\..\\windows\\system32", "windows_system32"),
        ("a/b/c", "a_b_c"),
        ("/absolute", "absolute"),
        ("C:\\temp", "C_temp"),
        ("<script>x</script>", "script_x_script"),
        ("naïve", "na_ve"),
        ("   ", "default"),
        ("", "default"),
    ],
)
def test_sanitize_collection_name_is_an_allowlist(raw, expected):
    assert sanitize_collection_name(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["../escape", "..\\escape", "../../../../etc", "/abs", "C:\\abs", "a/../../b"],
)
def test_collection_paths_never_leave_the_store(workdir, raw):
    for base in ("vector_stores", "metadata"):
        path = collection_path(base, raw)
        assert os.path.commonpath(
            [os.path.abspath(base), os.path.abspath(path)]
        ) == os.path.abspath(base)
    assert os.path.commonpath(
        [os.path.abspath("vector_stores"), os.path.abspath(get_collection_dir(raw))]
    ) == os.path.abspath("vector_stores")


def test_create_collection_keeps_directories_inside_the_trees(workdir):
    (workdir / "victim").mkdir()  # a sibling of vector_stores/, must stay untouched

    safe = create_collection("../../victim")

    assert safe == "victim"
    assert (workdir / "vector_stores" / "victim").is_dir()
    assert (workdir / "metadata" / "victim").is_dir()
    assert sorted(p.name for p in workdir.iterdir()) == ["metadata", "vector_stores", "victim"]


def test_delete_collection_cannot_reach_outside_the_store(workdir):
    victim = workdir / "victim"
    victim.mkdir()
    (victim / "important.txt").write_text("keep me", encoding="utf-8")
    (workdir / "vector_stores" / "keep").mkdir(parents=True)

    delete_collection("../victim")

    assert (victim / "important.txt").read_text(encoding="utf-8") == "keep me"
    assert (workdir / "vector_stores" / "keep").is_dir()


def test_delete_collection_still_deletes_the_named_collection(workdir):
    create_collection("papers")
    load_target = workdir / "metadata" / "papers" / "metadata.json"
    load_target.write_text('{"files": ["a.pdf"]}', encoding="utf-8")

    assert load_collection_metadata("papers") == {"files": ["a.pdf"]}
    delete_collection("papers")

    assert not (workdir / "vector_stores" / "papers").exists()
    assert not (workdir / "metadata" / "papers").exists()


def test_legacy_directory_name_is_kept_verbatim(workdir):
    """Names created by an older build (e.g. with a dot) must keep working.

    Sanitising ``paper.v2`` to ``paper_v2`` would point at a directory that does
    not exist, so the collection would silently disappear — and deleting it
    could hit a *different* collection that happens to own the sanitised name.
    """
    legacy_store = workdir / "vector_stores" / "paper.v2"
    legacy_store.mkdir(parents=True)
    legacy_meta = workdir / "metadata" / "paper.v2"
    legacy_meta.mkdir(parents=True)
    (legacy_meta / "metadata.json").write_text('{"files": ["a.pdf"]}', encoding="utf-8")

    assert resolve_collection_name("paper.v2") == "paper.v2"
    assert get_collection_dir("paper.v2") == os.path.join("vector_stores", "paper.v2")
    assert collection_path("metadata", "paper.v2") == os.path.join("metadata", "paper.v2")
    assert load_collection_metadata("paper.v2") == {"files": ["a.pdf"]}

    delete_collection("paper.v2")

    assert not legacy_store.exists()
    assert not legacy_meta.exists()


def test_legacy_name_does_not_delete_the_sanitised_sibling(workdir):
    """`paper.v2` must not be redirected onto an unrelated `paper_v2`."""
    (workdir / "vector_stores" / "paper.v2").mkdir(parents=True)
    other = workdir / "vector_stores" / "paper_v2"
    other.mkdir(parents=True)

    delete_collection("paper.v2")

    assert other.is_dir()
    assert not (workdir / "vector_stores" / "paper.v2").exists()


def _click_create(field_key: str, button_key: str, typed: str) -> AppTest:
    at = AppTest.from_file(str(APP_MAIN), default_timeout=120)
    at.run()
    [field] = [w for w in at.text_input if w.key == field_key]
    field.set_value(typed)
    at.run()  # the create button only renders once the field is non-empty
    [button] = [b for b in at.button if b.key == button_key]
    button.click()
    at.run()
    assert not at.exception, at.exception
    return at


def test_sidebar_create_form_sanitizes_the_typed_name(workdir):
    _click_create("new_collection_input", "create_collection_btn", "../../evil")

    assert (workdir / "vector_stores" / "evil").is_dir()
    assert (workdir / "metadata" / "evil").is_dir()
    assert not (workdir / "evil").exists()


def test_collections_tab_create_form_sanitizes_the_typed_name(workdir):
    _click_create("new_col_name_main", "create_col_main", "../Config")

    assert (workdir / "vector_stores" / "Config").is_dir()
    assert not (workdir / "Config").exists()


def test_sanitized_name_never_contains_a_separator(workdir):
    for raw in ["../x", "..\\x", "a/b", "C:\\x", "\\\\server\\share"]:
        safe = sanitize_collection_name(raw)
        assert "/" not in safe and "\\" not in safe
        assert safe not in {"", ".", ".."}
