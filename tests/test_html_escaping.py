"""
Regression tests for the HTML-escaping half of ISSUE-010.

Collection names (which can come from a directory that already exists on disk)
and PDF filenames were interpolated straight into ``unsafe_allow_html`` strings,
so a name like ``<img src=x onerror=…>`` became live markup in the status bar,
the source cards and the history badges. Semantic-search result cards also
embedded raw chunk text, i.e. arbitrary PDF content.

Every sink must render such data as text.
"""
from pathlib import Path

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from app.views.chat import _process_citations, _refs_html
from app.views.history import _search_result_html

APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"

PAYLOAD = "<img src=x onerror=alert(1)>"
ESCAPED = "&lt;img src=x onerror=alert(1)&gt;"


def _render(tmp_path, monkeypatch, **session_state) -> AppTest:
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(APP_MAIN), default_timeout=120)
    for key, value in session_state.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, at.exception
    return at


def _bodies(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def _entry(**overrides) -> dict:
    entry = {
        "id": "entry-1",
        "timestamp": "2026-10-20 10:00:00",
        "question": "What is alpha?",
        "answer": "Alpha applies to bravo.",
        "latency": {"total_sec": 0.4, "retrieval_sec": 0.1, "llm_sec": 0.3},
        "docs": [],
        "prompt_version": "v1_strict",
        "collection": "default",
    }
    entry.update(overrides)
    return entry


# ---------------------------------------------------------------------------
# Pure renderers
# ---------------------------------------------------------------------------

def test_refs_html_escapes_filename_and_page():
    out = _refs_html([(1, f"{PAYLOAD}.pdf", "page <b>2</b>", "")])

    assert PAYLOAD not in out
    assert ESCAPED in out
    assert "&lt;b&gt;2&lt;/b&gt;" in out


def test_citation_tooltip_cannot_break_out_of_its_attribute():
    docs = [
        Document(
            page_content='He said "hi" <b>loudly</b>',
            metadata={"source": f"{PAYLOAD}.pdf", "page_number": 2},
        )
    ]
    answer = f"Alpha [{PAYLOAD}.pdf | page 2] applies."

    out, citations = _process_citations(answer, docs)

    assert 'data-tooltip="' in out
    assert PAYLOAD not in out
    # escaped exactly once: the tooltip must not show a literal "&lt;"
    assert "&amp;lt;" not in out
    # the citation list keeps the raw name so downstream renderers escape once
    assert citations[0][1] == f"{PAYLOAD}.pdf"
    # and the reference list renders it safely
    assert PAYLOAD not in _refs_html(citations)


def test_search_result_card_escapes_pdf_text_and_filename():
    card = _search_result_html(1, f"{PAYLOAD}.pdf", 3, "<script>alert(1)</script>")

    assert "<script>" not in card
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in card
    assert ESCAPED in card


# ---------------------------------------------------------------------------
# Rendered app (the sinks the audit named)
# ---------------------------------------------------------------------------

def _status_bar_script(collection: str) -> None:
    """Render only the index status bar, with an arbitrary active collection."""
    import streamlit as st

    from app.views.chat import _index_status_html

    st.session_state["current_collection"] = collection
    st.markdown(_index_status_html(collection), unsafe_allow_html=True)


def test_index_status_bar_escapes_the_collection_name():
    # `<`, `>` are legal in directory names on Linux (where the Docker image
    # runs), so a store directory can genuinely carry this.
    at = AppTest.from_function(
        _status_bar_script, kwargs={"collection": PAYLOAD}, default_timeout=60
    )
    at.run()
    assert not at.exception, at.exception

    bodies = _bodies(at)

    assert PAYLOAD not in bodies
    assert f'</span>{ESCAPED}' in bodies


def test_active_collection_from_disk_is_escaped(tmp_path, monkeypatch):
    """A real, legal directory name with an ampersand must not become markup."""
    (tmp_path / "vector_stores" / "R&D papers").mkdir(parents=True)

    at = _render(tmp_path, monkeypatch)

    bodies = _bodies(at)

    assert "R&amp;D papers" in bodies
    assert "R&D papers" not in bodies


def test_source_card_and_reference_list_escape_the_filename(tmp_path, monkeypatch):
    doc = Document(
        page_content="Alpha body text",
        metadata={"source": f"{PAYLOAD}.pdf", "page_number": 2},
    )
    at = _render(
        tmp_path,
        monkeypatch,
        chat_history=[_entry(answer=f"Alpha [{PAYLOAD}.pdf | page 2] applies.", docs=[doc])],
    )

    bodies = _bodies(at)

    assert f'title="{PAYLOAD}' not in bodies            # source card attribute
    assert f'dm-ref-file">{PAYLOAD}' not in bodies      # numbered reference list
    assert f'title="{ESCAPED}.pdf"' in bodies
    assert f'dm-ref-file">{ESCAPED}.pdf</span>' in bodies


def test_history_badges_escape_stored_values(tmp_path, monkeypatch):
    at = _render(
        tmp_path,
        monkeypatch,
        chat_history=[_entry(collection=PAYLOAD, prompt_version=PAYLOAD)],
    )

    bodies = _bodies(at)

    assert f'dm-badge-violet">{PAYLOAD}' not in bodies   # history: collection
    assert f'dm-badge-amber">{PAYLOAD}' not in bodies    # history: prompt version
    assert f'dm-badge-green">{PAYLOAD}' not in bodies    # chat: latency badge
    assert f'dm-badge-violet">{ESCAPED}</span>' in bodies
    assert f'dm-badge-amber">{ESCAPED}</span>' in bodies
    assert f'dm-badge-green">{ESCAPED}</span>' in bodies
