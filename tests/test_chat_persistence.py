"""
tests/test_chat_persistence.py

End-to-end checks for ISSUE-009 through the real entrypoint (``app/main.py``).

Before the fix, 👍/👎 appended to a session-state list nothing ever read, and
chat history vanished on reload. These tests seed / inspect the on-disk log
(redirected to a temp dir by ``tests/conftest.py``) and drive the app with
Streamlit's ``AppTest`` harness — no API key, no Qdrant store, no network.
"""
from pathlib import Path

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from app.history_store import (
    append_entry,
    append_feedback,
    load_entries,
    load_feedback,
)

APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"
QUESTION = "What does the zephyr protocol require of auditors?"


def _seed_turn():
    return append_entry({
        "timestamp": "2026-10-28 09:00:00",
        "question": QUESTION,
        "answer": "Auditors must sample 5% of entries [zephyr_spec.pdf | page 3].",
        "latency": {"total_sec": 0.42, "retrieval_sec": 0.1, "llm_sec": 0.32},
        "docs": [
            Document(
                page_content="Auditors must sample 5% of entries.",
                metadata={"source": "/data/zephyr_spec.pdf", "page_number": 3},
            )
        ],
        "prompt_version": "v1_strict",
        "collection": "default",
    })


def _run_app():
    at = AppTest.from_file(str(APP_MAIN), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _all_text(at):
    return "\n".join(m.value for m in at.markdown)


def _captions(at):
    return "\n".join(c.value for c in at.caption)


def test_saved_turn_is_restored_after_a_reload():
    _seed_turn()

    at = _run_app()

    assert QUESTION in _all_text(at)
    # Restored chunks came back as Documents, so the source card renders too.
    assert "zephyr_spec.pdf" in _all_text(at)


def test_feedback_click_is_persisted_and_shown_in_history():
    turn = _seed_turn()
    at = _run_app()

    at.button(key=f"up_{turn['id']}").click().run()

    assert not at.exception, at.exception
    assert load_feedback()[turn["id"]]["helpful"] is True
    assert "Marked helpful" in _captions(at)
    assert "👍 1 helpful" in _captions(at)


def test_changing_the_verdict_keeps_one_rating_per_turn():
    turn = _seed_turn()
    at = _run_app()

    at.button(key=f"up_{turn['id']}").click().run()
    at.button(key=f"dn_{turn['id']}").click().run()

    assert not at.exception, at.exception
    feedback = load_feedback()
    assert len(feedback) == 1
    assert feedback[turn["id"]]["helpful"] is False
    assert "Marked not helpful" in _captions(at)
    assert "👎 1 not helpful" in _captions(at)


def test_clear_chat_removes_the_saved_log():
    turn = _seed_turn()
    append_feedback(turn["id"], True)
    at = _run_app()

    clear = next(b for b in at.button if b.label == "🗑️ Clear Chat")
    clear.click().run()

    assert not at.exception, at.exception
    assert load_entries() == []
    assert load_feedback() == {}
    # Not restored on the rerun either.
    assert QUESTION not in _all_text(at)


def test_new_turn_is_saved_to_disk(monkeypatch):
    import app.views.chat as chat_view

    docs = [
        Document(
            page_content="Auditors must sample 5% of entries.",
            metadata={"source": "/data/zephyr_spec.pdf", "page_number": 3},
        )
    ]

    def fake_query(question, prompt_version, temperature, collection):
        return (
            "Auditors must sample 5% [zephyr_spec.pdf | page 3].",
            docs,
            {"total_sec": 0.2, "retrieval_sec": 0.05, "llm_sec": 0.15},
        )

    monkeypatch.setattr(chat_view, "run_rag_query", fake_query)
    at = _run_app()

    at.chat_input[0].set_value("What about sampling?").run()

    assert not at.exception, at.exception
    [saved] = load_entries()
    assert saved["question"] == "What about sampling?"
    assert saved["answer"].startswith("Auditors must sample 5%")
    assert saved["docs"][0].metadata["source"].endswith("zephyr_spec.pdf")
    assert saved["id"]
