"""
Regression tests for ISSUE-003 — asking the same question twice crashed the page.

Feedback buttons were keyed by ``hash(question)`` (``up_<hash>`` / ``dn_<hash>``)
and rendered once per history entry, so any repeated question produced duplicate
Streamlit widget keys and raised ``StreamlitDuplicateElementKey`` for the whole
script — bricking every tab, not just the badges.

These tests drive the real entrypoint (``app/main.py``) through Streamlit's
``AppTest`` harness with a pre-seeded history, so they need no API key, no
Qdrant store and no network.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"

ANSWER = "Alpha applies to bravo [a.pdf | page 1]."


def _history_entry(question: str) -> dict:
    return {
        "timestamp": "2026-10-19 10:00:00",
        "question": question,
        "answer": ANSWER,
        "latency": {"total_sec": 0.4, "retrieval_sec": 0.1, "llm_sec": 0.3},
        "docs": [],
        "prompt_type": "v1_strict",
        "collection": "default",
    }


def _render_with_history(entries: list) -> AppTest:
    at = AppTest.from_file(str(APP_MAIN), default_timeout=120)
    at.session_state["chat_history"] = list(entries)
    at.run()
    return at


def _feedback_buttons(at: AppTest) -> list:
    return [
        b for b in at.button
        if b.key and (b.key.startswith("up_") or b.key.startswith("dn_"))
    ]


def test_repeated_question_renders_without_duplicate_keys():
    at = _render_with_history([_history_entry("What is alpha?")] * 2)

    assert not at.exception, at.exception

    buttons = _feedback_buttons(at)
    assert len(buttons) == 4  # 👍 + 👎 for both entries
    keys = [b.key for b in buttons]
    assert len(keys) == len(set(keys))


def test_distinct_questions_still_render_separately():
    at = _render_with_history(
        [_history_entry("What is alpha?"), _history_entry("What is bravo?")]
    )

    assert not at.exception, at.exception

    buttons = _feedback_buttons(at)
    assert len(buttons) == 4
    keys = [b.key for b in buttons]
    assert len(keys) == len(set(keys))


def test_asking_the_same_question_twice_in_a_session(monkeypatch):
    """The user-facing reproduction: two identical chat inputs in one session."""
    import app.views.chat as chat_view

    def fake_rag_query(question, prompt_type, temperature, collection_name):
        return ANSWER, [], {"total_sec": 0.4, "retrieval_sec": 0.1, "llm_sec": 0.3}

    monkeypatch.setattr(chat_view, "run_rag_query", fake_rag_query)

    at = AppTest.from_file(str(APP_MAIN), default_timeout=120)
    at.run()
    assert not at.exception

    at.chat_input[0].set_value("What is alpha?").run()
    assert not at.exception, at.exception

    at.chat_input[0].set_value("What is alpha?").run()
    assert not at.exception, at.exception

    assert len(at.session_state["chat_history"]) == 2
    keys = [b.key for b in _feedback_buttons(at)]
    assert len(keys) == 4
    assert len(keys) == len(set(keys))


def test_clicking_one_thumb_records_exactly_one_feedback():
    at = _render_with_history([_history_entry("What is alpha?")] * 2)

    _feedback_buttons(at)[0].click()
    at.run()

    assert not at.exception, at.exception

    feedback = at.session_state["feedback_data"]
    assert len(feedback) == 1
    assert feedback[0]["question"] == "What is alpha?"
    assert feedback[0]["helpful"] is True
