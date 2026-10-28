"""
tests/test_history_store.py

Unit tests for the durable chat log / feedback store (ISSUE-009, decision D3).

The autouse fixture in ``tests/conftest.py`` points ``CHAT_LOG_DIR`` at a
per-test temp directory, so these tests never touch ``data/chat_history/``.
"""
import json
import os

from langchain_core.documents import Document

from app.history_store import (
    append_entry,
    append_feedback,
    clear_log,
    feedback_summary,
    load_entries,
    load_feedback,
    log_dir,
)

QUESTION = "What does the zephyr protocol require of auditors?"


def _doc(name="zephyr_spec.pdf", page=3, text="Alpha applies to bravo."):
    return Document(
        page_content=text,
        metadata={
            "source": f"/data/{name}",
            "page_number": page,
            "chunk_index": 1,
            "retrieval": {"relevance": 0.87, "channels": ["vector", "bm25"]},
        },
    )


def _entry(question=QUESTION, **overrides):
    entry = {
        "timestamp": "2026-10-28 10:00:00",
        "question": question,
        "answer": "Auditors must sample 5% [zephyr_spec.pdf | page 3].",
        "latency": {"total_sec": 0.4, "retrieval_sec": 0.1, "llm_sec": 0.3},
        "docs": [_doc()],
        "prompt_version": "v1_strict",
        "collection": "default",
    }
    entry.update(overrides)
    return entry


def _raw_lines(filename):
    with open(os.path.join(log_dir(), filename), encoding="utf-8") as f:
        return [line for line in f.read().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Conversation turns
# ---------------------------------------------------------------------------

def test_log_dir_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAT_LOG_DIR", str(tmp_path / "elsewhere"))
    assert log_dir() == str(tmp_path / "elsewhere")


def test_append_entry_writes_one_json_line_and_returns_its_id():
    record = append_entry(_entry())

    assert record["id"]
    lines = _raw_lines("chat_history.jsonl")
    assert len(lines) == 1

    raw = json.loads(lines[0])
    assert raw["id"] == record["id"]
    assert raw["question"] == QUESTION
    # Documents are flattened to JSON-safe fields, not serialized as objects.
    assert raw["docs"] == [
        {
            "page_content": "Alpha applies to bravo.",
            "metadata": _doc().metadata,
        }
    ]


def test_load_entries_rebuilds_documents():
    append_entry(_entry())

    [loaded] = load_entries()

    [doc] = loaded["docs"]
    assert isinstance(doc, Document)
    assert doc.page_content == "Alpha applies to bravo."
    assert doc.metadata["source"] == "/data/zephyr_spec.pdf"
    assert doc.metadata["retrieval"]["relevance"] == 0.87
    assert loaded["latency"]["total_sec"] == 0.4
    assert loaded["prompt_version"] == "v1_strict"
    assert loaded["collection"] == "default"


def test_entries_load_in_append_order():
    first = append_entry(_entry(question="first?"))
    second = append_entry(_entry(question="second?"))

    entries = load_entries()

    assert [e["question"] for e in entries] == ["first?", "second?"]
    assert [e["id"] for e in entries] == [first["id"], second["id"]]


def test_reading_a_missing_log_returns_empty_and_creates_nothing():
    assert load_entries() == []
    assert load_feedback() == {}
    assert not os.path.isdir(log_dir())


def test_torn_final_line_is_skipped():
    append_entry(_entry())
    with open(os.path.join(log_dir(), "chat_history.jsonl"), "a", encoding="utf-8") as f:
        f.write('{"question": "half a rec')  # simulated crash mid-write

    entries = load_entries()

    assert len(entries) == 1
    assert entries[0]["question"] == QUESTION


def test_unicode_and_newlines_round_trip():
    question = "Résumé — what about\nthe second page? ✅"
    append_entry(_entry(question=question, answer="Line one\n\nLine two — naïve."))

    [loaded] = load_entries()

    assert loaded["question"] == question
    assert loaded["answer"] == "Line one\n\nLine two — naïve."
    # One record per line: an embedded newline must not split the record.
    assert len(_raw_lines("chat_history.jsonl")) == 1


def test_row_without_id_gets_a_stable_positional_id():
    append_entry(_entry())
    path = os.path.join(log_dir(), "chat_history.jsonl")
    with open(path, encoding="utf-8") as f:
        row = json.loads(f.read())
    row.pop("id")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    first = load_entries()
    second = load_entries()

    assert first[0]["id"] == "legacy_0"
    # Stable across reads, so feedback stored against it stays linked.
    assert second[0]["id"] == first[0]["id"]


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def test_feedback_is_readable_by_entry_id():
    turn = append_entry(_entry())

    append_feedback(turn["id"], True, question=QUESTION, collection="default")

    feedback = load_feedback()
    assert feedback[turn["id"]]["helpful"] is True
    assert feedback[turn["id"]]["question"] == QUESTION


def test_latest_click_replaces_the_earlier_verdict():
    turn = append_entry(_entry())

    append_feedback(turn["id"], True, question=QUESTION)
    append_feedback(turn["id"], False, question=QUESTION)

    feedback = load_feedback()
    assert len(feedback) == 1
    assert feedback[turn["id"]]["helpful"] is False
    # Both clicks stay in the log for later analysis.
    assert len(_raw_lines("feedback.jsonl")) == 2
    assert feedback_summary(feedback) == {"helpful": 0, "not_helpful": 1}


def test_summary_counts_distinct_turns_once():
    first = append_entry(_entry(question="first?"))
    second = append_entry(_entry(question="second?"))

    append_feedback(first["id"], True)
    append_feedback(second["id"], False)
    append_feedback(first["id"], True)  # same verdict again

    assert feedback_summary() == {"helpful": 1, "not_helpful": 1}


def test_feedback_stays_linked_after_a_reload():
    turn = append_entry(_entry())
    append_feedback(turn["id"], True)

    [reloaded] = load_entries()

    assert load_feedback()[reloaded["id"]]["helpful"] is True


# ---------------------------------------------------------------------------
# Clearing
# ---------------------------------------------------------------------------

def test_clear_log_removes_turns_and_feedback():
    turn = append_entry(_entry())
    append_feedback(turn["id"], False)

    clear_log()

    assert load_entries() == []
    assert load_feedback() == {}
    assert not os.path.exists(os.path.join(log_dir(), "chat_history.jsonl"))
    assert not os.path.exists(os.path.join(log_dir(), "feedback.jsonl"))


def test_clear_log_is_safe_when_nothing_was_saved():
    clear_log()  # no directory at all
    assert load_entries() == []
