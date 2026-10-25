"""
Unit tests for the recent-evaluations table builder (ISSUE-006).

The Analytics tab built its table from ``faithfulness_score`` and
``citation_validity.validity_ratio`` — keys ``evaluate_one()`` never writes — so
every row rendered ``0.00`` / ``0.00`` and a ⚠️ even for a clean run. These tests
pin the contract of ``app.analytics.recent_eval_rows`` against the evaluator's
real output shape: top-level ``citation_validity_ratio`` / citation counts, and
judge scores under ``llm_metrics``.
"""

import pytest

from app import analytics


def _result(**overrides):
    """A row shaped exactly like ``src.eval.evaluator.evaluate_one()`` output."""
    base = {
        "question": "What is alpha?",
        "answer": "Alpha applies to bravo [a.pdf | page 1].",
        "has_citations": True,
        "is_idk": False,
        "citation_validity_ratio": 0.8,
        "citation_count": 2,
        "valid_citation_count": 1,
        "sources": [],
        "llm_metrics": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Citation metrics
# ---------------------------------------------------------------------------

def test_citation_ratio_is_read_from_the_top_level_key():
    (row,) = analytics.recent_eval_rows([_result(citation_validity_ratio=0.8)])

    assert row["citation"] == pytest.approx(0.8)


def test_nonexistent_nested_citation_key_is_ignored():
    """The old view read ``citation_validity.validity_ratio``; nothing writes it."""
    (row,) = analytics.recent_eval_rows(
        [_result(citation_validity={"validity_ratio": 0.1})]
    )

    assert row["citation"] == pytest.approx(0.8)
    assert row["citation"] != pytest.approx(0.1)


def test_citation_counts_come_from_the_reported_counts():
    assert analytics.recent_eval_rows([_result()])[0]["citations"] == "1/2"

    missing = _result(citation_count=None, valid_citation_count=None)
    assert analytics.recent_eval_rows([missing])[0]["citations"] is None


def test_missing_citation_ratio_stays_missing():
    (row,) = analytics.recent_eval_rows([_result(citation_validity_ratio=None)])

    assert row["citation"] is None


# ---------------------------------------------------------------------------
# Judge metrics
# ---------------------------------------------------------------------------

def test_faithfulness_is_read_from_llm_metrics_when_judged():
    judged = _result(llm_metrics={
        "faithfulness": {"score": 0.73},
        "context_relevance": {"score": 0.9},
        "answer_completeness": {"score": 0.8},
        "composite_score": 0.81,
    })

    (row,) = analytics.recent_eval_rows([judged])

    assert row["faithfulness"] == pytest.approx(0.73)


def test_faithfulness_accepts_a_bare_number():
    (row,) = analytics.recent_eval_rows([_result(llm_metrics={"faithfulness": 0.66})])

    assert row["faithfulness"] == pytest.approx(0.66)


@pytest.mark.parametrize(
    "llm_metrics",
    [None, {}, {"error": "judge unavailable"}],
    ids=["not_requested", "empty", "judge_error"],
)
def test_unmeasured_faithfulness_is_not_zero(llm_metrics):
    (row,) = analytics.recent_eval_rows([_result(llm_metrics=llm_metrics)])

    assert row["faithfulness"] is None


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def test_good_citation_run_without_judging_passes():
    """The pre-fix table showed ⚠️ here because it read a non-existent score."""
    (row,) = analytics.recent_eval_rows([_result(citation_validity_ratio=0.9)])

    assert row["status"] == "✅"


def test_low_citation_run_is_flagged():
    (row,) = analytics.recent_eval_rows([_result(citation_validity_ratio=0.2)])

    assert row["status"] == "⚠️"


def test_every_measured_score_must_clear_the_threshold():
    judged = _result(
        citation_validity_ratio=0.4,
        llm_metrics={"faithfulness": {"score": 0.95}},
    )

    (row,) = analytics.recent_eval_rows([judged])

    assert row["status"] == "⚠️"


def test_refusal_is_not_scored():
    """An IDK answer has nothing to ground, so it is neither pass nor fail."""
    refusal = _result(
        answer="I don't know based on the provided documents.",
        has_citations=False,
        is_idk=True,
        citation_validity_ratio=0.0,
        citation_count=0,
        valid_citation_count=0,
    )

    (row,) = analytics.recent_eval_rows([refusal])

    assert row["status"] == "—"


def test_nothing_measured_is_not_scored():
    (row,) = analytics.recent_eval_rows(
        [_result(citation_validity_ratio=None, llm_metrics=None)]
    )

    assert row["status"] == "—"


# ---------------------------------------------------------------------------
# Ordering, truncation and tolerances
# ---------------------------------------------------------------------------

def test_rows_are_newest_first_and_limited():
    results = [_result(question=f"Q{i}") for i in range(1, 13)]

    rows = analytics.recent_eval_rows(results)

    assert len(rows) == analytics.RECENT_LIMIT == 10
    assert [row["question"] for row in rows] == [f"Q{i}" for i in range(12, 2, -1)]


def test_limit_of_zero_keeps_everything():
    results = [_result(question=f"Q{i}") for i in range(12)]

    assert len(analytics.recent_eval_rows(results, limit=0)) == 12


def test_question_is_truncated_only_when_needed():
    short = analytics.recent_eval_rows(
        [_result(question="What is alpha?")]
    )[0]["question"]
    assert short == "What is alpha?"

    long_question = "Why " + "does " * 30 + "this matter?"
    (long_row,) = analytics.recent_eval_rows([_result(question=long_question)])
    assert long_row["question"].endswith("…")
    assert len(long_row["question"]) <= analytics.QUESTION_WIDTH
    assert "  " not in long_row["question"]

    (wrapped,) = analytics.recent_eval_rows(
        [_result(question="First line\nsecond   line")]
    )
    assert wrapped["question"] == "First line second line"


def test_missing_latency_is_none_not_zero():
    (row,) = analytics.recent_eval_rows([_result(latency_sec=None)])

    assert row["latency"] is None


def test_non_dict_results_are_skipped():
    rows = analytics.recent_eval_rows([None, "broken", _result(question="Kept")])

    assert [row["question"] for row in rows] == ["Kept"]


def test_empty_results_make_no_rows():
    assert analytics.recent_eval_rows([]) == []
    assert analytics.recent_eval_rows(None) == []


def test_caption_states_scope_and_legend():
    caption = analytics.recent_eval_caption(3, 3)

    assert "3 evaluated questions" in caption
    assert "newest 3" not in caption
    assert "✅ ≥ 0.70" in caption
    assert "— refusal or not measured" in caption

    limited = analytics.recent_eval_caption(12, 10)
    assert "showing the newest 10" in limited

    assert "1 evaluated question ·" in analytics.recent_eval_caption(1, 1)
