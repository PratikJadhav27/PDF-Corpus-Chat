"""
AppTest coverage for the Recent Evaluations table (ISSUE-006).

The table used to read ``faithfulness_score`` and
``citation_validity.validity_ratio`` — keys ``evaluate_one()`` never produces — so
a clean run still rendered ``0.00`` / ``0.00`` and ⚠️ for every row. This drives
the real entrypoint with session results in the evaluator's real shape and checks
the rendered table cell by cell.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


def _result(question, *, ratio, judged=None, is_idk=False, latency=1.0,
            counts=(2, 1)):
    """One ``evaluate_one()`` row, as the batch runner stores it."""
    total, valid = counts
    llm_metrics = None
    if judged is not None:
        llm_metrics = {
            "faithfulness": {"score": judged},
            "context_relevance": {"score": 0.9},
            "answer_completeness": {"score": 0.8},
            "composite_score": 0.85,
        }
    return {
        "question": question,
        "answer": "…",
        "has_citations": total > 0,
        "is_idk": is_idk,
        "citation_validity_ratio": ratio,
        "citation_count": total,
        "valid_citation_count": valid,
        "sources": [],
        "llm_metrics": llm_metrics,
        "latency_sec": latency,
    }


def _render_with(results):
    at = AppTest.from_file(str(APP_MAIN), default_timeout=120)
    at.run()
    assert not at.exception, at.exception

    at.session_state["eval_results"] = results
    at.run()
    assert not at.exception, at.exception
    return at


def _table(at):
    frames = at.dataframe
    assert frames, "Recent Evaluations table was not rendered"
    return frames[0].value


def _captions(at):
    return "\n".join(c.value for c in at.caption)


def test_table_reads_the_keys_the_evaluator_writes():
    at = _render_with([
        _result("What is alpha?", ratio=0.5, latency=1.0),
        _result("What is bravo?", ratio=1.0, judged=0.9, latency=2.5, counts=(1, 1)),
        _result("Refused question", ratio=0.0, is_idk=True, latency=0.4,
                counts=(0, 0)),
    ])

    df = _table(at)
    assert list(df.columns) == ["Question", "Faithfulness", "Citation", "Latency", "Status"]

    # Newest first: the refusal, then the judged clean run, then the partial run.
    assert [list(row) for row in df.itertuples(index=False)] == [
        ["Refused question", "—", "0.00 (0/0)", "0.40s", "—"],
        ["What is bravo?", "0.90", "1.00 (1/1)", "2.50s", "✅"],
        ["What is alpha?", "—", "0.50 (1/2)", "1.00s", "⚠️"],
    ]

    captions = _captions(at)
    assert "This session · 3 evaluated questions" in captions
    assert "✅ ≥ 0.70" in captions


def test_judge_error_is_shown_as_unmeasured():
    result = _result("What is charlie?", ratio=0.9)
    result["llm_metrics"] = {"error": "judge unavailable"}

    df = _table(_render_with([result]))

    assert df.iloc[0]["Faithfulness"] == "—"
    assert df.iloc[0]["Citation"] == "0.90 (1/2)"
    assert df.iloc[0]["Status"] == "✅"


def test_empty_session_keeps_the_empty_state():
    at = AppTest.from_file(str(APP_MAIN), default_timeout=120)
    at.run()
    assert not at.exception, at.exception

    assert not at.dataframe
    body = "\n".join(m.value for m in at.markdown)
    assert "No evaluation runs yet" in body
