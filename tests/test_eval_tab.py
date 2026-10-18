"""
Regression tests for ISSUE-002 — the in-app batch evaluator.

The Analytics tab used to call ``build_report(results)`` positionally against
``build_report(collection_name, retrieval_mode, top_k, results)``, so pressing
"Run Evaluation" always raised ``TypeError``. When the call was corrected, the
success cards still read ``avg_faithfulness`` / ``avg_citation_validity_ratio``
from the report's top level even though they live under ``report["summary"]``.

These tests drive the real entrypoint (``app/main.py``) through Streamlit's
``AppTest`` harness with the RAG call and heuristic evaluator stubbed, so they
need no API key, no Qdrant store and no network.
"""

from pathlib import Path

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"

QUESTIONS = "What is alpha?\nWhat is bravo?"

DOCS = [
    Document(
        page_content="Alpha applies to bravo.",
        metadata={"source": "a.pdf", "page_number": 1, "chunk_index": 0},
    )
]

CITATION_RATIO = 0.5


def _fake_rag_query(question, prompt_type, temperature, collection_name):
    """Deterministic stand-in for run_rag_query (no retrieval, no Gemini)."""
    return (
        "Alpha applies to bravo [a.pdf | page 1].",
        DOCS,
        {"total_sec": 0.42, "retrieval_sec": 0.1, "llm_sec": 0.3},
    )


def _fake_evaluate_one(question, answer, docs, include_llm_metrics=False):
    """Heuristic result with a fixed validity ratio and optional judge scores."""
    llm_metrics = None
    if include_llm_metrics:
        llm_metrics = {
            "faithfulness": {"score": 0.8},
            "context_relevance": {"score": 0.9},
            "answer_completeness": {"score": 0.7},
            "composite_score": 0.8,
        }
    return {
        "question": question,
        "answer": answer,
        "has_citations": True,
        "is_idk": False,
        "citation_validity_ratio": CITATION_RATIO,
        "citation_count": 2,
        "valid_citation_count": 1,
        "sources": [],
        "llm_metrics": llm_metrics,
    }


def _run_batch_eval(monkeypatch, *, include_llm_metrics: bool):
    """Render the app, run the batch evaluator, and return (AppTest, calls)."""
    import app.views.evaluation as evaluation_view

    calls = {"include_llm_metrics": []}

    def recording_evaluate_one(question, answer, docs, include_llm_metrics=False):
        calls["include_llm_metrics"].append(include_llm_metrics)
        return _fake_evaluate_one(question, answer, docs, include_llm_metrics)

    monkeypatch.setattr(evaluation_view, "run_rag_query", _fake_rag_query)
    monkeypatch.setattr(evaluation_view, "evaluate_one", recording_evaluate_one)

    at = AppTest.from_file(str(APP_MAIN), default_timeout=120)
    at.run()
    assert not at.exception

    questions_area = [t for t in at.text_area if t.label == "Questions (one per line)"][0]
    questions_area.set_value(QUESTIONS)

    if include_llm_metrics:
        checkbox = [c for c in at.checkbox if c.key == "eval_include_llm_metrics"][0]
        checkbox.check()

    run_button = [b for b in at.button if "Run Evaluation" in b.label][0]
    run_button.click()
    at.run()

    assert not at.exception, at.exception
    return at, calls


def _metric_values(at):
    return {m.label: m.value for m in at.metric}


def test_run_evaluation_button_builds_and_renders_report(monkeypatch):
    at, calls = _run_batch_eval(monkeypatch, include_llm_metrics=False)

    # The button no longer raises: both questions were retrieved and evaluated.
    results = at.session_state["eval_results"]
    assert len(results) == 2
    assert calls["include_llm_metrics"] == [False, False]

    # build_report was called with the run's metadata and the actual results.
    report = at.session_state["eval_report"]
    assert report["collection_name"] == at.session_state["eval_collection"]
    assert report["retrieval_mode"] == "hybrid"
    assert report["top_k"] == 4
    assert len(report["results"]) == 2
    assert report["summary"]["total_questions"] == 2
    assert report["summary"]["avg_citation_validity_ratio"] == CITATION_RATIO

    # Cards read the summary section (not the old, non-existent top-level keys).
    metrics = _metric_values(at)
    assert metrics["Avg Citation Valid."] == "0.50"
    assert metrics["Citation Rate"] == "100%"
    assert metrics["Avg Faithfulness"] == "—"  # LLM judging not requested

    assert any("2 questions evaluated" in s.value for s in at.success)
    assert len(at.get("download_button")) == 1


def test_llm_metrics_toggle_is_forwarded_and_averaged(monkeypatch):
    at, calls = _run_batch_eval(monkeypatch, include_llm_metrics=True)

    assert calls["include_llm_metrics"] == [True, True]

    report = at.session_state["eval_report"]
    assert report["summary"]["avg_faithfulness"] == 0.8
    assert report["summary"]["avg_composite_score"] == 0.8

    metrics = _metric_values(at)
    assert metrics["Avg Faithfulness"] == "0.80"
    assert metrics["Avg Citation Valid."] == "0.50"
