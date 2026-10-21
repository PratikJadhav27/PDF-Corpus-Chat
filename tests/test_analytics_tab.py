"""
AppTest coverage for the Analytics tab (ISSUE-005).

The tab used to show fabricated numbers: hardcoded benchmark scores, invented
"↑ +4.2% vs last run" trends, and card fallbacks that appeared whenever the
baseline lacked the keys. These tests drive the real entrypoint with controlled
artifacts and assert that:

  - with no artifacts, the tab shows empty states and no invented numbers;
  - with artifacts, every displayed number comes from them;
  - a fresh in-session report takes precedence over the files on disk.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import analytics

APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"

# Numbers that only ever existed as demo data in the old view.
FORMERLY_FABRICATED = ["0.948", "0.864", "0.812", "0.695", "0.89", "0.91", "1.24", "0.87",
                       "+4.2%", "+2.8%", "−180ms"]

BENCHMARK = {
    "generated_at": "2026-07-14T06:22:30",
    "collection": "ci_collection",
    "top_k": 5,
    "n_queries": 12,
    "strategies": {
        "hybrid": {"recall_at_5": 0.912, "ndcg_at_5": 0.884},
        "similarity": {"recall_at_5": 0.864, "ndcg_at_5": 0.835},
    },
}


def _summary(**overrides):
    base = {
        "total_questions": 5,
        "citation_rate": 80.0,
        "idk_rate": 20.0,
        "avg_latency_sec": 2.5,
        "avg_citation_validity_ratio": 0.80,
        "avg_faithfulness": 0.90,
        "avg_entailment_rate": 0.70,
    }
    base.update(overrides)
    return base


def _report(generated_at, summary=None):
    return {
        "generated_at": generated_at,
        "collection_name": "ci_collection",
        "retrieval_mode": "hybrid",
        "top_k": 5,
        "prompt_version": "v1_strict",
        "summary": summary or _summary(),
        "results": [],
    }


def _render(monkeypatch, benchmark, reports):
    monkeypatch.setattr(analytics, "load_benchmark", lambda *a, **k: benchmark)
    monkeypatch.setattr(analytics, "load_eval_reports", lambda *a, **k: reports)

    at = AppTest.from_file(str(APP_MAIN), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _markdown(at):
    return "\n".join(m.value for m in at.markdown)


def _captions(at):
    return "\n".join(c.value for c in at.caption)


def test_empty_artifacts_render_honest_empty_states(monkeypatch):
    at = _render(monkeypatch, {}, [])

    body = _markdown(at)
    for value in FORMERLY_FABRICATED:
        assert value not in body, f"fabricated value {value!r} still rendered"

    # Cards exist but admit they measured nothing.
    assert body.count('class="dm-metric-card"') >= 4
    assert "no runs yet" in body

    assert "No evaluation data yet" in _captions(at)
    assert "No retrieval benchmark has been recorded yet" in body
    assert "benchmark_retrieval.py" in "\n".join(c.value for c in at.code)


def test_real_artifacts_drive_cards_and_benchmark(monkeypatch):
    reports = [
        _report("2026-10-19T10:00:00"),
        _report("2026-10-18T09:00:00", summary=_summary(
            avg_faithfulness=0.85,
            avg_citation_validity_ratio=0.75,
            avg_latency_sec=3.0,
            avg_entailment_rate=0.60,
        )),
    ]

    at = _render(monkeypatch, BENCHMARK, reports)
    body = _markdown(at)
    captions = _captions(at)

    # Headline values are the latest run's numbers, not placeholders.
    assert "0.90" in body          # avg_faithfulness
    assert "0.80" in body          # avg_citation_validity_ratio
    assert "2.50s" in body         # avg_latency_sec
    assert "0.70" in body          # avg_entailment_rate
    assert "↑ 0.05 vs previous run" in body
    assert "↓ 0.50s vs previous run" in body

    # Provenance is stated on the tab.
    assert "Latest run 2026-10-19 10:00" in captions
    assert "ci_collection" in captions

    # Benchmark numbers come from the artifact, with its real coverage.
    assert "Recall 91.2%" in body
    assert "nDCG 0.884" in body
    assert "Recall 86.4%" in body
    assert "12 golden queries" in captions

    for value in FORMERLY_FABRICATED:
        assert value not in body, f"fabricated value {value!r} still rendered"


def test_in_session_report_takes_precedence_over_disk(monkeypatch):
    at = _render(monkeypatch, {}, [_report("2026-10-18T09:00:00")])

    at.session_state["eval_report"] = _report(
        "2026-10-19T12:00:00",
        summary=_summary(avg_faithfulness=0.55, avg_latency_sec=7.25),
    )
    at.run()
    assert not at.exception, at.exception

    body = _markdown(at)
    assert "0.55" in body
    assert "7.25s" in body
    assert "Latest run 2026-10-19 12:00" in _captions(at)
