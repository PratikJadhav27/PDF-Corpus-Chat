"""
Unit tests for the Analytics data layer (ISSUE-005).

The Analytics tab used to hardcode its numbers. These tests pin the contract of
``app/analytics.py``: every value comes from a real artifact, a missing metric
stays missing, and deltas are only computed between comparable runs.
"""

import json

import pytest

from app import analytics


def _summary(**overrides):
    base = {
        "total_questions": 5,
        "citation_rate": 80.0,
        "idk_rate": 20.0,
        "avg_latency_sec": 2.5,
        "avg_citation_validity_ratio": 0.80,
        "avg_faithfulness": 0.90,
        "avg_context_relevance": 0.70,
        "avg_answer_completeness": 0.60,
        "avg_composite_score": 0.73,
    }
    base.update(overrides)
    return base


def _report(generated_at, summary=None, collection="ci_collection", mode="hybrid"):
    return {
        "generated_at": generated_at,
        "collection_name": collection,
        "retrieval_mode": mode,
        "top_k": 5,
        "prompt_version": "v1_strict",
        "summary": summary if summary is not None else _summary(),
        "results": [],
    }


BENCHMARK = {
    "generated_at": "2026-07-14T06:22:30",
    "collection": "ci_collection",
    "top_k": 5,
    "n_queries": 12,
    "strategies": {
        "similarity": {"recall_at_5": 0.864, "ndcg_at_5": 0.835},
        "hybrid": {"recall_at_5": 0.912, "ndcg_at_5": 0.884},
        "bm25": {"recall_at_5": 0.695, "ndcg_at_5": 0.684},
        "mmr": {"recall_at_5": 0.812, "ndcg_at_5": 0.790},
    },
}


# ---------------------------------------------------------------------------
# Benchmark artifact
# ---------------------------------------------------------------------------

def test_load_benchmark_treats_repo_scaffold_as_missing(tmp_path):
    """The committed scaffold (n_queries 0, strategies {}) is not a benchmark."""
    path = tmp_path / "latest.json"
    path.write_text(json.dumps({
        "_note": "auto-generated",
        "_generated_at": None,
        "collection": None,
        "top_k": 5,
        "n_queries": 0,
        "strategies": {},
    }))

    assert analytics.load_benchmark(path) == {}


def test_load_benchmark_missing_or_corrupt_file(tmp_path):
    assert analytics.load_benchmark(tmp_path / "absent.json") == {}

    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert analytics.load_benchmark(broken) == {}


def test_load_benchmark_returns_real_run(tmp_path):
    path = tmp_path / "latest.json"
    path.write_text(json.dumps(BENCHMARK))

    loaded = analytics.load_benchmark(path)

    assert loaded["n_queries"] == 12
    assert loaded["strategies"]["hybrid"]["recall_at_5"] == pytest.approx(0.912)


def test_benchmark_rows_are_real_values_in_display_order():
    rows = analytics.benchmark_rows(BENCHMARK)

    assert [r["key"] for r in rows] == ["hybrid", "similarity", "mmr", "bm25"]
    hybrid = rows[0]
    assert hybrid["name"] == "Hybrid (Dense + BM25)"
    assert hybrid["recall"] == pytest.approx(0.912)
    assert hybrid["ndcg"] == pytest.approx(0.884)
    assert hybrid["active"] is True
    assert [r["active"] for r in rows[1:]] == [False, False, False]


def test_benchmark_rows_fall_back_to_any_recall_suffix():
    """A baseline measured at a different top_k must not blank the section."""
    benchmark = {
        "top_k": 3,
        "n_queries": 4,
        "strategies": {"hybrid": {"recall_at_5": 0.5, "ndcg_at_5": 0.4}},
    }

    rows = analytics.benchmark_rows(benchmark)

    assert rows[0]["recall"] == pytest.approx(0.5)
    assert rows[0]["ndcg"] == pytest.approx(0.4)


def test_benchmark_rows_skip_strategies_without_numbers():
    benchmark = {
        "top_k": 5,
        "n_queries": 2,
        "strategies": {"hybrid": {"recall_at_5": 0.5}, "ghost": {}},
    }

    rows = analytics.benchmark_rows(benchmark)

    assert [r["key"] for r in rows] == ["hybrid"]
    assert rows[0]["ndcg"] is None


def test_benchmark_rows_and_caption_on_empty_baseline():
    assert analytics.benchmark_rows({}) == []
    assert analytics.benchmark_caption({}) == ""

    caption = analytics.benchmark_caption(BENCHMARK)
    assert "Recall@5" in caption and "nDCG@5" in caption
    assert "12 golden queries" in caption
    assert "ci_collection" in caption
    assert "2026-07-14 06:22" in caption


# ---------------------------------------------------------------------------
# Evaluation reports
# ---------------------------------------------------------------------------

def test_load_eval_reports_newest_first_and_skips_non_reports(tmp_path):
    (tmp_path / "older.json").write_text(json.dumps(_report("2026-10-18T09:00:00")))
    (tmp_path / "newer.json").write_text(json.dumps(_report("2026-10-19T10:00:00")))
    (tmp_path / "unrelated.json").write_text(json.dumps({"hello": "world"}))
    (tmp_path / "corrupt.json").write_text("nope")

    reports = analytics.load_eval_reports(tmp_path)

    assert [r["generated_at"] for r in reports] == [
        "2026-10-19T10:00:00",
        "2026-10-18T09:00:00",
    ]


def test_load_eval_reports_missing_directory(tmp_path):
    assert analytics.load_eval_reports(tmp_path / "nope") == []


def test_merge_reports_prefers_session_and_drops_duplicates():
    on_disk = [_report("2026-10-18T09:00:00")]
    session = _report("2026-10-19T10:00:00")

    merged = analytics.merge_reports(session, on_disk)
    assert [r["generated_at"] for r in merged] == [
        "2026-10-19T10:00:00",
        "2026-10-18T09:00:00",
    ]

    # Passing the same report twice (e.g. after an export to disk) must not
    # create a fake "previous run" to compare against.
    assert len(analytics.merge_reports(session, [session] + on_disk)) == 2


# ---------------------------------------------------------------------------
# Headline cards
# ---------------------------------------------------------------------------

def test_cards_without_any_report_are_explicitly_empty():
    cards = analytics.metric_cards([])

    assert [c["value"] for c in cards] == ["—", "—", "—", "—"]
    assert all(c["measured"] is False for c in cards)
    assert all(c["trend"] == "no runs yet" for c in cards)


def test_cards_do_not_invent_an_unmeasured_metric():
    report = _report("2026-10-19T10:00:00", summary=_summary(
        avg_faithfulness=None, avg_entailment_rate=None,
    ))

    cards = {c["key"]: c for c in analytics.metric_cards([report])}

    assert cards["avg_faithfulness"]["value"] == "—"
    assert cards["avg_faithfulness"]["measured"] is False
    assert cards["avg_faithfulness"]["trend"] == "not measured"
    assert cards["avg_citation_validity_ratio"]["value"] == "0.80"
    assert cards["avg_latency_sec"]["value"] == "2.50s"
    assert cards["avg_latency_sec"]["trend"] == "first run"


def test_cards_report_real_deltas_and_improvement_direction():
    older = _report("2026-10-18T09:00:00", summary=_summary(
        avg_faithfulness=0.85,
        avg_citation_validity_ratio=0.75,
        avg_latency_sec=3.0,
        avg_entailment_rate=0.70,
    ))
    newer = _report("2026-10-19T10:00:00", summary=_summary(
        avg_faithfulness=0.90,
        avg_citation_validity_ratio=0.80,
        avg_latency_sec=2.5,
        avg_entailment_rate=0.60,
    ))

    cards = {c["key"]: c for c in analytics.metric_cards([newer, older])}

    assert cards["avg_faithfulness"]["trend"] == "↑ 0.05 vs previous run"
    assert cards["avg_faithfulness"]["trend_cls"] == "dm-trend-up"
    # Lower latency is an improvement and must not be coloured as a regression.
    assert cards["avg_latency_sec"]["trend"] == "↓ 0.50s vs previous run"
    assert cards["avg_latency_sec"]["trend_cls"] == "dm-trend-up"
    # A drop in a quality score is a regression.
    assert cards["avg_entailment_rate"]["trend_cls"] == "dm-trend-down"


def test_cards_treat_a_tiny_change_as_unchanged():
    older = _report("2026-10-18T09:00:00", summary=_summary(avg_faithfulness=0.900))
    newer = _report("2026-10-19T10:00:00", summary=_summary(avg_faithfulness=0.901))

    cards = {c["key"]: c for c in analytics.metric_cards([newer, older])}

    assert cards["avg_faithfulness"]["trend"] == "± unchanged vs previous run"
    assert cards["avg_faithfulness"]["trend_cls"] == "dm-trend-stable"


def test_cards_only_compare_runs_on_the_same_collection_and_mode():
    hybrid = _report("2026-10-18T09:00:00", mode="hybrid")
    mmr = _report("2026-10-19T10:00:00", summary=_summary(avg_faithfulness=0.95), mode="mmr")

    cards = {c["key"]: c for c in analytics.metric_cards([mmr, hybrid])}

    assert cards["avg_faithfulness"]["trend"] == "first run"


def test_report_caption_is_provenance_not_marketing():
    caption = analytics.report_caption(_report("2026-10-19T10:00:00"))

    assert "2026-10-19 10:00" in caption
    assert "ci_collection" in caption
    assert "hybrid" in caption
    assert "5 questions" in caption
    assert "No evaluation data yet" in analytics.report_caption(None)
