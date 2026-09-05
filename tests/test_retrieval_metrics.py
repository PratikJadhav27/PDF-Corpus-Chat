"""
tests/test_retrieval_metrics.py

Unit tests for src/eval/retrieval_metrics.py.

All tests use only small hand-crafted inputs so they run instantly
with no external dependencies.
"""
import math
import pytest
from src.eval.retrieval_metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    average_precision,
    compute_run_metrics,
)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------
class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_zero_recall_no_overlap(self):
        assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, k=3) == pytest.approx(2/3)

    def test_k_limits_to_top_k(self):
        # only first 2 checked — "b" is at pos 3, not seen
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, k=2) == pytest.approx(0.5)

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k(["a", "b"], set(), k=2) == 0.0

    def test_empty_retrieved_returns_zero(self):
        assert recall_at_k([], {"a"}, k=5) == 0.0


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------
class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_zero_precision(self):
        assert precision_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_half_precision(self):
        assert precision_at_k(["a", "x"], {"a", "b"}, k=2) == pytest.approx(0.5)

    def test_k_zero_returns_zero(self):
        assert precision_at_k(["a"], {"a"}, k=0) == 0.0

    def test_k_larger_than_retrieved(self):
        # top-5 but only 2 retrieved; precision = hits / k = 1/5
        assert precision_at_k(["a", "x"], {"a"}, k=5) == pytest.approx(1/5)


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------
class TestMRR:
    def test_first_hit_at_rank_1(self):
        assert mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_first_hit_at_rank_2(self):
        assert mrr(["x", "a", "c"], {"a"}) == pytest.approx(0.5)

    def test_first_hit_at_rank_3(self):
        assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1/3)

    def test_no_hit_returns_zero(self):
        assert mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_empty_retrieved(self):
        assert mrr([], {"a"}) == 0.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------
class TestNDCGAtK:
    def test_perfect_ndcg(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        assert ndcg_at_k(["x", "y"], {"a", "b"}, k=2) == pytest.approx(0.0)

    def test_second_place_hit(self):
        # Retrieved: [x, a] vs ideal [a, x]
        # DCG = 0/log2(2) + 1/log2(3) ≈ 0.631
        # IDCG = 1/log2(2) + 0/log2(3) = 1.0
        result = ndcg_at_k(["x", "a"], {"a"}, k=2)
        expected = (1 / math.log2(3)) / (1 / math.log2(2))
        assert result == pytest.approx(expected, rel=1e-5)

    def test_empty_relevant(self):
        assert ndcg_at_k(["a"], set(), k=1) == 0.0


# ---------------------------------------------------------------------------
# average_precision
# ---------------------------------------------------------------------------
class TestAveragePrecision:
    def test_perfect_ap(self):
        assert average_precision(["a", "b"], {"a", "b"}) == pytest.approx(1.0)

    def test_zero_ap(self):
        assert average_precision(["x", "y"], {"a"}) == pytest.approx(0.0)

    def test_ap_with_gap(self):
        # Retrieved: [a, x, b] — relevant {a, b}
        # Hit at 1: P@1 = 1/1 = 1.0
        # Hit at 3: P@3 = 2/3
        # AP = (1.0 + 2/3) / 2
        result = average_precision(["a", "x", "b"], {"a", "b"})
        assert result == pytest.approx((1.0 + 2/3) / 2)

    def test_empty_retrieved_returns_zero(self):
        assert average_precision([], {"a"}) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert average_precision(["a"], set()) == 0.0


# ---------------------------------------------------------------------------
# compute_run_metrics
# ---------------------------------------------------------------------------
class TestComputeRunMetrics:
    def _perfect_run(self):
        return [
            {
                "retrieved": ["report.pdf", "paper.pdf"],
                "relevant": ["report.pdf"],
                "latency_sec": 0.10,
            },
            {
                "retrieved": ["paper.pdf", "report.pdf"],
                "relevant": ["paper.pdf"],
                "latency_sec": 0.20,
            },
        ]

    def test_recall_at_2_perfect(self):
        m = compute_run_metrics(self._perfect_run(), k=2)
        assert m["recall_at_2"] == pytest.approx(1.0)

    def test_mrr_perfect_rank1(self):
        m = compute_run_metrics(self._perfect_run(), k=2)
        assert m["mrr"] == pytest.approx(1.0)

    def test_n_queries_correct(self):
        m = compute_run_metrics(self._perfect_run(), k=2)
        assert m["n_queries"] == 2

    def test_latency_p50_and_p95(self):
        run = [
            {"retrieved": ["a"], "relevant": ["a"], "latency_sec": lat}
            for lat in [0.1, 0.2, 0.3, 0.4, 0.5]
        ]
        m = compute_run_metrics(run, k=1)
        assert m["p50_latency_sec"] == pytest.approx(0.3)
        assert m["p95_latency_sec"] == pytest.approx(0.5)

    def test_empty_run_returns_zeros(self):
        m = compute_run_metrics([], k=5)
        assert m["recall_at_5"] == 0.0
        assert m["mrr"] == 0.0
        assert m["n_queries"] == 0

    def test_all_metric_keys_present(self):
        m = compute_run_metrics(self._perfect_run(), k=3)
        expected_keys = {
            "recall_at_3", "precision_at_3", "mrr",
            "ndcg_at_3", "map",
            "p50_latency_sec", "p95_latency_sec", "mean_latency_sec",
            "n_queries",
        }
        assert expected_keys.issubset(m.keys())
