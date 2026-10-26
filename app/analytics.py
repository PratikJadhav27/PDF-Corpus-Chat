"""
app/analytics.py

Data layer for the Analytics tab (ISSUE-005).

Every number the tab renders is read from a file produced by a real run:

  - ``evals/baselines/latest.json``  — written by ``scripts/benchmark_retrieval.py``
  - ``evals/reports/*.json``         — written by ``scripts/run_eval.py``
  - the in-session report           — produced by the in-app batch evaluation

This module never invents a value. A metric that was not measured is returned as
``None`` so the view can render an honest empty state ("—", "not measured")
instead of a plausible-looking placeholder. Formats are the ones the scripts
actually write, so the tab cannot drift from the pipeline again silently.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_ROOT / "evals" / "baselines" / "latest.json"
REPORTS_DIR = PROJECT_ROOT / "evals" / "reports"

# Display labels and order for the strategies benchmark_retrieval.py writes.
STRATEGY_LABELS = {
    "hybrid": "Hybrid (Dense + BM25)",
    "similarity": "Dense Similarity (Cosine)",
    "mmr": "Maximal Marginal Relevance",
    "bm25": "Sparse BM25 Keyword",
    "hybrid_norrf": "Hybrid (no RRF)",
}
STRATEGY_ORDER = ["hybrid", "similarity", "mmr", "bm25", "hybrid_norrf"]

# The four headline cards. ``unit`` decides formatting and the trend direction
# that counts as an improvement (lower latency is better; higher scores are).
CARD_SPECS = [
    {"label": "AVG FAITHFULNESS", "key": "avg_faithfulness", "unit": "ratio",
     "color": "#3B82F6", "icon": "🎯"},
    {"label": "CITATION VALIDITY", "key": "avg_citation_validity_ratio", "unit": "ratio",
     "color": "#10B981", "icon": "🔗"},
    {"label": "AVG LATENCY", "key": "avg_latency_sec", "unit": "secs",
     "color": "#F59E0B", "icon": "⚡"},
    {"label": "ENTAILMENT RATE", "key": "avg_entailment_rate", "unit": "ratio",
     "color": "#8B5CF6", "icon": "🧩"},
]

BENCHMARK_COMMAND = (
    "python scripts/benchmark_retrieval.py \\\n"
    "    --collection <name> --top-k 5"
)

# Recent-evaluations table (ISSUE-006). ``evaluate_one()`` emits the citation
# metrics at the top level and the judge metrics under ``llm_metrics``, so those
# are the only keys read here.
RECENT_LIMIT = 10
PASS_THRESHOLD = 0.70
QUESTION_WIDTH = 60


# ---------------------------------------------------------------------------
# Reading artifacts
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[dict]:
    """Read a JSON object, or return None when the file is absent/unusable."""
    try:
        path = Path(path)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_benchmark(path: Optional[Path] = None) -> Dict[str, Any]:
    """Return the retrieval baseline, or ``{}`` when it holds no real numbers.

    The repository ships a scaffold (``n_queries: 0``, ``strategies: {}``) so a
    successful read is not enough — an empty baseline must behave like a missing
    one, otherwise the tab would claim a benchmark that never ran.
    """
    data = _read_json(path or BASELINE_PATH) or {}
    strategies = data.get("strategies")
    if not isinstance(strategies, dict) or not strategies:
        return {}
    if not _as_float(data.get("n_queries")):
        return {}
    return data


def load_eval_reports(reports_dir: Optional[Path] = None, limit: int = 20) -> List[dict]:
    """Return evaluation reports from disk, newest first.

    Files without a ``summary`` object (partial writes, unrelated JSON) are
    skipped rather than rendered as a run with zeroed metrics. Reports missing a
    timestamp fall back to the file's modification time so ordering stays real.
    """
    directory = Path(reports_dir or REPORTS_DIR)
    if not directory.is_dir():
        return []

    reports: List[dict] = []
    for path in directory.glob("*.json"):
        data = _read_json(path)
        if not data or not isinstance(data.get("summary"), dict):
            continue
        if not data.get("generated_at"):
            try:
                data["generated_at"] = datetime.fromtimestamp(
                    path.stat().st_mtime
                ).isoformat()
            except OSError:
                pass
        reports.append(data)

    reports.sort(key=_report_sort_key, reverse=True)
    return reports[:limit]


def merge_reports(session_report: Optional[dict], reports: Optional[List[dict]]) -> List[dict]:
    """Combine the in-session report with the reports on disk, newest first.

    A session report is prepended only when it carries a summary, and exact
    duplicates (same timestamp, collection and mode) are dropped so a run is
    never compared against itself.
    """
    candidates = ([session_report] if session_report else []) + list(reports or [])
    merged: List[dict] = []
    seen = set()
    for report in candidates:
        if not isinstance(report, dict) or not isinstance(report.get("summary"), dict):
            continue
        identity = (
            str(report.get("generated_at")),
            str(report.get("collection_name")),
            str(report.get("retrieval_mode")),
        )
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(report)
    merged.sort(key=_report_sort_key, reverse=True)
    return merged


def _report_sort_key(report: dict) -> float:
    stamp = report.get("generated_at")
    if isinstance(stamp, (int, float)):
        return float(stamp)
    try:
        return datetime.fromisoformat(str(stamp)).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _as_float(value) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_timestamp(stamp) -> str:
    """Render an ISO timestamp as ``YYYY-MM-DD HH:MM``; ``—`` when unusable."""
    if not stamp:
        return "—"
    try:
        return datetime.fromisoformat(str(stamp)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(stamp)


# ---------------------------------------------------------------------------
# Retrieval benchmark rows
# ---------------------------------------------------------------------------

def _strategy_metric(metrics: dict, prefix: str, top_k: int) -> Optional[float]:
    """Read ``<prefix>_at_<top_k>``, falling back to any ``<prefix>_at_*`` key."""
    value = metrics.get(f"{prefix}_at_{top_k}")
    if value is None:
        for key, candidate in metrics.items():
            if str(key).startswith(f"{prefix}_at_"):
                value = candidate
                break
    return _as_float(value)


def benchmark_rows(benchmark: Optional[dict]) -> List[dict]:
    """Turn a benchmark artifact into display rows; ``[]`` when it has no data."""
    strategies = (benchmark or {}).get("strategies")
    if not isinstance(strategies, dict) or not strategies:
        return []

    top_k = int(_as_float((benchmark or {}).get("top_k")) or 5)
    ordered = STRATEGY_ORDER + [name for name in strategies if name not in STRATEGY_ORDER]

    rows = []
    for name in ordered:
        metrics = strategies.get(name)
        if not isinstance(metrics, dict):
            continue
        recall = _strategy_metric(metrics, "recall", top_k)
        ndcg = _strategy_metric(metrics, "ndcg", top_k)
        if recall is None and ndcg is None:
            continue
        rows.append(
            {
                "key": name,
                "name": STRATEGY_LABELS.get(name, str(name).replace("_", " ").title()),
                "recall": recall,
                "ndcg": ndcg,
                "active": name == "hybrid",
            }
        )
    return rows


def benchmark_caption(benchmark: Optional[dict]) -> str:
    """One honest line of provenance for the benchmark section."""
    if not benchmark:
        return ""
    top_k = int(_as_float(benchmark.get("top_k")) or 5)
    parts = [
        f"Recall@{top_k} · nDCG@{top_k}",
        f"{int(_as_float(benchmark.get('n_queries')) or 0)} golden queries",
    ]
    if benchmark.get("collection"):
        parts.append(f"collection {benchmark['collection']}")
    parts.append(f"generated {fmt_timestamp(benchmark.get('generated_at'))}")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Headline metric cards
# ---------------------------------------------------------------------------

def _fmt_value(value: Optional[float], unit: str) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}s" if unit == "secs" else f"{value:.2f}"


def _previous_comparable(reports: List[dict], latest: dict) -> Optional[dict]:
    """The newest earlier report measured on the same collection and mode.

    Comparing a hybrid run against an MMR run on another collection would make
    the delta meaningless, so no such comparison is offered.
    """
    for report in reports[1:]:
        if (
            report.get("collection_name") == latest.get("collection_name")
            and report.get("retrieval_mode") == latest.get("retrieval_mode")
        ):
            return report
    return None


def _trend(current: float, prior: float, unit: str) -> dict:
    """Describe the change, colouring it by whether it is an improvement."""
    delta = current - prior
    tolerance = 0.01 if unit == "secs" else 0.005
    if abs(delta) < tolerance:
        return {"trend": "± unchanged vs previous run", "trend_cls": "dm-trend-stable"}

    suffix = "s" if unit == "secs" else ""
    arrow = "↑" if delta > 0 else "↓"
    improved = delta < 0 if unit == "secs" else delta > 0
    return {
        "trend": f"{arrow} {abs(delta):.2f}{suffix} vs previous run",
        "trend_cls": "dm-trend-up" if improved else "dm-trend-down",
    }


def metric_cards(reports: Optional[List[dict]]) -> List[dict]:
    """Build the four headline cards from real reports.

    ``reports`` must be newest first (see :func:`merge_reports`). Each card
    carries ``measured`` so the view can style an unmeasured metric as absent
    rather than showing a number.
    """
    reports = list(reports or [])
    latest = reports[0] if reports else None
    latest_summary = (latest or {}).get("summary") or {}
    previous = _previous_comparable(reports, latest) if latest else None
    previous_summary = (previous or {}).get("summary") or {}

    cards = []
    for spec in CARD_SPECS:
        current = _as_float(latest_summary.get(spec["key"]))
        prior = _as_float(previous_summary.get(spec["key"]))

        card = dict(spec)
        card["value"] = _fmt_value(current, spec["unit"])
        card["measured"] = current is not None
        card["trend"] = ""
        card["trend_cls"] = "dm-trend-stable"

        if current is None:
            card["trend"] = "not measured" if latest else "no runs yet"
        elif prior is None:
            card["trend"] = "first run" if latest else "no runs yet"
        else:
            card.update(_trend(current, prior, spec["unit"]))
        cards.append(card)
    return cards


def report_caption(report: Optional[dict]) -> str:
    """Provenance line for the headline cards."""
    if not report:
        return (
            "No evaluation data yet — run the batch evaluation below, or "
            "`python scripts/run_eval.py --collection <name>`."
        )
    parts = [f"Latest run {fmt_timestamp(report.get('generated_at'))}"]
    if report.get("collection_name"):
        parts.append(f"collection {report['collection_name']}")
    if report.get("retrieval_mode"):
        parts.append(f"mode {report['retrieval_mode']}")
    if report.get("top_k") is not None:
        parts.append(f"top-{report['top_k']}")
    if report.get("prompt_version"):
        parts.append(f"prompt {report['prompt_version']}")
    total = (report.get("summary") or {}).get("total_questions")
    if total:
        parts.append(f"{total} questions")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Recent-evaluations table
# ---------------------------------------------------------------------------

def _judge_score(llm_metrics, name: str) -> Optional[float]:
    """Read ``llm_metrics[name]["score"]`` (or a bare number), else ``None``.

    ``evaluate_one()`` sets ``llm_metrics`` to ``None`` when judging was not
    requested and to ``{"error": ...}`` when the judge call failed; neither is
    a measured score, so both read as missing instead of zero.
    """
    if not isinstance(llm_metrics, dict) or "error" in llm_metrics:
        return None
    entry = llm_metrics.get(name)
    if isinstance(entry, dict):
        entry = entry.get("score")
    return _as_float(entry)


def _truncate(text, width: int = QUESTION_WIDTH) -> str:
    """Collapse whitespace and shorten to ``width`` characters, with an ellipsis
    only when something was actually cut."""
    text = " ".join(str(text or "").split())
    if len(text) <= width:
        return text
    return text[: width - 1].rstrip() + "…"


def _citation_counts(row: dict) -> Optional[str]:
    """``"valid/total"`` from the counts ``evaluate_one()`` reports, else ``None``."""
    total = _as_float(row.get("citation_count"))
    if total is None:
        return None
    valid = _as_float(row.get("valid_citation_count")) or 0.0
    return f"{int(valid)}/{int(total)}"


def _row_status(faithfulness: Optional[float], citation: Optional[float],
                is_idk: bool) -> str:
    """✅ / ⚠️ / — from the scores that were actually measured.

    A refusal has no answer to ground, so it is not scored. Otherwise the run
    passes only when every measured quality score clears ``PASS_THRESHOLD`` —
    a faithful answer with invalid citations is still a weak run.
    """
    if is_idk:
        return "—"
    measured = [score for score in (faithfulness, citation) if score is not None]
    if not measured:
        return "—"
    return "✅" if min(measured) >= PASS_THRESHOLD else "⚠️"


def recent_eval_rows(results, limit: int = RECENT_LIMIT) -> List[dict]:
    """Turn ``evaluate_one()`` outputs into newest-first table rows.

    ``results`` is the session's append-ordered list. Every field is either the
    value the evaluator produced or ``None`` (rendered as "—"), never a default
    that would read as a real score.
    """
    items = [r for r in (results or []) if isinstance(r, dict)]
    if limit and limit > 0:
        items = items[-limit:]

    rows = []
    for row in reversed(items):
        faithfulness = _judge_score(row.get("llm_metrics"), "faithfulness")
        citation = _as_float(row.get("citation_validity_ratio"))
        rows.append(
            {
                "question": _truncate(row.get("question", "")),
                "faithfulness": faithfulness,
                "citation": citation,
                "citations": _citation_counts(row),
                "latency": _as_float(row.get("latency_sec")),
                "status": _row_status(
                    faithfulness, citation, bool(row.get("is_idk"))
                ),
            }
        )
    return rows


def recent_eval_caption(total: int, shown: int) -> str:
    """Provenance and legend for the recent-evaluations table."""
    label = "question" if total == 1 else "questions"
    parts = [f"This session · {total} evaluated {label}"]
    if shown < total:
        parts.append(f"showing the newest {shown}")
    parts.append(
        f"✅ ≥ {PASS_THRESHOLD:.2f} on every measured score · "
        "⚠️ below · — refusal or not measured"
    )
    return " · ".join(parts)
