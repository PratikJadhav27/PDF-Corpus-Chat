"""
app/views/evaluation.py

Tab 2 — Analytics & Evaluation Dashboard.

Shows:
  - 4 metric cards read from the most recent real evaluation run
  - Retrieval strategy benchmark read from evals/baselines/latest.json
  - Recent evaluation runs table (this session)
  - In-app batch evaluation runner (collapsed expander)

Every value on this tab comes from an artifact written by an actual run
(ISSUE-005). Unmeasured metrics render as "—" with a reason, and missing
artifacts render an empty state with the command that produces them — the tab
never falls back to demo numbers.
"""
from datetime import datetime

import streamlit as st

from app import analytics
from src.eval.evaluator import evaluate_one, build_report
from src.rag.inference import run_rag_query
from src.rag.prompt_registry import DEFAULT_PROMPT_VERSION
from src.utils.json_export import to_pretty_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_score(value) -> str:
    """Format an optional 0-1 score; "—" when the metric was not computed."""
    return "—" if value is None else f"{value:.2f}"


def _render_eval_report(report: dict):
    """Render the persisted report for the most recent in-app batch evaluation."""
    summary = report.get("summary", {})
    st.success(f"✅ {summary.get('total_questions', 0)} questions evaluated")
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Avg Faithfulness", _fmt_score(summary.get("avg_faithfulness")))
    rc2.metric("Avg Citation Valid.", _fmt_score(summary.get("avg_citation_validity_ratio")))
    rc3.metric("Citation Rate", f"{summary.get('citation_rate', 0):.0f}%")
    st.caption(
        f"Collection: {report.get('collection_name', '—')} · "
        f"mode: {report.get('retrieval_mode', '—')} · "
        f"top-k: {report.get('top_k', '—')} · "
        f"prompt: {report.get('prompt_version', '—')}"
    )
    st.download_button(
        "⬇️ Export JSON Report",
        data=to_pretty_json(report),
        file_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )


def _metric_card(col, card: dict):
    """Render one headline card; unmeasured metrics are greyed out, not faked."""
    with col:
        value_color = card["color"] if card["measured"] else "#64748B"
        trend_html = (
            f'<span class="dm-trend {card["trend_cls"]}">{card["trend"]}</span>'
            if card["trend"]
            else ""
        )
        st.markdown(
            f"""
            <div class="dm-metric-card">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                        <div class="dm-metric-label">{card["label"]}</div>
                        <div class="dm-metric-value" style="color:{value_color};">{card["value"]}</div>
                    </div>
                    <span style="font-size:1.6rem;opacity:0.5;">{card["icon"]}</span>
                </div>
                {trend_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _score_text(value, as_percent: bool) -> str:
    if value is None:
        return "—"
    return f"{value:.1%}" if as_percent else f"{value:.3f}"


def _benchmark_section(benchmark: dict):
    """Render the retrieval strategy benchmark from the recorded baseline."""
    st.markdown('<div class="dm-section">🏆 Retrieval Strategy Benchmark</div>', unsafe_allow_html=True)
    rows = analytics.benchmark_rows(benchmark)

    if not rows:
        st.markdown(
            '<div class="dm-empty"><div class="icon">🏆</div>'
            'No retrieval benchmark has been recorded yet.<br>'
            'Run the benchmark script to measure Recall@k and nDCG@k.</div>',
            unsafe_allow_html=True,
        )
        st.code(analytics.BENCHMARK_COMMAND, language="bash")
        return

    st.caption(analytics.benchmark_caption(benchmark))
    for row in rows:
        active_tag = " ✦ Production" if row["active"] else ""
        st.markdown(
            f"""
            <div class="dm-bench-label">
                <span>{row["name"]}<span style="color:#10B981;font-size:0.68rem;margin-left:6px;">{active_tag}</span></span>
                <span class="score">Recall {_score_text(row["recall"], True)} · nDCG {_score_text(row["ndcg"], False)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if row["recall"] is not None:
            st.progress(min(max(row["recall"], 0.0), 1.0), text="")


def _recent_evals_section():
    """Show this session's evaluated questions, or an honest empty state.

    Rows come from ``analytics.recent_eval_rows``, which reads the keys
    ``evaluate_one()`` actually writes (``citation_validity_ratio`` and
    ``llm_metrics``) instead of the old ``faithfulness_score`` /
    ``citation_validity.validity_ratio`` pair that never existed (ISSUE-006).
    """
    st.markdown('<div class="dm-section">📋 Recent Evaluations</div>', unsafe_allow_html=True)
    results = st.session_state.get("eval_results", [])
    if not isinstance(results, list) or not results:
        st.markdown(
            '<div class="dm-empty"><div class="icon">📊</div>'
            'No evaluation runs yet.<br>Run a batch evaluation below.</div>',
            unsafe_allow_html=True,
        )
        return

    import pandas as pd
    rows = analytics.recent_eval_rows(results)
    table = []
    for row in rows:
        citation = _fmt_score(row["citation"])
        if row["citations"] and citation != "—":
            citation = f"{citation} ({row['citations']})"
        table.append({
            "Question":     row["question"],
            "Faithfulness": _fmt_score(row["faithfulness"]),
            "Citation":     citation,
            "Latency":      "—" if row["latency"] is None else f"{row['latency']:.2f}s",
            "Status":       row["status"],
        })

    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    st.caption(analytics.recent_eval_caption(len(results), len(rows)))


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def render_evaluation_tab(collections):
    st.markdown(
        """
        <div class="dm-page-header">
            <h2>Analytics &amp; Evaluation</h2>
            <p>RAG pipeline performance metrics · grounding verification · retrieval benchmarks</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Real runs only: the in-session batch report (if any) plus the artifacts
    # the CLI/CI wrote to evals/reports/, newest first.
    reports = analytics.merge_reports(
        st.session_state.get("eval_report"),
        analytics.load_eval_reports(),
    )

    # ── 4 Metric cards ────────────────────────────────────────────────────
    cards = analytics.metric_cards(reports)
    cols = st.columns(4)
    for col, card in zip(cols, cards):
        _metric_card(col, card)
    st.caption(analytics.report_caption(reports[0] if reports else None))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Benchmark, then the recent-evaluations table at full width ────────
    # The table carries question, faithfulness, citation, latency and status;
    # in a 40% column the measured columns fell outside the visible area.
    _benchmark_section(analytics.load_benchmark())

    st.markdown("<br>", unsafe_allow_html=True)
    _recent_evals_section()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Batch evaluator ───────────────────────────────────────────────────
    with st.expander("🧪 Run Batch Evaluation", expanded=False):
        eval_collection = st.selectbox("Collection", collections, key="eval_collection")
        eval_questions_text = st.text_area(
            "Questions (one per line)",
            height=130,
            placeholder="What is the main topic?\nSummarize the conclusion.\nList key requirements.",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            eval_mode = st.selectbox("Mode", ["hybrid", "similarity", "mmr"], key="eval_mode")
        with col_b:
            eval_temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1, key="eval_temp")
        # Batch runs use the same prompt style as Chat (ISSUE-004). The old
        # hard-coded "standard" version no longer exists and silently fell back.
        eval_prompt_version = st.session_state.get(
            "prompt_version", DEFAULT_PROMPT_VERSION
        )
        include_llm_metrics = st.checkbox(
            "Include LLM-judge metrics (faithfulness, relevance, completeness)",
            value=False,
            key="eval_include_llm_metrics",
            help="Adds three Gemini judge calls per question — slower and uses API quota.",
        )

        if st.button("🚀 Run Evaluation", type="primary"):
            questions = [q.strip() for q in eval_questions_text.split("\n") if q.strip()]
            if not questions:
                st.warning("Add at least one question first.")
            else:
                results = []
                bar = st.progress(0)
                status_txt = st.empty()
                for i, q in enumerate(questions):
                    status_txt.caption(f"Evaluating {i+1}/{len(questions)}: {q[:60]}…")
                    st.session_state.retrieval_mode  = eval_mode
                    st.session_state.top_k           = 4
                    st.session_state.candidate_k     = 12
                    st.session_state.rerank_top_n    = 4
                    answer, docs, latency = run_rag_query(
                        q, eval_prompt_version, eval_temp, eval_collection
                    )
                    if answer:
                        ev = evaluate_one(
                            q, answer, docs or [], include_llm_metrics=include_llm_metrics
                        )
                        ev["question"]    = q
                        ev["latency_sec"] = latency.get("total_sec", 0)
                        results.append(ev)
                    bar.progress((i + 1) / len(questions))

                status_txt.empty()
                if results:
                    report = build_report(
                        collection_name=eval_collection,
                        retrieval_mode=eval_mode,
                        top_k=st.session_state.top_k,
                        results=results,
                    )
                    report["prompt_version"] = eval_prompt_version
                    st.session_state["eval_results"] = results
                    st.session_state["eval_report"] = report
                else:
                    st.session_state.pop("eval_report", None)
                    st.warning("No answers were produced — check the model configuration.")

        # Report lives in session state so it survives reruns instead of being
        # wiped by an immediate st.rerun() (which used to hide success + export).
        if st.session_state.get("eval_report"):
            _render_eval_report(st.session_state["eval_report"])
