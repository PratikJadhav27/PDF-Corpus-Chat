"""
app/pages/evaluation.py

Tab 7 — In-app RAG Evaluation Mode.

Allows running batch evaluation questions against a collection from the UI,
displaying a summary and exporting a JSON report.
"""
import time

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

from src.rag.embeddings import get_embeddings
from src.rag.vector_store import load_vector_store
from src.rag.retriever import retrieve_docs, retrieve_hybrid_reranked
from src.rag.prompting import build_context_with_citations, build_grounded_prompt
from src.rag.validators import guard_answer_with_citation_check, citation_validity
from src.observability.tracing import append_rag_trace, build_trace_record, new_trace_id
from src.eval.evaluator import evaluate_one, build_report
from src.utils.json_export import to_pretty_json
from datetime import datetime

LLM_MODEL_NAME = "gemini-2.5-flash"


def render_evaluation_tab(collections):
    st.subheader("🧪 RAG Evaluation Mode")
    st.caption("Run batch questions to measure grounding quality and export results.")

    eval_collection = st.selectbox("Select Collection", collections, key="eval_collection")

    eval_questions_text = st.text_area(
        "Evaluation Questions (one per line)", height=180,
        placeholder="Example:\nWhat is the main topic of the document?\nList key requirements mentioned.\nSummarize the conclusion.",
    )

    colA, colB, colC = st.columns(3)
    with colA:
        eval_mode = st.selectbox("Retrieval Mode", ["similarity", "mmr", "hybrid"], index=0, key="eval_mode")
    with colB:
        eval_top_k = st.slider("Top-K", 2, 10, 4, 1, key="eval_top_k")
    with colC:
        eval_temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1, key="eval_temp")

    if not st.button("🚀 Run Evaluation", type="primary"):
        return

    questions = [q.strip() for q in eval_questions_text.split("\n") if q.strip()]
    if not questions:
        st.error("Please enter at least 1 evaluation question.")
        return

    embeddings = get_embeddings()
    db = load_vector_store(eval_collection, embeddings)
    if db is None:
        st.error("This collection is empty. Upload and process PDFs first.")
        return

    results = []
    progress = st.progress(0)
    candidate_k = max(12, eval_top_k * 3)

    for idx, q in enumerate(questions):
        eval_trace_id = new_trace_id()
        start = time.time()
        t_r0 = time.time()

        if eval_mode == "hybrid":
            docs = retrieve_hybrid_reranked(db=db, collection_name=eval_collection, query=q, top_k=eval_top_k, candidate_k=candidate_k)
        else:
            docs = retrieve_docs(db, q, mode=eval_mode, k=eval_top_k)
        retrieval_sec = round(time.time() - t_r0, 4)

        if not docs:
            answer = "I don't know based on the provided documents."
            append_rag_trace(build_trace_record(
                trace_id=eval_trace_id, phase="eval", collection=eval_collection,
                query=q, retrieval_mode=eval_mode, prompt_version=st.session_state.prompt_version,
                model_name=LLM_MODEL_NAME, temperature=eval_temperature, top_k=eval_top_k,
                candidate_k=candidate_k if eval_mode == "hybrid" else None,
                rerank_top_n=eval_top_k if eval_mode == "hybrid" else None,
                docs=[], latency_sec={"retrieval_sec": retrieval_sec, "total_sec": round(time.time() - start, 4)},
                answer_preview=answer[:500], eval_index=idx + 1, eval_total=len(questions),
            ))
        else:
            context = build_context_with_citations(docs)
            prompt = build_grounded_prompt(context=context, question=q, version=st.session_state.prompt_version)
            model = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=eval_temperature)
            t_l0 = time.time()
            try:
                answer = model.invoke(prompt).content
            except Exception as e:
                append_rag_trace(build_trace_record(
                    trace_id=eval_trace_id, phase="eval", collection=eval_collection,
                    query=q, retrieval_mode=eval_mode, prompt_version=st.session_state.prompt_version,
                    model_name=LLM_MODEL_NAME, temperature=eval_temperature, top_k=eval_top_k,
                    candidate_k=candidate_k if eval_mode == "hybrid" else None,
                    rerank_top_n=eval_top_k if eval_mode == "hybrid" else None,
                    docs=docs,
                    latency_sec={"retrieval_sec": retrieval_sec, "llm_sec": round(time.time() - t_l0, 4), "total_sec": round(time.time() - start, 4)},
                    error=str(e), eval_index=idx + 1, eval_total=len(questions),
                ))
                raise
            llm_sec = round(time.time() - t_l0, 4)
            answer = guard_answer_with_citation_check(answer, docs, min_validity_ratio=0.8)
            cv = citation_validity(answer, docs)
            append_rag_trace(build_trace_record(
                trace_id=eval_trace_id, phase="eval", collection=eval_collection,
                query=q, retrieval_mode=eval_mode, prompt_version=st.session_state.prompt_version,
                model_name=LLM_MODEL_NAME, temperature=eval_temperature, top_k=eval_top_k,
                candidate_k=candidate_k if eval_mode == "hybrid" else None,
                rerank_top_n=eval_top_k if eval_mode == "hybrid" else None,
                docs=docs,
                latency_sec={"retrieval_sec": retrieval_sec, "llm_sec": llm_sec, "total_sec": round(time.time() - start, 4)},
                citation_validity_ratio=cv.get("validity_ratio", 0.0),
                answer_preview=(answer or "")[:500], eval_index=idx + 1, eval_total=len(questions),
            ))

        latency = round(time.time() - start, 3)
        row = evaluate_one(q, answer, docs)
        row["latency_sec"] = latency
        results.append(row)
        progress.progress((idx + 1) / len(questions))

    report = build_report(collection_name=eval_collection, retrieval_mode=eval_mode, top_k=eval_top_k, results=results)
    st.success("✅ Evaluation complete!")

    st.write("## 📊 Summary")
    summary = report["summary"]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total Questions", summary["total_questions"])
    with c2: st.metric("Citation Rate", f"{summary['citation_rate']}%")
    with c3: st.metric("IDK Rate", f"{summary['idk_rate']}%")
    with c4: st.metric("Avg Latency", f"{summary['avg_latency_sec']}s")
    with c5: st.metric("Avg Citation Validity", summary.get("avg_citation_validity_ratio", 0.0))

    st.write("---")
    st.write("## 📋 Results")
    for i, r in enumerate(report["results"], 1):
        with st.expander(f"Q{i}: {r['question']}"):
            st.write("**Answer:**")
            st.write(r["answer"])
            st.write("**Has Citations:**", r["has_citations"])
            st.write("**IDK Answer:**", r["is_idk"])
            with st.expander("Retrieved Sources"):
                for s in r.get("sources", []):
                    st.write(f"- {s.get('source')} | page {s.get('page')} | chunk {s.get('chunk_index')}")

    st.write("---")
    st.write("## 📥 Export Report")
    st.download_button(
        label="Download JSON Report",
        data=to_pretty_json(report),
        file_name=f"rag_eval_report_{eval_collection}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )
