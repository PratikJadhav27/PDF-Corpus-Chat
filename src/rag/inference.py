"""RAG inference helper functions.

This module contains the core retrieval‑generation‑validation pipeline that was
previously embedded in ``app/pages/chat.py``.  Keeping the logic here makes it
easier to unit‑test, reuse in other front‑ends (FastAPI, CLI), and keeps the
Streamlit page focused on UI rendering.
"""

import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

from src.rag.embeddings import get_embeddings
from src.rag.vector_store import load_vector_store
from src.rag.retriever import retrieve_docs, retrieve_hybrid_reranked
from src.rag.prompting import build_context_with_citations, build_grounded_prompt
from src.rag.prompt_registry import DEFAULT_PROMPT_VERSION, PROMPTS, resolve_prompt_version
from src.rag.usage import accumulate_usage, usage_from_message
from src.rag.validators import guard_answer_with_citation_check, citation_validity
from src.observability.langfuse_guard import observe, CallbackHandler, langfuse_flush
from src.observability.tracing import (
    append_rag_trace,
    build_trace_record,
    new_trace_id,
)

# LLM model name is read from environment (default matches previous hard‑coded value).
import os

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")


@observe()
def run_rag_query(
    user_question: str,
    prompt_version: str,
    temperature: float,
    collection_name: str,
) -> Tuple[Optional[str], Optional[List], Dict[str, float]]:
    """Full RAG pipeline.

    ``prompt_version`` is the single source of truth for prompt style: it is
    used to build the prompt *and* written to the trace, so the recorded
    version is always the one that actually ran. Unknown/legacy versions
    resolve to the default (see ``resolve_prompt_version``).

    Returns ``(answer, docs, latency)`` where:
    - ``answer``  is the generated string (or None on fatal error)
    - ``docs``    is the list of retrieved Document objects (or None on fatal error)
    - ``latency`` is a dict with keys total_sec, retrieval_sec, llm_sec
    """
    _NO_LATENCY = {"total_sec": 0.0, "retrieval_sec": 0.0, "llm_sec": 0.0}
    trace_id = new_trace_id()
    t0 = time.time()
    resolved_prompt_version = resolve_prompt_version(prompt_version)

    embeddings = get_embeddings()
    db = load_vector_store(collection_name, embeddings)

    if db is None:
        append_rag_trace(
            build_trace_record(
                trace_id=trace_id,
                phase="chat",
                collection=collection_name,
                query=user_question,
                retrieval_mode=st.session_state.retrieval_mode,
                prompt_version=resolved_prompt_version,
                model_name=LLM_MODEL_NAME,
                temperature=temperature,
                top_k=st.session_state.top_k,
                candidate_k=(
                    st.session_state.candidate_k
                    if st.session_state.retrieval_mode == "hybrid"
                    else None
                ),
                rerank_top_n=(
                    st.session_state.rerank_top_n
                    if st.session_state.retrieval_mode == "hybrid"
                    else None
                ),
                docs=[],
                latency_sec={"total_sec": round(time.time() - t0, 4)},
                error="vector_store_missing",
            )
        )
        st.error(
            f"Please upload and process PDF files first in collection '{collection_name}'!"
        )
        return None, None, _NO_LATENCY

    # --- Retrieval ---
    t_r0 = time.time()
    if st.session_state.retrieval_mode == "hybrid":
        docs = retrieve_hybrid_reranked(
            db=db,
            collection_name=collection_name,
            query=user_question,
            top_k=st.session_state.rerank_top_n,
            candidate_k=st.session_state.candidate_k,
        )
    else:
        docs = retrieve_docs(
            db,
            user_question,
            mode=st.session_state.retrieval_mode,
            k=st.session_state.top_k,
        )
    retrieval_sec = round(time.time() - t_r0, 4)

    if not docs:
        response = "I don't know based on the provided documents."
        append_rag_trace(
            build_trace_record(
                trace_id=trace_id,
                phase="chat",
                collection=collection_name,
                query=user_question,
                retrieval_mode=st.session_state.retrieval_mode,
                prompt_version=resolved_prompt_version,
                model_name=LLM_MODEL_NAME,
                temperature=temperature,
                top_k=st.session_state.top_k,
                candidate_k=(
                    st.session_state.candidate_k
                    if st.session_state.retrieval_mode == "hybrid"
                    else None
                ),
                rerank_top_n=(
                    st.session_state.rerank_top_n
                    if st.session_state.retrieval_mode == "hybrid"
                    else None
                ),
                docs=[],
                latency_sec={
                    "retrieval_sec": retrieval_sec,
                    "total_sec": round(time.time() - t0, 4),
                },
                answer_preview=response[:500],
            )
        )
        return response, [], {"total_sec": round(time.time() - t0, 4), "retrieval_sec": retrieval_sec, "llm_sec": 0.0}

    # --- Generation ---
    context = build_context_with_citations(docs)
    prompt = build_grounded_prompt(
        context=context, question=user_question, version=resolved_prompt_version
    )
    model = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=temperature)

    t_l0 = time.time()
    try:
        langfuse_handler = CallbackHandler()
        # Keep the whole message: the text is the answer, ``usage_metadata`` is
        # the only place the provider reports what the call actually cost.
        reply = model.invoke(prompt, config={"callbacks": [langfuse_handler]})
        response = reply.content
        usage = usage_from_message(reply)
    except Exception as e:
        append_rag_trace(
            build_trace_record(
                trace_id=trace_id,
                phase="chat",
                collection=collection_name,
                query=user_question,
                retrieval_mode=st.session_state.retrieval_mode,
                prompt_version=resolved_prompt_version,
                model_name=LLM_MODEL_NAME,
                temperature=temperature,
                top_k=st.session_state.top_k,
                candidate_k=(
                    st.session_state.candidate_k
                    if st.session_state.retrieval_mode == "hybrid"
                    else None
                ),
                rerank_top_n=(
                    st.session_state.rerank_top_n
                    if st.session_state.retrieval_mode == "hybrid"
                    else None
                ),
                docs=docs,
                latency_sec={
                    "retrieval_sec": retrieval_sec,
                    "llm_sec": round(time.time() - t_l0, 4),
                    "total_sec": round(time.time() - t0, 4),
                },
                error=str(e),
            )
        )
        st.error(f"Model error: {e}")
        return None, None, _NO_LATENCY
    llm_sec = round(time.time() - t_l0, 4)

    # ISSUE-013: record the billable tokens before the answer is post-processed,
    # so the session's Usage & Cost panel reflects calls that really happened.
    accumulate_usage(st.session_state, usage)

    response = guard_answer_with_citation_check(response, docs, min_validity_ratio=0.8)
    citation_stats = citation_validity(response, docs)

    append_rag_trace(
        build_trace_record(
            trace_id=trace_id,
            phase="chat",
            collection=collection_name,
            query=user_question,
            retrieval_mode=st.session_state.retrieval_mode,
            prompt_version=resolved_prompt_version,
            model_name=LLM_MODEL_NAME,
            temperature=temperature,
            top_k=st.session_state.top_k,
            candidate_k=(
                st.session_state.candidate_k
                if st.session_state.retrieval_mode == "hybrid"
                else None
            ),
            rerank_top_n=(
                st.session_state.rerank_top_n
                if st.session_state.retrieval_mode == "hybrid"
                else None
            ),
            docs=docs,
            latency_sec={
                "retrieval_sec": retrieval_sec,
                "llm_sec": llm_sec,
                "total_sec": round(time.time() - t0, 4),
            },
            citation_validity_ratio=citation_stats.get("validity_ratio", 0.0),
            answer_preview=(response or "")[:500],
        )
    )
    langfuse_flush()
    latency = {"total_sec": round(time.time() - t0, 4), "retrieval_sec": retrieval_sec, "llm_sec": llm_sec}
    return response, docs, latency
