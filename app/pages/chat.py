"""
app/pages/chat.py

Tab 1 — Interactive Chat (retrieval + generation + feedback).
Also owns the sidebar PDF upload and collection creation panel.

The core RAG pipeline (run_rag_query) lives in src.rag.inference so it
can be unit-tested and reused outside of Streamlit.
"""
import os

import streamlit as st

from src.config import CHUNKING
from src.rag.inference import run_rag_query
from src.rag.prompt_registry import DEFAULT_PROMPT_VERSION, PROMPTS
from app.helpers import get_collections, process_and_index_pdfs


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _render_citations(docs) -> str:
    citations = []
    for d in docs:
        src = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page_number", "?")
        citations.append(f"- **{src}** | page **{page}**")
    return "\n".join(sorted(set(citations)))


# ---------------------------------------------------------------------------
# Tab 1 renderer
# ---------------------------------------------------------------------------


def render_chat_tab(collections):
    st.subheader("💬 Interactive Chat")

    col1, _ = st.columns([2, 1])
    with col1:
        selected_collection = st.selectbox("Select Document Collection", collections, key="chat_collection")
        st.session_state.current_collection = selected_collection

    user_question = st.text_area("Ask a Question from the PDF Files uploaded .. ✍️📝", height=80)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        ask_button = st.button("🔍 Ask", use_container_width=True)
    with col3:
        if st.button("🔄 Clear", use_container_width=True):
            st.rerun()

    st.markdown("---")
    st.subheader("⚙️ Settings")
    st.markdown("### 🔎 Retrieval Settings")

    colA, colB = st.columns(2)
    with colA:
        st.session_state.retrieval_mode = st.selectbox(
            "Retrieval Mode", ["similarity", "mmr", "hybrid"],
            index=["similarity", "mmr", "hybrid"].index(st.session_state.retrieval_mode)
            if st.session_state.retrieval_mode in ["similarity", "mmr", "hybrid"] else 0,
            help="Hybrid uses vector + BM25 + RRF + reranking.",
        )
    with colB:
        st.session_state.top_k = st.slider(
            "Top-K Chunks", min_value=2, max_value=10,
            value=int(st.session_state.top_k), step=1,
            help="Higher K = more context, but may increase noise/cost.",
        )

    if st.session_state.retrieval_mode == "hybrid":
        colH1, colH2 = st.columns(2)
        with colH1:
            st.session_state.candidate_k = st.slider(
                "Hybrid Candidate-K", min_value=6, max_value=30,
                value=int(st.session_state.candidate_k), step=2,
                help="Candidates from each retriever before fusion/rerank.",
            )
        with colH2:
            st.session_state.rerank_top_n = st.slider(
                "Rerank Top-N", min_value=2, max_value=10,
                value=int(st.session_state.rerank_top_n), step=1,
                help="Final chunks after reranking.",
            )

    col1, col2 = st.columns(2)
    with col1:
        prompt_type = st.selectbox("Response Style", ["standard", "summarize", "detailed"])
    with col2:
        temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.3, 0.1)

    st.session_state.prompt_version = st.selectbox(
        "Prompt Version", list(PROMPTS.keys()),
        index=list(PROMPTS.keys()).index(st.session_state.prompt_version)
        if st.session_state.prompt_version in PROMPTS else 0,
        help="Versioned grounded prompt templates.",
    )

    st.info(
        f"📦 Collection: **{selected_collection}** | "
        f"Chunking: **{CHUNKING.chunk_size}/{CHUNKING.chunk_overlap}** | "
        f"Mode: **{st.session_state.retrieval_mode}** | Top-k: **{st.session_state.top_k}**"
    )

    if ask_button and user_question:
        result = run_rag_query(user_question, prompt_type, temperature, selected_collection)
        if result[0] is not None:
            response, docs = result

            st.subheader("📝 Response")
            st.write(response)

            st.subheader("📌 Citations")
            st.markdown(_render_citations(docs))

            with st.expander("📍 View Sources"):
                for i, doc in enumerate(docs, 1):
                    src = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page_number", "?")
                    chunk_idx = doc.metadata.get("chunk_index", "?")
                    with st.expander(f"Source {i} | {src} | Page {page}"):
                        st.caption(f"Chunk Index: {chunk_idx}")
                        st.text_area(f"Excerpt {i}", value=doc.page_content[:800] + "...", height=160, disabled=True)

            st.markdown("---")
            st.subheader("📋 Was this answer helpful?")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("👍 Helpful"):
                    st.session_state.feedback_data.append({"question": user_question, "helpful": True, "timestamp": datetime.now().isoformat()})
                    st.success("Thank you for the feedback!")
            with c2:
                if st.button("👎 Not Helpful"):
                    st.session_state.feedback_data.append({"question": user_question, "helpful": False, "timestamp": datetime.now().isoformat()})
                    st.info("We'll improve!")
            with c3:
                feedback_text = st.text_input("Additional feedback (optional)")
                if feedback_text:
                    st.session_state.feedback_data.append({"question": user_question, "feedback": feedback_text, "timestamp": datetime.now().isoformat()})


# ---------------------------------------------------------------------------
# Sidebar renderer
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.title("📁 PDF Management")
        st.subheader("Create or Select Collection")

        new_collection = st.text_input("New Collection Name (optional)", placeholder="e.g., 'Project A'", key="new_collection_input")
        if new_collection and new_collection.strip():
            if st.button("✅ Create Collection", key="create_collection_btn"):
                import os
                safe = new_collection.strip().replace(" ", "_")
                os.makedirs(f"vector_stores/{safe}", exist_ok=True)
                os.makedirs(f"metadata/{safe}", exist_ok=True)
                st.session_state.current_collection = safe
                st.success(f"✅ Collection '{new_collection}' created!")
                st.rerun()

        st.subheader("Select Active Collection")
        updated_collections = get_collections() or ["default"]
        selected_active = st.selectbox(
            "Active Collection", updated_collections,
            index=updated_collections.index(st.session_state.current_collection)
            if st.session_state.current_collection in updated_collections else 0,
            key="sidebar_collection_select",
        )
        st.session_state.current_collection = selected_active
        st.info(f"📍 Active: **{selected_active}**")

        st.write("---")
        st.write("### 📤 Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf")
        st.caption(f"Chunk size: {CHUNKING.chunk_size} | Overlap: {CHUNKING.chunk_overlap}")

        if st.button("Submit & Process", type="primary"):
            if pdf_docs:
                with st.spinner("Processing and indexing..."):
                    result = process_and_index_pdfs(pdf_docs, st.session_state.current_collection)
                    if result["ok"]:
                        st.success(f"✅ Done! Processed {len(pdf_docs)} PDF(s) in '{st.session_state.current_collection}'")
                    else:
                        st.error(result["message"])

                    stats = result.get("stats")
                    if stats:
                        overall = stats.get("overall", {})
                        st.markdown("### 📊 Extraction Stats")
                        st.write(
                            f"**Success Rate:** {overall.get('success_rate_percent', 0)}% "
                            f"({overall.get('extracted_pages', 0)}/{overall.get('total_pages', 0)} pages extracted)"
                        )
                        c1, c2, c3 = st.columns(3)
                        with c1: st.metric("📄 Total Pages", overall.get("total_pages", 0))
                        with c2: st.metric("✅ Extracted Pages", overall.get("extracted_pages", 0))
                        with c3: st.metric("⚠️ Failed Pages", overall.get("failed_pages", 0))
                        with st.expander("📁 Per-file breakdown"):
                            for f in stats.get("per_file", []):
                                st.write(
                                    f"**{f.get('file_name')}** → "
                                    f"{f.get('extracted_pages')}/{f.get('total_pages')} extracted "
                                    f"({f.get('extraction_success_rate')}%) | "
                                    f"Empty: {f.get('empty_pages')} | Failed: {f.get('failed_pages')}"
                                )
            else:
                st.error("Please upload at least one PDF file")

        st.write("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()

        st.write("---")
        try:
            st.image("img/Robot.jpg", use_column_width=True)
        except Exception:
            pass
        st.caption("AI App created by @ Pratik Jadhav")
