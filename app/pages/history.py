"""
app/pages/history.py

Tab 3 — Conversation History
Tab 4 — Advanced Search (semantic only; keyword search removed — requires PyPDF2)
Tab 5 — Batch Q&A (translation and auto-QA generation removed as dead code)
"""
import json
from datetime import datetime

import streamlit as st
from src.rag.embeddings import get_embeddings
from src.rag.vector_store import load_vector_store
from src.rag.prompt_registry import DEFAULT_PROMPT_VERSION
from app.pages.chat import run_rag_query


def render_history_tab():
    st.subheader("📜 Conversation History")

    if st.session_state.chat_history:
        available_collections = list(
            set(chat.get("collection", "default") for chat in st.session_state.chat_history)
        )
        filter_collection = st.selectbox("Filter by Collection", ["All"] + available_collections)

        filtered = (
            st.session_state.chat_history
            if filter_collection == "All"
            else [c for c in st.session_state.chat_history if c.get("collection") == filter_collection]
        )

        for i, chat in enumerate(reversed(filtered), 1):
            with st.expander(f"💬 Q{len(filtered) - i + 1}: {chat['question'][:60]}..."):
                st.write(f"**Time:** {chat['timestamp']}")
                if chat.get("trace_id"):
                    st.caption(f"trace_id: {chat['trace_id']}")
                st.write(f"**Type:** {chat['prompt_type']} | **Collection:** {chat.get('collection', 'default')}")
                st.write(f"**Prompt Version:** {chat.get('prompt_version', DEFAULT_PROMPT_VERSION)}")
                st.write(f"**Tokens:** {chat.get('tokens_used', 0)}")
                st.write(f"**Citation Validity Ratio:** {chat.get('citation_validity_ratio', 0.0)}")
                st.write(f"\n**Question:**\n{chat['question']}")
                st.write(f"\n**Answer:**\n{chat['answer']}")

                with st.expander("Sources"):
                    for s in chat.get("sources", []):
                        st.write(f"- {s.get('source', 'Unknown')} | Page {s.get('page', '?')} | Chunk {s.get('chunk_index', '?')}")
                        dbg = s.get("retrieval")
                        if dbg:
                            st.caption(
                                f"rrf={dbg.get('rrf_score')} | rerank={dbg.get('rerank_score')} | "
                                f"vector={dbg.get('vector_score')} | bm25={dbg.get('bm25_score')}"
                            )

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(filtered, indent=2),
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        with c2:
            txt_content = "\n\n".join(
                f"Q: {chat['question']}\n\nA: {chat['answer']}\n\n{'='*80}" for chat in filtered
            )
            st.download_button(
                label="📥 Download TXT",
                data=txt_content,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )
    else:
        st.info("📌 No chat history yet. Start asking questions!")


def render_search_tab(collections):
    st.subheader("🔍 Advanced Search Features")

    # Keyword search required PyPDF2 (removed in Phase 0) — semantic only.
    st.write("Semantic search using vector store")

    selected_collection = st.selectbox("Select Collection", collections, key="search_collection")
    semantic_query = st.text_input("Enter your semantic search query")
    num_results = st.slider("Number of results", 1, 10, 5)

    if st.button("🔍 Semantic Search"):
        embeddings = get_embeddings()
        db = load_vector_store(selected_collection, embeddings)

        if db is None:
            st.error(f"Collection '{selected_collection}' is empty")
        else:
            results = db.similarity_search(semantic_query, k=num_results)
            st.success(f"Found {len(results)} results")
            for i, result in enumerate(results, 1):
                src = result.metadata.get("source", "Unknown")
                page = result.metadata.get("page_number", "?")
                with st.expander(f"Result {i} | {src} | Page {page}"):
                    st.write(result.page_content[:700] + "...")


def render_batch_tab():
    st.subheader("🌐 Batch Q&A")
    st.write("Ask multiple questions at once against the active collection.")

    questions_text = st.text_area("Enter questions (one per line)", height=150)

    if st.button("📤 Process Batch"):
        if questions_text:
            questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
            progress_bar = st.progress(0)
            results_list = []

            for idx, q in enumerate(questions):
                with st.spinner(f"Processing Q{idx+1}/{len(questions)}..."):
                    result = run_rag_query(q, "standard", 0.3, st.session_state.current_collection)
                    if result[0]:
                        results_list.append({"question": q, "answer": result[0]})
                progress_bar.progress((idx + 1) / len(questions))

            st.success(f"Processed {len(results_list)} questions!")
            for i, item in enumerate(results_list, 1):
                with st.expander(f"Q{i}: {item['question']}"):
                    st.write(item["answer"])
