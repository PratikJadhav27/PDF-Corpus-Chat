"""
app/views/chat.py

Tab 1 — Interactive Chat (conversation thread + source cards).
Also owns the sidebar (PDF upload, collection management, advanced settings).

UI design: DocMind dark theme.
  - st.chat_input()  → sticky bottom input bar
  - st.chat_message() → user/AI message bubbles (styled via CSS in app/styles.py)
  - Source cards rendered as HTML below each AI answer
  - Latency badge shown on every AI response
  - Advanced settings collapsed in sidebar by default
"""
import os
from datetime import datetime

import streamlit as st

from src.config import CHUNKING
from src.rag.inference import run_rag_query
from src.rag.prompt_registry import DEFAULT_PROMPT_VERSION, PROMPTS
from app.helpers import get_collections, process_and_index_pdfs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_cards_html(docs) -> str:
    """Render horizontal-scrolling source excerpt cards as HTML."""
    if not docs:
        return ""
    cards = []
    for doc in docs:
        src   = doc.metadata.get("source", "Unknown")
        fname = os.path.basename(src)
        page  = doc.metadata.get("page_number", "?")
        score = doc.metadata.get("rerank_score") or doc.metadata.get("score")
        score_html = (
            f'<div class="dm-source-card-score">▲ {score:.1%} match</div>'
            if score is not None else ""
        )
        excerpt = doc.page_content[:200].replace("<", "&lt;").replace(">", "&gt;")
        cards.append(f"""
        <div class="dm-source-card">
            <div class="dm-source-card-title" title="{fname}">{fname}</div>
            <div class="dm-source-card-meta">Page {page}</div>
            {score_html}
            <div class="dm-source-card-excerpt">{excerpt}…</div>
        </div>
        """)
    return (
        '<div style="margin-top:0.75rem">'
        '<div style="font-size:0.72rem;color:var(--text-3);margin-bottom:4px;'
        'text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">'
        f'Referenced sources ({len(docs)} chunks)</div>'
        '<div class="dm-source-strip">' + "".join(cards) + "</div>"
        "</div>"
    )


def _latency_badge(latency: dict) -> str:
    total = latency.get("total_sec", 0)
    ret   = latency.get("retrieval_sec", 0)
    llm   = latency.get("llm_sec", 0)
    return (
        f'<span class="dm-badge dm-badge-blue">⚡ {total:.2f}s total</span>&nbsp;'
        f'<span class="dm-badge dm-badge-violet">🔍 {ret:.2f}s retrieval</span>&nbsp;'
        f'<span class="dm-badge dm-badge-amber">✨ {llm:.2f}s LLM</span>'
    )


def _index_status_html(collection: str) -> str:
    return (
        f'<div class="dm-status-bar">'
        f'<span class="dm-status-dot">{collection}</span>'
        f'<span style="color:var(--text-3);font-size:0.72rem;font-family:var(--font-mono);">'
        f'Hybrid · Top-{st.session_state.get("top_k", 4)}'
        f'</span></div>'
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(collections):
    with st.sidebar:
        # Brand
        st.markdown(
            """
            <div style="padding-bottom:1rem;border-bottom:1px solid var(--border);margin-bottom:1rem;">
                <div style="font-family:var(--font-display);font-weight:700;font-size:1.1rem;
                            background:linear-gradient(135deg,#fff,#94A3B8);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                    🧠 DocMind
                </div>
                <div style="font-size:0.68rem;color:var(--text-3);font-family:var(--font-mono);
                            letter-spacing:0.05em;margin-top:2px;">
                    AI Research Assistant
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Active collection badge
        current = st.session_state.get("current_collection", "default")
        st.markdown(
            f'<div class="collection-badge">{current}</div>',
            unsafe_allow_html=True,
        )

        # Collection selector
        st.markdown('<div class="sidebar-label">Collection</div>', unsafe_allow_html=True)
        updated_collections = get_collections() or ["default"]
        selected = st.selectbox(
            "Active Collection",
            updated_collections,
            index=updated_collections.index(current)
            if current in updated_collections else 0,
            key="sidebar_collection_select",
            label_visibility="collapsed",
        )
        st.session_state.current_collection = selected

        # New collection
        new_col = st.text_input(
            "New Collection",
            placeholder="e.g. research-2024",
            key="new_collection_input",
            label_visibility="collapsed",
        )
        if new_col and new_col.strip():
            if st.button("＋ Create Collection", key="create_collection_btn", use_container_width=True):
                safe = new_col.strip().replace(" ", "_")
                import os as _os
                _os.makedirs(f"vector_stores/{safe}", exist_ok=True)
                _os.makedirs(f"metadata/{safe}", exist_ok=True)
                st.session_state.current_collection = safe
                st.success(f"✅ '{new_col}' created!")
                st.rerun()

        st.markdown("---")

        # Upload zone
        st.markdown('<div class="sidebar-label">Upload PDFs</div>', unsafe_allow_html=True)
        pdf_docs = st.file_uploader(
            "Upload PDFs",
            accept_multiple_files=True,
            type="pdf",
            label_visibility="collapsed",
        )
        st.caption(f"Chunk size {CHUNKING.chunk_size} · Overlap {CHUNKING.chunk_overlap}")

        if st.button("🚀 Submit & Process", type="primary", use_container_width=True):
            if pdf_docs:
                with st.status("Indexing PDFs…", expanded=True) as s:
                    st.write("Extracting text & chunking…")
                    result = process_and_index_pdfs(pdf_docs, st.session_state.current_collection)
                    st.write("Building vector index & BM25…")
                    if result["ok"]:
                        s.update(label=f"✅ Indexed {len(pdf_docs)} PDF(s)", state="complete")
                        stats = result.get("stats", {}).get("overall", {})
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Pages", stats.get("total_pages", 0))
                        c2.metric("Extracted", stats.get("extracted_pages", 0))
                        c3.metric("Failed", stats.get("failed_pages", 0))
                    else:
                        s.update(label="❌ Failed", state="error")
                        st.error(result["message"])
            else:
                st.warning("Upload at least one PDF first.")

        st.markdown("---")

        # Advanced settings (collapsed by default)
        with st.expander("⚙️ Advanced Settings"):
            st.session_state.retrieval_mode = st.selectbox(
                "Retrieval Mode",
                ["hybrid", "similarity", "mmr"],
                index=["hybrid", "similarity", "mmr"].index(
                    st.session_state.get("retrieval_mode", "hybrid")
                ),
                help="Hybrid = dense + BM25 + RRF + cross-encoder reranking.",
            )
            st.session_state.top_k = st.slider(
                "Top-K Chunks", 2, 10,
                value=int(st.session_state.get("top_k", 4)),
                help="Final chunks sent to the LLM.",
            )
            if st.session_state.retrieval_mode == "hybrid":
                st.session_state.candidate_k = st.slider(
                    "Candidate-K", 6, 30,
                    value=int(st.session_state.get("candidate_k", 12)),
                    step=2,
                )
                st.session_state.rerank_top_n = st.slider(
                    "Rerank Top-N", 2, 10,
                    value=int(st.session_state.get("rerank_top_n", 4)),
                )
            st.session_state.prompt_version = st.selectbox(
                "Prompt Version",
                list(PROMPTS.keys()),
                index=list(PROMPTS.keys()).index(st.session_state.get("prompt_version", DEFAULT_PROMPT_VERSION))
                if st.session_state.get("prompt_version") in PROMPTS else 0,
            )

        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with col_b:
            prompt_type = st.selectbox(
                "Style",
                ["standard", "summarize", "detailed"],
                key="sidebar_prompt_type",
                label_visibility="visible",
            )
            st.session_state["prompt_type"] = prompt_type

        st.markdown(
            '<div style="margin-top:1rem;text-align:center;font-size:0.68rem;'
            'color:var(--text-3);font-family:var(--font-mono);">v2.0 · RAG Engine</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Chat Tab
# ---------------------------------------------------------------------------

def render_chat_tab(collections):
    selected_collection = st.session_state.get("current_collection", "default")

    # Index status bar
    st.markdown(_index_status_html(selected_collection), unsafe_allow_html=True)

    # ── Render existing conversation history ──────────────────────────────
    for entry in st.session_state.get("chat_history", []):
        with st.chat_message("user", avatar="👤"):
            st.markdown(entry["question"])

        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(entry["answer"])
            # Latency badges
            if entry.get("latency"):
                st.markdown(_latency_badge(entry["latency"]), unsafe_allow_html=True)
            # Source cards
            if entry.get("docs"):
                st.markdown(_source_cards_html(entry["docs"]), unsafe_allow_html=True)
            # Feedback row
            _render_feedback_row(entry["question"])

    # ── Chat input (sticky bottom) ────────────────────────────────────────
    user_question = st.chat_input(
        "Ask anything about your documents…",
        key="chat_input",
    )

    if user_question:
        # Display user message immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_question)

        # Run RAG pipeline
        with st.chat_message("assistant", avatar="🧠"):
            with st.status("Thinking…", expanded=False) as status:
                st.write("🔍 Retrieving relevant chunks…")
                prompt_type = st.session_state.get("prompt_type", "standard")
                temperature  = 0.3
                answer, docs, latency = run_rag_query(
                    user_question, prompt_type, temperature, selected_collection
                )
                st.write(f"✨ Generating response ({latency.get('llm_sec', 0):.2f}s)…")
                status.update(label="✅ Done", state="complete")

            if answer:
                st.markdown(answer)
                st.markdown(_latency_badge(latency), unsafe_allow_html=True)
                if docs:
                    st.markdown(_source_cards_html(docs), unsafe_allow_html=True)
                _render_feedback_row(user_question)

                # Persist to chat history
                st.session_state.chat_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": user_question,
                    "answer": answer,
                    "latency": latency,
                    "docs": docs or [],
                    "prompt_type": prompt_type,
                    "collection": selected_collection,
                })


def _render_feedback_row(question: str):
    """Compact inline thumbs feedback."""
    c1, c2, _, _ = st.columns([1, 1, 3, 1])
    with c1:
        if st.button("👍", key=f"up_{hash(question)}", help="Helpful"):
            st.session_state.setdefault("feedback_data", []).append(
                {"question": question, "helpful": True,
                 "timestamp": datetime.now().isoformat()}
            )
    with c2:
        if st.button("👎", key=f"dn_{hash(question)}", help="Not helpful"):
            st.session_state.setdefault("feedback_data", []).append(
                {"question": question, "helpful": False,
                 "timestamp": datetime.now().isoformat()}
            )
