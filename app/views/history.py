"""
app/pages/history.py

Tab 3 — Conversation History & Advanced Search.
Merged from old Tabs 3 + 4 + 5 into one clean page.

Semantic Search is independent of the chat log (ISSUE-008): it lists the
collections that exist on disk, so a session that has not chatted yet — or has
only chatted in one collection — can still search every indexed collection.
"""
import json
import os
from datetime import datetime

import streamlit as st
from app.helpers import get_collections
from app.history_store import feedback_summary, load_feedback
from src.rag.embeddings import get_embeddings
from src.rag.vector_store import load_vector_store
from src.rag.prompt_registry import DEFAULT_PROMPT_VERSION


def render_history_tab():
    st.markdown(
        """
        <div style="margin-bottom:1.25rem;">
            <div style="font-family:var(--font-display);font-size:1.1rem;font-weight:600;
                        color:var(--text-1);">Conversation History</div>
            <div style="font-size:0.78rem;color:var(--text-3);margin-top:2px;">
                Saved chat log (survives reloads) · semantic search · batch Q&amp;A
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    inner_tab_hist, inner_tab_search = st.tabs(["📜 History", "🔍 Semantic Search"])

    with inner_tab_hist:
        _render_history_panel()

    with inner_tab_search:
        _render_semantic_search_panel()


def _render_history_panel():
    """Saved chat log with collection/keyword filters, feedback and exports."""
    history = st.session_state.get("chat_history", [])

    if not history:
        st.markdown(
            """
            <div class="dm-metric-card" style="text-align:center;padding:3rem;">
                <div style="font-size:2.5rem;opacity:0.3;">💬</div>
                <div style="color:var(--text-3);font-size:0.85rem;margin-top:0.75rem;">
                    No conversation history yet.<br>Ask something in the Chat tab!
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Filter
    collections_in_history = sorted(set(
        c.get("collection", "default") for c in history
    ))
    col_f, col_s = st.columns([2, 1])
    with col_f:
        filter_col = st.selectbox("Filter by collection", ["All"] + collections_in_history, key="hist_filter")
    with col_s:
        search_term = st.text_input("Search questions", placeholder="keyword…", key="hist_search")

    filtered = history
    if filter_col != "All":
        filtered = [c for c in filtered if c.get("collection") == filter_col]
    if search_term:
        filtered = [c for c in filtered if search_term.lower() in c["question"].lower()]

    # Reverse (most recent first)
    filtered = list(reversed(filtered))
    total = len(filtered)
    st.caption(f"{total} conversation{'s' if total != 1 else ''} found")

    # Feedback is persisted per turn (ISSUE-009); show the tally so the state of
    # the log is visible without opening every entry.
    saved_feedback = load_feedback()
    counts = feedback_summary(saved_feedback)
    rated = counts["helpful"] + counts["not_helpful"]
    if rated:
        st.caption(
            f"Feedback · 👍 {counts['helpful']} helpful · "
            f"👎 {counts['not_helpful']} not helpful · {rated} rated"
        )
    else:
        st.caption(
            "No feedback yet — rate an answer with 👍/👎 in the Chat tab."
        )

    for i, chat in enumerate(filtered, 1):
        lat = chat.get("latency", {})
        lat_str = f"⚡ {lat.get('total_sec', 0):.2f}s" if lat else ""
        col_label = f"[{chat.get('collection','?')}]" if chat.get("collection") else ""
        fb = saved_feedback.get(chat.get("id")) if chat.get("id") else None
        thumb = "👍 " if fb and fb.get("helpful") else "👎 " if fb else ""
        fb_badge = (
            '<span class="dm-badge dm-badge-green">👍 Helpful</span>'
            if fb and fb.get("helpful") else
            '<span class="dm-badge dm-badge-amber">👎 Not helpful</span>'
            if fb else ""
        )

        with st.expander(f"{thumb}Q{total-i+1} · {chat['question'][:70]}… {col_label} {lat_str}"):
            # Meta row
            st.markdown(
                f"""
                <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:0.75rem;">
                    <span class="dm-badge dm-badge-blue">{chat.get('timestamp','')}</span>
                    <span class="dm-badge dm-badge-violet">{chat.get('collection','default')}</span>
                    <span class="dm-badge dm-badge-amber">{chat.get('prompt_version') or chat.get('prompt_type') or '—'}</span>
                    {f'<span class="dm-badge dm-badge-green">{lat_str}</span>' if lat_str else ''}
                    {fb_badge}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f"**Question:** {chat['question']}")
            st.markdown(f"**Answer:** {chat['answer']}")

            # Sources
            docs = chat.get("docs", [])
            if docs:
                st.caption(f"{len(docs)} source chunk(s) retrieved")
                for doc in docs:
                    src  = os.path.basename(doc.metadata.get("source", "?"))
                    page = doc.metadata.get("page_number", "?")
                    st.caption(f"  · {src} | page {page}")

    # Download buttons
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        exportable = [
            {
                **{k: v for k, v in c.items() if k != "docs"},
                "feedback": (
                    "helpful" if saved_feedback.get(c.get("id"), {}).get("helpful")
                    else "not_helpful" if saved_feedback.get(c.get("id"))
                    else None
                ),
            }
            for c in filtered
        ]
        st.download_button(
            "⬇️ Export JSON",
            data=json.dumps(exportable, indent=2, default=str),
            file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with c2:
        txt = "\n\n".join(
            f"Q: {c['question']}\n\nA: {c['answer']}\n\n{'─'*60}"
            for c in filtered
        )
        st.download_button(
            "⬇️ Export TXT",
            data=txt,
            file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )


def _render_semantic_search_panel():
    """Vector search over one indexed collection, chosen from disk, not history."""
    st.markdown('<div class="dm-section">🔍 Semantic Document Search</div>', unsafe_allow_html=True)

    collections = sorted(get_collections())

    if not collections:
        st.info("No collections yet — upload PDFs in the Collections tab to build one.")
        return

    current = st.session_state.get("current_collection", "default")

    col_c, col_k = st.columns([3, 1])
    with col_c:
        s_collection = st.selectbox(
            "Collection",
            collections,
            index=collections.index(current) if current in collections else 0,
            key="search_collection",
        )
    with col_k:
        k_results = st.slider("Top-K", 1, 10, 5, key="search_k")

    s_query = st.text_input("Search query", placeholder="Search across your documents…", key="search_query")

    if st.button("🔍 Search", type="primary"):
        if not s_query:
            st.warning("Enter a query first.")
        else:
            embeddings = get_embeddings()
            db = load_vector_store(s_collection, embeddings)
            if db is None:
                st.error(f"Collection '{s_collection}' has no indexed documents.")
            else:
                results = db.similarity_search(s_query, k=k_results)
                st.success(f"Found {len(results)} chunk(s)")
                for i, r in enumerate(results, 1):
                    src  = os.path.basename(r.metadata.get("source", "Unknown"))
                    page = r.metadata.get("page_number", "?")
                    st.markdown(
                        f"""
                        <div class="dm-metric-card" style="margin-bottom:0.75rem;">
                            <div style="display:flex;justify-content:space-between;
                                        align-items:center;margin-bottom:0.5rem;">
                                <span style="font-weight:600;font-size:0.85rem;">
                                    Result {i} · {src}
                                </span>
                                <span class="dm-badge dm-badge-blue">Page {page}</span>
                            </div>
                            <div style="font-size:0.78rem;color:var(--text-2);
                                        line-height:1.6;">{r.page_content[:500]}…</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
