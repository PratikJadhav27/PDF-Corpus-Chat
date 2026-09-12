"""
app/pages/collections.py

Tab 4 — Collections & Cost Management.
Merged from old Tabs 2 (Doc Stats) + 6 (Collections & Cost).
"""
import streamlit as st
from app.helpers import (
    get_collections,
    delete_collection,
    load_collection_metadata,
    estimate_cost,
)


def render_collections_tab(collections):
    st.markdown(
        """
        <div style="margin-bottom:1.25rem;">
            <div style="font-family:var(--font-display);font-size:1.1rem;font-weight:600;
                        color:var(--text-1);">Collections & Documents</div>
            <div style="font-size:0.78rem;color:var(--text-3);margin-top:2px;">
                Collection stats · cost tracking · index management
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Cost & usage overview ─────────────────────────────────────────────
    tokens = st.session_state.get("token_count", 0)
    cost   = estimate_cost(tokens)

    st.markdown('<div class="dm-section-header">💰 Usage & Cost</div>', unsafe_allow_html=True)
    cu1, cu2, cu3 = st.columns(3)
    cu1.metric("Total Tokens", f"{tokens:,}")
    cu2.metric("Est. Cost (USD)", f"${cost:.6f}")
    cu3.metric("Conversations", len(st.session_state.get("chat_history", [])))

    st.markdown("---")

    # ── Collection browser ────────────────────────────────────────────────
    st.markdown('<div class="dm-section-header">📦 Vector Store Collections</div>', unsafe_allow_html=True)

    if not collections:
        st.markdown(
            """
            <div class="dm-metric-card" style="text-align:center;padding:2.5rem;">
                <div style="font-size:2rem;opacity:0.3;">📦</div>
                <div style="color:var(--text-3);font-size:0.82rem;margin-top:0.5rem;">
                    No collections yet. Upload PDFs in the sidebar to get started.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    selected = st.selectbox("Inspect Collection", collections, key="stats_collection")
    metadata = load_collection_metadata(selected)

    if metadata:
        # Top metrics
        overall = metadata.get("extraction_stats", {}).get("overall", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Files", len(metadata.get("files", [])))
        c2.metric("Pages",  metadata.get("total_pages", 0))
        c3.metric("Chunks", metadata.get("total_chunks", 0))
        c4.metric(
            "Extraction Rate",
            f"{overall.get('success_rate_percent', 100)}%",
        )

        # File list
        st.markdown('<div class="dm-section-header" style="margin-top:1rem;">📄 Files</div>', unsafe_allow_html=True)
        for file_info in metadata.get("files", []):
            st.markdown(
                f'<div class="dm-metric-card" style="padding:0.65rem 1rem;margin-bottom:6px;">'
                f'<span style="font-size:0.82rem;">📄 {file_info}</span></div>',
                unsafe_allow_html=True,
            )

        # Extraction quality
        if overall:
            st.markdown('<div class="dm-section-header" style="margin-top:1rem;">🔍 Extraction Quality</div>', unsafe_allow_html=True)
            eq1, eq2, eq3 = st.columns(3)
            eq1.metric("Extracted", overall.get("extracted_pages", 0))
            eq2.metric("Empty",     overall.get("empty_pages", 0))
            eq3.metric("Failed",    overall.get("failed_pages", 0))

            with st.expander("Per-file breakdown"):
                for f in metadata.get("extraction_stats", {}).get("per_file", []):
                    st.markdown(
                        f'<div class="dm-metric-card" style="padding:0.65rem 1rem;margin-bottom:6px;">'
                        f'<span style="font-size:0.78rem;font-weight:600;">{f.get("file_name")}</span>'
                        f'<span style="font-size:0.72rem;color:var(--text-3);margin-left:8px;">'
                        f'{f.get("extracted_pages")}/{f.get("total_pages")} pages · '
                        f'{f.get("extraction_success_rate")}% success</span></div>',
                        unsafe_allow_html=True,
                    )

        st.caption(f"Last processed: {metadata.get('processed_at', 'N/A')}")

    else:
        st.markdown(
            f'<div class="dm-metric-card" style="text-align:center;padding:2rem;">'
            f'<div style="color:var(--text-3);">No documents in <strong>{selected}</strong> yet.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Danger zone ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="dm-section-header">⚠️ Manage Collections</div>', unsafe_allow_html=True)
    with st.expander("Delete a Collection"):
        col_to_delete = st.selectbox("Select collection to delete", collections, key="del_collection")
        st.warning(f"This will permanently delete **{col_to_delete}** and all its indexed data.")
        if st.button(f"🗑️ Delete '{col_to_delete}'", type="primary"):
            delete_collection(col_to_delete)
            st.success(f"Deleted '{col_to_delete}'")
            st.rerun()


# Keep the old name for backward compat with any external callers
def render_stats_tab(collections):
    """Alias — merged into render_collections_tab."""
    render_collections_tab(collections)
