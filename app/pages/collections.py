"""
app/pages/collections.py

Tab 2 — Document Stats
Tab 6 — Collections & Cost Management
"""
import streamlit as st
from app.helpers import (
    get_collections,
    delete_collection,
    load_collection_metadata,
    estimate_cost,
)


def render_stats_tab(collections):
    st.subheader("📊 Document Analytics")

    selected = st.selectbox("Select Collection to View Stats", collections, key="stats_collection")
    metadata = load_collection_metadata(selected)

    if metadata:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("📄 Total Files", len(metadata.get("files", [])))
        with c2: st.metric("📃 Total Pages", metadata.get("total_pages", 0))
        with c3: st.metric("🔗 Total Chunks", metadata.get("total_chunks", 0))
        with c4: st.metric("📏 Chunk Size", f"{metadata.get('chunk_size', 0)} / {metadata.get('chunk_overlap', 0)}")

        st.write("---")
        st.subheader("Uploaded Files")
        for file_info in metadata.get("files", []):
            st.write(f"• **{file_info}**")

        st.write("---")
        st.write(f"*Last processed: {metadata.get('processed_at', 'N/A')}*")

        overall = metadata.get("extraction_stats", {}).get("overall", {})
        if overall:
            st.write("---")
            st.subheader("📊 Extraction Quality")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("✅ Extracted Pages", overall.get("extracted_pages", 0))
            with c2: st.metric("📄 Total Pages", overall.get("total_pages", 0))
            with c3: st.metric("⚠️ Empty Pages", overall.get("empty_pages", 0))
            with c4: st.metric("💥 Failed Pages", overall.get("failed_pages", 0))
            st.write(f"**Success Rate:** {overall.get('success_rate_percent', 0)}%")

            with st.expander("Per-file extraction breakdown"):
                for f in metadata.get("extraction_stats", {}).get("per_file", []):
                    st.write(
                        f"• **{f.get('file_name')}** → "
                        f"{f.get('extracted_pages')}/{f.get('total_pages')} extracted "
                        f"({f.get('extraction_success_rate')}%) | "
                        f"Empty: {f.get('empty_pages')} | Failed: {f.get('failed_pages')}"
                    )
    else:
        st.info(f"📌 No documents in '{selected}' collection yet.")


def render_collections_tab(collections):
    st.subheader("⚙️ Collections & Cost Management")

    st.write("### 📦 Vector Store Collections")
    with st.expander("View All Collections"):
        for collection in collections:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"📁 **{collection}**")
            with col_b:
                if st.button("🗑️ Delete", key=f"delete_{collection}"):
                    delete_collection(collection)
                    st.success(f"Deleted '{collection}'")
                    st.rerun()

    st.write("### 💰 Cost & Token Tracking")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("🔤 Total Tokens Used", st.session_state.token_count)
    with c2: st.metric("💵 Estimated Cost", f"${estimate_cost(st.session_state.token_count):.6f}")
    with c3: st.metric("📝 Total Conversations", len(st.session_state.chat_history))
