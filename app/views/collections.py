"""
app/views/collections.py

Tab 4 — Collections, Upload & Cost Management.

This tab is the primary place for:
  - Uploading new PDF documents (not sidebar-dependent)
  - Creating / switching collections
  - Viewing indexed document stats
  - Cost tracking
  - Deleting collections

The sidebar also shows upload shortcuts for power users.
"""
import html
import streamlit as st
from app.helpers import (
    create_collection,
    get_collections,
    delete_collection,
    load_collection_metadata,
    estimate_cost,
    process_and_index_pdfs,
)
from src.config import CHUNKING
from app.state import (
    choose_collection,
    collection_selector,
    current_collection,
    forget_collection_pickers,
    sync_collection_from_widget,
)


def _create_collection_callback(name_key: str) -> None:
    """Create the typed collection and make it active (``on_click``)."""
    raw = st.session_state.get(name_key, "")
    if not raw or not raw.strip():
        return
    choose_collection(create_collection(raw))
    st.session_state[name_key] = ""


def _delete_collection_callback(name: str) -> None:
    """Delete a collection and keep the active selection pointing at a live one."""
    delete_collection(name)
    if current_collection() == name:
        remaining = [c for c in get_collections() if c != name]
        choose_collection(remaining[0] if remaining else "default")
    else:
        forget_collection_pickers()


def render_collections_tab(collections):
    # This tab is rendered on every run (st.tabs builds all children), so its
    # "Target Collection" selectbox must never write the active collection on
    # its own: a stale widget value would overwrite the sidebar's choice.
    st.markdown(
        """
        <div class="dm-page-header">
            <h2>Collections &amp; Documents</h2>
            <p>Upload PDFs · create collections · view index stats · track cost</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── UPLOAD SECTION (primary — no sidebar needed) ──────────────────────
    st.markdown('<div class="dm-section">📤 Upload &amp; Index PDFs</div>', unsafe_allow_html=True)

    up_col, settings_col = st.columns([5, 3])

    with up_col:
        pdf_docs = st.file_uploader(
            "Drop PDF files here",
            accept_multiple_files=True,
            type="pdf",
            key="main_uploader",
            label_visibility="collapsed",
        )

    with settings_col:
        # Collection target selector
        all_cols = get_collections() or ["default"]
        target_col = collection_selector(
            "Target Collection",
            all_cols,
            "upload_target_col",
            on_change=sync_collection_from_widget,
            args=("upload_target_col",),
        )
        st.caption(f"Chunk {CHUNKING.chunk_size} · Overlap {CHUNKING.chunk_overlap}")
        new_col_name = st.text_input(
            "Or create new collection",
            placeholder="e.g. research-q4",
            key="new_col_name_main",
        )
        if new_col_name.strip():
            st.button(
                "＋ Create",
                key="create_col_main",
                on_click=_create_collection_callback,
                args=("new_col_name_main",),
            )

    if st.button("🚀 Submit & Process", type="primary", use_container_width=True, key="main_process_btn"):
        if not pdf_docs:
            st.warning("Please upload at least one PDF file first.")
        else:
            with st.status(f"Indexing {len(pdf_docs)} PDF(s) into '{target_col}'…", expanded=True) as s:
                st.write("📄 Extracting text & chunking…")
                result = process_and_index_pdfs(pdf_docs, target_col)
                st.write("🔗 Building vector index & BM25…")
                if result["ok"]:
                    s.update(label=f"✅ Indexed {len(pdf_docs)} PDF(s) successfully", state="complete")
                    stats = result.get("stats", {}).get("overall", {})
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("Pages", stats.get("total_pages", 0))
                    rc2.metric("Extracted", stats.get("extracted_pages", 0))
                    rc3.metric("Failed", stats.get("failed_pages", 0))
                    st.rerun()
                else:
                    s.update(label="❌ Indexing failed", state="error")
                    st.error(result.get("message", "Unknown error"))

    st.markdown("---")

    # ── USAGE & COST ──────────────────────────────────────────────────────
    st.markdown('<div class="dm-section">💰 Usage &amp; Cost</div>', unsafe_allow_html=True)
    tokens = st.session_state.get("token_count", 0)
    cu1, cu2, cu3 = st.columns(3)
    cu1.metric("Total Tokens", f"{tokens:,}")
    cu2.metric("Est. Cost (USD)", f"${estimate_cost(tokens):.6f}")
    cu3.metric("Conversations", len(st.session_state.get("chat_history", [])))

    st.markdown("---")

    # ── COLLECTION BROWSER ────────────────────────────────────────────────
    st.markdown('<div class="dm-section">📦 Collection Browser</div>', unsafe_allow_html=True)

    refreshed = get_collections() or []
    if not refreshed:
        st.info("No collections yet. Upload PDFs above to get started.")
        return

    selected = st.selectbox("Inspect Collection", refreshed, key="stats_collection")
    metadata = load_collection_metadata(selected)

    if metadata:
        overall = metadata.get("extraction_stats", {}).get("overall", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Files",          len(metadata.get("files", [])))
        c2.metric("Pages",          metadata.get("total_pages", 0))
        c3.metric("Chunks",         metadata.get("total_chunks", 0))
        c4.metric("Extraction",     f"{overall.get('success_rate_percent', 100)}%")

        # Indexed file list
        files = metadata.get("files", [])
        if files:
            st.markdown('<div class="dm-section" style="margin-top:1rem;">📄 Indexed Files</div>', unsafe_allow_html=True)
            for fname in files:
                # Filenames come from the uploaded PDFs: escape before HTML.
                st.markdown(
                    f'<div style="background:var(--bg-surface);border:1px solid var(--border);'
                    f'border-radius:8px;padding:8px 14px;margin-bottom:5px;font-size:0.82rem;">'
                    f'📄 {html.escape(str(fname))}</div>',
                    unsafe_allow_html=True,
                )

        # Extraction quality
        if overall:
            st.markdown('<div class="dm-section" style="margin-top:1rem;">🔍 Extraction Quality</div>', unsafe_allow_html=True)
            eq1, eq2, eq3 = st.columns(3)
            eq1.metric("Extracted",  overall.get("extracted_pages", 0))
            eq2.metric("Empty",      overall.get("empty_pages", 0))
            eq3.metric("Failed",     overall.get("failed_pages", 0))

            with st.expander("Per-file breakdown"):
                for f in metadata.get("extraction_stats", {}).get("per_file", []):
                    fname = f.get("file_name", "?")
                    ext   = f.get("extracted_pages", 0)
                    tot   = f.get("total_pages", 0)
                    rate  = f.get("extraction_success_rate", 0)
                    st.markdown(
                        f'<div style="background:var(--bg-surface);border:1px solid var(--border);'
                        f'border-radius:8px;padding:8px 14px;margin-bottom:5px;font-size:0.78rem;">'
                        f'<strong>{html.escape(str(fname))}</strong>'
                        f'<span style="color:var(--text-3);margin-left:8px;">'
                        f'{ext}/{tot} pages · {rate}% success</span></div>',
                        unsafe_allow_html=True,
                    )

        st.caption(f"Last processed: {metadata.get('processed_at', 'N/A')}")

    else:
        st.info(f"No documents indexed in '{selected}' yet. Upload PDFs above.")

    # ── DANGER ZONE ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="dm-section">⚠️ Manage Collections</div>', unsafe_allow_html=True)
    with st.expander("Delete a Collection"):
        col_to_delete = st.selectbox(
            "Select collection to delete", refreshed, key="del_collection"
        )
        st.warning(f"This will permanently delete **{col_to_delete}** and all its indexed data.")
        st.button(
            f"🗑️ Delete '{col_to_delete}'",
            type="primary",
            key="delete_col_btn",
            on_click=_delete_collection_callback,
            args=(col_to_delete,),
        )


def render_stats_tab(collections):
    """Alias kept for backward compatibility."""
    render_collections_tab(collections)
