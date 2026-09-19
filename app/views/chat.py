"""
app/views/chat.py

Tab 1 — Interactive Chat (conversation thread + source cards).
Also owns the sidebar (PDF upload, collection management, advanced settings).

UI design: DocMind dark theme.
  - st.chat_input()  → sticky bottom input bar
  - st.chat_message() → user/AI message bubbles (styled via CSS in app/styles.py)
  - Citations:  [filename | page X] → Perplexity-style numbered pills with hover tooltips
  - Source cards rendered via per-card st.markdown() (avoids big-HTML Streamlit bug)
  - Latency badge shown on every AI response
  - Advanced settings collapsed in sidebar by default
"""
import os
import re
from datetime import datetime

import streamlit as st

from src.config import CHUNKING
from src.rag.inference import run_rag_query
from src.rag.prompt_registry import DEFAULT_PROMPT_VERSION, PROMPTS
from app.helpers import get_collections, process_and_index_pdfs


# ---------------------------------------------------------------------------
# Citation processing (Perplexity-style)
# ---------------------------------------------------------------------------

# Pattern: [any text | page N] or [filename | page N]
_CITE_RE = re.compile(r'\[([^\]]+\s*\|\s*page\s*[\d]+)\]', re.IGNORECASE)


def _process_citations(answer: str, docs: list) -> tuple:
    """
    Convert [filename | page N] inline citations to numbered superscript pills.

    Returns:
        processed_html : str  – the answer with citation pills injected (safe HTML)
        citation_list  : list – [(idx, fname, page, excerpt), ...]
    """
    citation_map: dict = {}   # raw_citation_text -> index
    citation_list: list = []  # ordered: (idx, fname, page, excerpt)

    def _make_pill(match):
        raw = match.group(1).strip()          # e.g. "guide.pdf | page 2"
        parts = [p.strip() for p in raw.split("|")]
        fname_raw = parts[0] if parts else raw
        page_raw  = parts[1] if len(parts) > 1 else ""
        page_num  = re.sub(r"[^0-9]", "", page_raw)

        if raw not in citation_map:
            idx = len(citation_map) + 1
            citation_map[raw] = idx
            # Match against retrieved docs to grab excerpt
            excerpt = ""
            for doc in docs:
                src = os.path.basename(doc.metadata.get("source", ""))
                pg  = str(doc.metadata.get("page_number", ""))
                # loose match: filename overlaps AND page matches (if available)
                fname_match = (
                    src.lower() in fname_raw.lower() or
                    fname_raw.lower() in src.lower() or
                    src.lower().replace("_", " ") in fname_raw.lower()
                )
                page_match = (not page_num) or (pg == page_num)
                if fname_match and page_match:
                    excerpt = doc.page_content[:120].replace('"', '&quot;').replace('\n', ' ').strip()
                    break
            citation_list.append((idx, fname_raw, page_raw, excerpt))
        else:
            idx = citation_map[raw]

        # Build tooltip text (shown on hover)
        short_fname = os.path.basename(fname_raw)
        tooltip = f"📄 {short_fname}"
        if page_raw:
            tooltip += f"  ·  {page_raw}"
        # Add excerpt if found
        for _idx, _f, _p, _exc in citation_list:
            if _idx == idx and _exc:
                tooltip += f"\n\n{_exc[:100]}…"
                break
        tooltip = tooltip.replace('"', '&quot;').replace("'", '&#39;')

        return (
            f'<sup><span class="dm-cite-pill" data-tooltip="{tooltip}">'
            f'[{idx}]</span></sup>'
        )

    # Replace all citation patterns
    # We need to render the answer as HTML (so markdown links/bold still work)
    # First escape raw HTML chars that came from the LLM, then replace citations
    safe_answer = answer.replace("<", "&lt;").replace(">", "&gt;")
    processed   = _CITE_RE.sub(_make_pill, safe_answer)

    # Convert basic markdown to HTML (bold, italic, code, newlines)
    # Simple transforms — Streamlit's st.markdown can't be used directly with
    # unsafe_allow_html when we've already built our own HTML.
    processed = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', processed)
    processed = re.sub(r'\*(.+?)\*',     r'<em>\1</em>',         processed)
    processed = re.sub(r'`(.+?)`',       r'<code style="font-family:\'JetBrains Mono\',monospace;'
                                          r'font-size:0.82em;background:rgba(255,255,255,0.07);'
                                          r'padding:1px 5px;border-radius:4px;">\1</code>', processed)
    # Convert newlines to paragraph breaks
    paras = [p for p in processed.split('\n\n') if p.strip()]
    processed_html = ''.join(
        f'<p style="margin:0 0 0.65em;line-height:1.7;color:#E2E8F0;font-family:Inter,sans-serif;'
        f'font-size:0.9rem;">{p.replace(chr(10), "<br>")}</p>'
        for p in paras
    )

    return processed_html, citation_list


def _refs_html(citation_list: list) -> str:
    """Build the numbered reference list shown at the bottom of an AI message."""
    if not citation_list:
        return ""
    items = ""
    for idx, fname, page, excerpt in citation_list:
        short = os.path.basename(fname)
        items += (
            f'<div class="dm-ref-item">'
            f'  <span class="dm-ref-num">{idx}</span>'
            f'  <span class="dm-ref-text">'
            f'    <span class="dm-ref-file">{short}</span>'
            f'    <span class="dm-ref-page">  ·  {page}</span>'
            f'  </span>'
            f'</div>'
        )
    return (
        '<div class="dm-refs">'
        '  <div class="dm-refs-title">Sources</div>'
        f'  {items}'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Source card renderer (reliable per-card calls, no big HTML blob)
# ---------------------------------------------------------------------------

def _render_source_cards(docs: list):
    """
    Render source reference cards below an AI message.

    Replaces the old _source_cards_html() which produced one large HTML string
    that Streamlit would partially render and then fall back to displaying as
    plain text. This version renders each card as its own small st.markdown()
    call inside st.columns(), which is always reliable.
    """
    if not docs:
        return

    st.markdown(
        '<div style="font-size:0.67rem;color:#475569;text-transform:uppercase;'
        'letter-spacing:0.07em;font-weight:700;margin-bottom:6px;margin-top:10px;">'
        f'📚 Referenced Sources &nbsp;·&nbsp; {len(docs)} chunks</div>',
        unsafe_allow_html=True,
    )

    n = min(len(docs), 4)                # max 4 cards per row
    cols = st.columns(n) if n > 0 else []

    for i, doc in enumerate(docs[:8]):   # max 8 total
        fname   = os.path.basename(doc.metadata.get("source", "Unknown"))
        page    = doc.metadata.get("page_number", "?")
        score   = doc.metadata.get("rerank_score") or doc.metadata.get("score")
        excerpt = (
            doc.page_content[:140]
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("\n", " ")
            .strip()
        )

        score_row = ""
        if score is not None:
            score_row = (
                f'<div class="dm-sc2-score">▲ {score:.1%} relevance</div>'
            )

        card_html = (
            f'<div class="dm-source-card-v2">'
            f'  <div class="dm-sc2-fname" title="{fname}">{fname}</div>'
            f'  <div class="dm-sc2-meta">Page {page}</div>'
            f'  {score_row}'
            f'  <div class="dm-sc2-excerpt">{excerpt}…</div>'
            f'</div>'
        )

        with cols[i % n]:
            st.markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Latency badge
# ---------------------------------------------------------------------------

def _latency_badge(latency: dict) -> str:
    total = latency.get("total_sec", 0)
    ret   = latency.get("retrieval_sec", 0)
    llm   = latency.get("llm_sec", 0)
    return (
        f'<span class="dm-badge dm-badge-blue">⚡ {total:.2f}s total</span>&nbsp;'
        f'<span class="dm-badge dm-badge-violet">🔍 {ret:.2f}s retrieval</span>&nbsp;'
        f'<span class="dm-badge dm-badge-amber">✨ {llm:.2f}s LLM</span>'
    )


# ---------------------------------------------------------------------------
# Index status bar
# ---------------------------------------------------------------------------

def _index_status_html(collection: str) -> str:
    return (
        f'<div class="dm-status-bar">'
        f'<span class="dm-status-dot-inner"></span>'
        f'{collection}&nbsp;&nbsp;'
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
            f'<div class="collection-badge"><div class="dot"></div>{current}</div>',
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
                        s.update(label="✅ Indexed successfully", state="complete")
                        st.session_state.current_collection = st.session_state.current_collection
                        st.rerun()
                    else:
                        s.update(label="❌ Failed", state="error")
                        st.error(result.get("message", "Unknown error"))
            else:
                st.warning("Upload PDFs first.")

        st.markdown("---")

        # Advanced settings (collapsed)
        with st.expander("⚙️ Advanced Settings", expanded=False):
            st.markdown('<div class="sidebar-label">Retrieval Mode</div>', unsafe_allow_html=True)
            retrieval_mode = st.selectbox(
                "Retrieval Mode",
                ["hybrid", "similarity", "mmr"],
                index=["hybrid", "similarity", "mmr"].index(
                    st.session_state.get("retrieval_mode", "hybrid")
                ),
                key="sidebar_retrieval_mode",
                label_visibility="collapsed",
            )
            st.session_state.retrieval_mode = retrieval_mode

            st.markdown('<div class="sidebar-label">Top-K Results</div>', unsafe_allow_html=True)
            top_k = st.slider("Top-K", 1, 10, st.session_state.get("top_k", 4), key="sidebar_top_k", label_visibility="collapsed")
            st.session_state.top_k = top_k

            st.markdown('<div class="sidebar-label">Temperature</div>', unsafe_allow_html=True)
            temperature = st.slider(
                "Temperature", 0.0, 1.0, st.session_state.get("temperature", 0.3),
                0.05, key="sidebar_temperature", label_visibility="collapsed"
            )
            st.session_state.temperature = temperature

            st.markdown('<div class="sidebar-label">Prompt Style</div>', unsafe_allow_html=True)
            prompt_keys = list(PROMPTS.keys())
            # Sanitise stored value — may be "standard" (legacy) which no longer exists
            stored_prompt = st.session_state.get("prompt_type", DEFAULT_PROMPT_VERSION)
            if stored_prompt not in prompt_keys:
                stored_prompt = DEFAULT_PROMPT_VERSION
                st.session_state["prompt_type"] = DEFAULT_PROMPT_VERSION
            prompt_type = st.selectbox(
                "Prompt Style",
                prompt_keys,
                index=prompt_keys.index(stored_prompt),
                key="sidebar_prompt_type",
                label_visibility="visible",
            )
            st.session_state["prompt_type"] = prompt_type

        # Clear chat
        st.markdown("---")
        col_a, col_b = st.columns([3, 2])
        with col_a:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with col_b:
            st.markdown(
                '<div style="margin-top:0.5rem;text-align:center;font-size:0.68rem;'
                'color:var(--text-3);font-family:var(--font-mono);">v2.0 · RAG Engine</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Chat Tab
# ---------------------------------------------------------------------------

def _render_ai_message(answer: str, docs: list, latency: dict, question: str):
    """Render one complete AI response: processed answer + citations + source cards + feedback."""
    # Process citations → HTML with pills
    processed_html, citation_list = _process_citations(answer, docs or [])

    # Reference list HTML
    refs_html = _refs_html(citation_list)

    # Render the answer + inline citations as HTML
    full_html = (
        f'<div style="font-family:Inter,sans-serif;line-height:1.7;font-size:0.9rem;">'
        f'{processed_html}'
        f'{refs_html}'
        f'</div>'
    )
    st.markdown(full_html, unsafe_allow_html=True)

    # Latency badges
    st.markdown(_latency_badge(latency), unsafe_allow_html=True)

    # Source cards (reliable native rendering)
    if docs:
        _render_source_cards(docs)

    # Feedback row
    _render_feedback_row(question)


def render_chat_tab(collections):
    selected_collection = st.session_state.get("current_collection", "default")

    # Index status bar
    st.markdown(_index_status_html(selected_collection), unsafe_allow_html=True)

    # ── Render existing conversation history ──────────────────────────────
    for entry in st.session_state.get("chat_history", []):
        with st.chat_message("user", avatar="👤"):
            st.markdown(entry["question"])

        with st.chat_message("assistant", avatar="🧠"):
            _render_ai_message(
                entry["answer"],
                entry.get("docs", []),
                entry.get("latency", {}),
                entry["question"],
            )

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
                prompt_type = st.session_state.get("prompt_type", DEFAULT_PROMPT_VERSION)
                temperature  = st.session_state.get("temperature", 0.3)
                answer, docs, latency = run_rag_query(
                    user_question, prompt_type, temperature, selected_collection
                )
                st.write(f"✨ Generating response ({latency.get('llm_sec', 0):.2f}s)…")
                status.update(label="✅ Done", state="complete")

            if answer:
                _render_ai_message(answer, docs or [], latency, user_question)

                # Persist to chat history
                st.session_state.chat_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question":  user_question,
                    "answer":    answer,
                    "latency":   latency,
                    "docs":      docs or [],
                    "prompt_type": prompt_type,
                    "collection":  selected_collection,
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
