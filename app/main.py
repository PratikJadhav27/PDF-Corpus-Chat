"""
app/main.py  —  thin entrypoint

Responsibility:
  - Configure the Streamlit page (DocMind branding)
  - Inject the CSS design system from app/styles.py
  - Initialise session state
  - Assemble the 4-tab layout
  - Delegate all rendering to app/pages/*

Run with:
    streamlit run app/main.py
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path when running via `streamlit run app/main.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

from app.state import init_session_state
from app.helpers import get_collections
from app.styles import get_css
from app.pages.chat import render_chat_tab, render_sidebar
from app.pages.collections import render_collections_tab
from app.pages.history import render_history_tab
from app.pages.evaluation import render_evaluation_tab

load_dotenv()


def _render_header():
    st.markdown(
        """
        <div class="docmind-header">
            <div class="docmind-header-icon">🧠</div>
            <div class="docmind-header-text">
                <h1>DocMind</h1>
                <span>Multi-PDF RAG Research Assistant · v2.0</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="DocMind — Multi-PDF RAG",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject full design system CSS
    st.markdown(get_css(), unsafe_allow_html=True)

    init_session_state()

    collections = get_collections() or ["default"]

    # Sidebar (PDF upload, collection management, settings)
    render_sidebar(collections)

    # Page header
    _render_header()

    # 4-tab navigation (pill style via CSS)
    tab_chat, tab_analytics, tab_history, tab_collections = st.tabs([
        "💬 Chat",
        "📊 Analytics",
        "📜 History",
        "⚙️ Collections",
    ])

    with tab_chat:
        render_chat_tab(collections)

    with tab_analytics:
        render_evaluation_tab(collections)

    with tab_history:
        render_history_tab()

    with tab_collections:
        render_collections_tab(collections)


if __name__ == "__main__":
    main()
