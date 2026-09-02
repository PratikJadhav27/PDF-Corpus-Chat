"""
app/main.py  —  thin entrypoint (~60 lines)

Responsibility:
  - Configure the Streamlit page
  - Initialise session state
  - Assemble the tab layout
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
from app.pages.chat import render_chat_tab, render_sidebar
from app.pages.collections import render_stats_tab, render_collections_tab
from app.pages.history import render_history_tab, render_search_tab, render_batch_tab
from app.pages.evaluation import render_evaluation_tab

load_dotenv()


def main() -> None:
    st.set_page_config(
        page_title="Multi-PDF RAG Chat",
        page_icon=":scroll:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab-list"] button { font-weight: bold; }
        .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()

    st.header("Multi-PDF's 📚 - Chat Agent 🤖")

    collections = get_collections() or ["default"]

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "💬 Chat",
        "📊 Document Stats",
        "📜 Chat History",
        "🔍 Advanced Search",
        "🌐 Batch Q&A",
        "⚙️ Collections & Cost",
        "🧪 Evaluation",
    ])

    with tab1:
        render_chat_tab(collections)
    with tab2:
        render_stats_tab(collections)
    with tab3:
        render_history_tab()
    with tab4:
        render_search_tab(collections)
    with tab5:
        render_batch_tab()
    with tab6:
        render_collections_tab(collections)
    with tab7:
        render_evaluation_tab(collections)

    render_sidebar()

    st.markdown(
        """
        <div style="position:fixed;bottom:0;left:0;width:100%;background-color:#0E1117;padding:15px;text-align:center;">
            © <a href="https://github.com/PratikJadhav27" target="_blank">Pratik Jadhav</a>
            | Made with ❤️ | Multi-PDF RAG Chat App
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
