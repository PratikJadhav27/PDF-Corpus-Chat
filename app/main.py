import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import streamlit as st
import os
import json
from datetime import datetime
import shutil

from dotenv import load_dotenv
import google.generativeai as genai

from PyPDF2 import PdfReader  # kept for a couple old utility funcs (we'll remove later)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import CHUNKING, RETRIEVAL
from src.ingestion.pdf_loader import load_pdf_pages
from src.ingestion.chunking import chunk_pdf_pages
from src.rag.vector_store import (
    build_vector_store_from_chunks,
    save_vector_store,
    load_vector_store,
    get_embeddings,
)

load_dotenv()

# Configure Gemini API (used for token counting)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# -----------------------------
# Session State
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_metadata" not in st.session_state:
    st.session_state.document_metadata = None
if "current_collection" not in st.session_state:
    st.session_state.current_collection = "default"
if "token_count" not in st.session_state:
    st.session_state.token_count = 0
if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = []


# -----------------------------
# Utility / Collection helpers
# -----------------------------
def get_collections():
    """Get list of all vector store collections"""
    collections_dir = "vector_stores"
    if not os.path.exists(collections_dir):
        return []
    return [
        d
        for d in os.listdir(collections_dir)
        if os.path.isdir(os.path.join(collections_dir, d))
    ]


def delete_collection(collection_name):
    """Delete a vector store collection"""
    collection_dir = f"vector_stores/{collection_name}"
    metadata_dir = f"metadata/{collection_name}"

    if os.path.exists(collection_dir):
        shutil.rmtree(collection_dir)
    if os.path.exists(metadata_dir):
        shutil.rmtree(metadata_dir)

    return True


def estimate_cost(tokens_used):
    """Estimate API cost based on tokens used (rough estimate)"""
    cost_per_million = 0.15
    estimated_cost = (tokens_used / 1_000_000) * cost_per_million
    return estimated_cost


def load_collection_metadata(collection_name):
    """Load metadata for a specific collection"""
    metadata_file = f"metadata/{collection_name}/metadata.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            return json.load(f)
    return None


# -----------------------------
# Old feature helpers (kept)
# -----------------------------
def keyword_search_in_pdf(pdf_docs, keyword):
    """Search for keywords in uploaded PDFs"""
    results = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text() or ""
            if keyword.lower() in text.lower():
                start = max(0, text.lower().find(keyword.lower()) - 100)
                end = min(len(text), text.lower().find(keyword.lower()) + 200)
                results.append(
                    {
                        "file": pdf.name,
                        "page": page_num,
                        "excerpt": text[start:end],
                    }
                )
    return results


def translate_text(text, target_language):
    """Translate text using Gemini"""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(
        template="Translate the following text to {target_lang}. Only provide the translation.\n\nText:\n{text}",
        input_variables=["text", "target_lang"],
    )
    chain = prompt | model | StrOutputParser()

    try:
        return chain.invoke({"text": text, "target_lang": target_language})
    except Exception:
        return text


def generate_qa_pairs(text, num_pairs=5):
    """Generate Q&A pairs from document text"""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
    prompt_template = f"""Based on the following text, generate exactly {num_pairs} important question-answer pairs.
Format each pair as:
Q: [question]
A: [answer]

Text:
{{text}}

Q&A Pairs:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = prompt | model | StrOutputParser()

    try:
        return chain.invoke({"text": text[:2000]})
    except Exception:
        return "Could not generate Q&A pairs"


# -----------------------------
# RAG Core
# -----------------------------
def get_conversational_chain(prompt_type="standard", temperature=0.3):
    prompts = {
        "standard": """
Answer the question as detailed as possible from the provided context.
If the answer is not in the provided context, say: "answer is not available in the context".
Do not make up information.

Context:
{context}

Question:
{question}

Answer:""",
        "summarize": """
Based on the provided context, create a concise summary that answers the question.
If the answer is not in the provided context, say: "answer is not available in the context".

Context:
{context}

Question:
{question}

Summary:""",
        "detailed": """
Provide an extremely detailed and comprehensive answer based on the context.
If the answer is not in the provided context, say: "answer is not available in the context".

Context:
{context}

Question:
{question}

Detailed Answer:""",
    }

    prompt_template = prompts.get(prompt_type, prompts["standard"])

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = prompt | model | StrOutputParser()
    return chain


def user_input(user_question, prompt_type, temperature, collection_name="default"):
    embeddings = get_embeddings()

    db = load_vector_store(collection_name, embeddings)
    if db is None:
        st.error(
            f"Please upload and process PDF files first in collection '{collection_name}'!"
        )
        return None, None

    docs = db.similarity_search(user_question, k=RETRIEVAL.top_k)

    chain = get_conversational_chain(prompt_type=prompt_type, temperature=temperature)

    # context includes chunk text only (Day 3 we will add strict citations)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    response = chain.invoke({"context": context, "question": user_question})

    # Token count (best effort)
    token_count = 0
    try:
        token_count = genai.count_text_tokens(response).total_tokens
        st.session_state.token_count += token_count
    except Exception:
        pass

    # Add to chat history with better source info
    st.session_state.chat_history.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": user_question,
            "answer": response,
            "sources": [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page_number", "?"),
                    "chunk_index": doc.metadata.get("chunk_index", "?"),
                }
                for doc in docs
            ],
            "prompt_type": prompt_type,
            "tokens_used": token_count,
            "collection": collection_name,
        }
    )

    return response, docs


# -----------------------------
# Processing PDFs (Day 2 fixed pipeline)
# -----------------------------
def process_and_index_pdfs(pdf_docs, collection_name: str):
    """
    Day 2 pipeline:
    - load page-wise text
    - chunk per page with metadata
    - build FAISS with metadatas
    - persist per collection
    - write collection metadata.json
    """
    all_pages = []
    files_info = []
    total_pages = 0

    for pdf in pdf_docs:
        pages = load_pdf_pages(pdf)
        all_pages.extend(pages)

        files_info.append({"name": pdf.name, "pages": len(pages)})
        total_pages += len(pages)

    chunks = chunk_pdf_pages(all_pages)

    if not chunks:
        return {
            "ok": False,
            "message": "No text extracted from PDFs. They might be scanned/image-based PDFs.",
        }

    vector_store = build_vector_store_from_chunks(chunks)
    save_vector_store(vector_store, collection_name)

    # Store metadata for stats tab
    metadata = {
        "files": [f["name"] for f in files_info],
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "chunk_size": CHUNKING.chunk_size,
        "chunk_overlap": CHUNKING.chunk_overlap,
        "processed_at": datetime.now().isoformat(),
        "collection_name": collection_name,
    }

    metadata_dir = f"metadata/{collection_name}"
    os.makedirs(metadata_dir, exist_ok=True)

    with open(f"{metadata_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    st.session_state.document_metadata = metadata

    return {"ok": True, "message": f"Indexed {len(pdf_docs)} PDF(s) successfully."}


# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(
        "Multi PDF Chatbot",
        page_icon=":scroll:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab-list"] button {
            font-weight: bold;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.header("Multi-PDF's 📚 - Chat Agent 🤖")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "💬 Chat",
            "📊 Document Stats",
            "📜 Chat History",
            "🔍 Advanced Search",
            "🌐 Batch & Translation",
            "⚙️ Collections & Cost",
        ]
    )

    # Collections
    collections = get_collections()
    if not collections:
        collections = ["default"]

    # ---------------- Tab 1: Chat ----------------
    with tab1:
        st.subheader("💬 Interactive Chat")

        col1, col2 = st.columns([2, 1])
        with col1:
            selected_collection = st.selectbox(
                "Select Document Collection", collections, key="chat_collection"
            )
            st.session_state.current_collection = selected_collection

        user_question = st.text_area(
            "Ask a Question from the PDF Files uploaded .. ✍️📝", height=80
        )

        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            ask_button = st.button("🔍 Ask", use_container_width=True)
        with col3:
            clear_question = st.button("🔄 Clear", use_container_width=True)

        if clear_question:
            st.rerun()

        st.markdown("---")
        st.subheader("⚙️ Settings")

        col1, col2 = st.columns(2)
        with col1:
            prompt_type = st.selectbox(
                "Response Style",
                ["standard", "summarize", "detailed"],
                help="Different styles for different needs",
            )
        with col2:
            temperature = st.slider(
                "Temperature (Creativity)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Lower = more factual, Higher = more creative",
            )

        st.info(
            f"📦 Collection: **{selected_collection}** | Chunking: **{CHUNKING.chunk_size}/{CHUNKING.chunk_overlap}** | Top-k: **{RETRIEVAL.top_k}**"
        )

        if ask_button and user_question:
            result = user_input(
                user_question, prompt_type, temperature, selected_collection
            )
            if result[0] is not None:
                response, docs = result

                st.subheader("📝 Response")
                st.write(response)

                with st.expander("📍 View Sources"):
                    for i, doc in enumerate(docs, 1):
                        src = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page_number", "?")
                        chunk_idx = doc.metadata.get("chunk_index", "?")

                        with st.expander(f"Source {i} | {src} | Page {page}"):
                            st.caption(f"Chunk Index: {chunk_idx}")
                            st.text_area(
                                f"Excerpt {i}",
                                value=doc.page_content[:800] + "...",
                                height=160,
                                disabled=True,
                            )

                st.markdown("---")
                st.subheader("📋 Was this answer helpful?")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("👍 Helpful"):
                        st.session_state.feedback_data.append(
                            {
                                "question": user_question,
                                "helpful": True,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                        st.success("Thank you for the feedback!")
                with col2:
                    if st.button("👎 Not Helpful"):
                        st.session_state.feedback_data.append(
                            {
                                "question": user_question,
                                "helpful": False,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                        st.info("We'll improve!")
                with col3:
                    feedback_text = st.text_input("Additional feedback (optional)")
                    if feedback_text:
                        st.session_state.feedback_data.append(
                            {
                                "question": user_question,
                                "feedback": feedback_text,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

    # ---------------- Tab 2: Stats ----------------
    with tab2:
        st.subheader("📊 Document Analytics")

        selected_collection_stats = st.selectbox(
            "Select Collection to View Stats", collections, key="stats_collection"
        )

        metadata = load_collection_metadata(selected_collection_stats)

        if metadata:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 Total Files", len(metadata.get("files", [])))
            with col2:
                st.metric("📃 Total Pages", metadata.get("total_pages", 0))
            with col3:
                st.metric("🔗 Total Chunks", metadata.get("total_chunks", 0))
            with col4:
                st.metric(
                    "📏 Chunk Size",
                    f"{metadata.get('chunk_size', 0)} / {metadata.get('chunk_overlap', 0)}",
                )

            st.write("---")
            st.subheader("Uploaded Files")
            for file_info in metadata.get("files", []):
                st.write(f"• **{file_info}**")

            st.write("---")
            st.write(f"*Last processed: {metadata.get('processed_at', 'N/A')}*")
        else:
            st.info(f"📌 No documents in '{selected_collection_stats}' collection yet.")

    # ---------------- Tab 3: History ----------------
    with tab3:
        st.subheader("📜 Conversation History")

        if st.session_state.chat_history:
            available_collections_in_history = list(
                set([chat.get("collection", "default") for chat in st.session_state.chat_history])
            )
            filter_collection = st.selectbox(
                "Filter by Collection", ["All"] + available_collections_in_history
            )

            filtered_history = (
                st.session_state.chat_history
                if filter_collection == "All"
                else [
                    c
                    for c in st.session_state.chat_history
                    if c.get("collection") == filter_collection
                ]
            )

            for i, chat in enumerate(reversed(filtered_history), 1):
                with st.expander(
                    f"💬 Q{len(filtered_history) - i + 1}: {chat['question'][:60]}..."
                ):
                    st.write(f"**Time:** {chat['timestamp']}")
                    st.write(
                        f"**Type:** {chat['prompt_type']} | **Collection:** {chat.get('collection', 'default')}"
                    )
                    st.write(f"**Tokens:** {chat.get('tokens_used', 0)}")
                    st.write(f"\n**Question:**\n{chat['question']}")
                    st.write(f"\n**Answer:**\n{chat['answer']}")

                    with st.expander("Sources"):
                        for s in chat.get("sources", []):
                            st.write(
                                f"- {s.get('source', 'Unknown')} | Page {s.get('page', '?')} | Chunk {s.get('chunk_index', '?')}"
                            )

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                chat_json = json.dumps(filtered_history, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=chat_json,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )

            with col2:
                txt_content = "\n\n".join(
                    [
                        f"Q: {chat['question']}\n\nA: {chat['answer']}\n\n{'='*80}"
                        for chat in filtered_history
                    ]
                )
                st.download_button(
                    label="📥 Download TXT",
                    data=txt_content,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )
        else:
            st.info("📌 No chat history yet. Start asking questions!")

    # ---------------- Tab 4: Search ----------------
    with tab4:
        st.subheader("🔍 Advanced Search Features")

        search_type = st.radio("Select Search Type", ["Keyword Search", "Semantic Search"])

        if search_type == "Keyword Search":
            st.write("Search for specific keywords in your PDFs")
            st.info("📌 Keyword search works only on uploaded files in the current session.")
        else:
            st.write("Semantic search using vector store")

            selected_collection_search = st.selectbox(
                "Select Collection", collections, key="search_collection"
            )

            semantic_query = st.text_input("Enter your semantic search query")
            num_results = st.slider("Number of results", 1, 10, 5)

            if st.button("🔍 Semantic Search"):
                embeddings = get_embeddings()
                db = load_vector_store(selected_collection_search, embeddings)

                if db is None:
                    st.error(f"Collection '{selected_collection_search}' is empty")
                else:
                    results = db.similarity_search(semantic_query, k=num_results)
                    st.success(f"Found {len(results)} results")

                    for i, result in enumerate(results, 1):
                        src = result.metadata.get("source", "Unknown")
                        page = result.metadata.get("page_number", "?")
                        with st.expander(f"Result {i} | {src} | Page {page}"):
                            st.write(result.page_content[:700] + "...")

    # ---------------- Tab 5: Batch & Translation ----------------
    with tab5:
        st.subheader("🌐 Batch Processing & Translation")

        batch_tab1, batch_tab2, batch_tab3 = st.tabs(
            ["Batch Q&A", "Translate", "Auto Q&A Generation"]
        )

        with batch_tab1:
            st.write("Ask multiple questions at once")
            questions_text = st.text_area("Enter questions (one per line)", height=150)

            if st.button("📤 Process Batch"):
                if questions_text:
                    questions = [q.strip() for q in questions_text.split("\n") if q.strip()]

                    progress_bar = st.progress(0)
                    results_list = []

                    for idx, q in enumerate(questions):
                        with st.spinner(f"Processing Q{idx+1}/{len(questions)}..."):
                            result = user_input(
                                q, "standard", 0.3, st.session_state.current_collection
                            )
                            if result[0]:
                                results_list.append({"question": q, "answer": result[0]})
                        progress_bar.progress((idx + 1) / len(questions))

                    st.success(f"Processed {len(results_list)} questions!")

                    for i, item in enumerate(results_list, 1):
                        with st.expander(f"Q{i}: {item['question']}"):
                            st.write(item["answer"])

        with batch_tab2:
            st.write("Translate text to different languages")
            text_to_translate = st.text_area("Enter text to translate")
            target_lang = st.selectbox(
                "Target Language",
                ["Spanish", "French", "German", "Chinese", "Japanese", "Hindi", "Portuguese"],
            )

            if st.button("🌍 Translate"):
                with st.spinner("Translating..."):
                    translated = translate_text(text_to_translate, target_lang)
                    st.write("**Translated Text:**")
                    st.write(translated)

        with batch_tab3:
            st.write("Automatically generate Q&A pairs from document text")

            selected_collection_qa = st.selectbox(
                "Select Collection", collections, key="qa_collection"
            )
            num_qa_pairs = st.slider("Number of Q&A pairs to generate", 3, 15, 5)

            if st.button("🚀 Generate Q&A Pairs"):
                metadata = load_collection_metadata(selected_collection_qa)
                if metadata:
                    with st.spinner("Generating Q&A pairs..."):
                        qa_pairs = generate_qa_pairs("Sample document content", num_qa_pairs)
                        st.write(qa_pairs)
                else:
                    st.error(f"Collection '{selected_collection_qa}' is empty")

    # ---------------- Tab 6: Collections & Cost ----------------
    with tab6:
        st.subheader("⚙️ Collections & Cost Management")

        st.write("### 📦 Vector Store Collections")

        with st.expander("View All Collections"):
            for collection in collections:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"📁 **{collection}**")
                with col_b:
                    if st.button(f"🗑️ Delete", key=f"delete_{collection}"):
                        delete_collection(collection)
                        st.success(f"Deleted '{collection}'")
                        st.rerun()

        st.write("### 💰 Cost & Token Tracking")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔤 Total Tokens Used", st.session_state.token_count)
        with col2:
            estimated_cost = estimate_cost(st.session_state.token_count)
            st.metric("💵 Estimated Cost", f"${estimated_cost:.6f}")
        with col3:
            st.metric("📝 Total Conversations", len(st.session_state.chat_history))

    # ---------------- Sidebar: PDF Management ----------------
    with st.sidebar:
        st.title("📁 PDF Management")

        st.subheader("Create or Select Collection")

        new_collection = st.text_input(
            "New Collection Name (optional)",
            placeholder="e.g., 'Project A'",
            key="new_collection_input",
        )

        if new_collection and new_collection.strip():
            if st.button("✅ Create Collection", key="create_collection_btn"):
                os.makedirs(f"vector_stores/{new_collection}", exist_ok=True)
                os.makedirs(f"metadata/{new_collection}", exist_ok=True)
                st.session_state.current_collection = new_collection
                st.success(f"✅ Collection '{new_collection}' created!")
                st.rerun()

        st.subheader("Select Active Collection")
        updated_collections = get_collections()
        if not updated_collections:
            updated_collections = ["default"]

        selected_active_collection = st.selectbox(
            "Active Collection",
            updated_collections,
            index=updated_collections.index(st.session_state.current_collection)
            if st.session_state.current_collection in updated_collections
            else 0,
            key="sidebar_collection_select",
        )
        st.session_state.current_collection = selected_active_collection
        st.info(f"📍 Active: **{selected_active_collection}**")

        st.write("---")
        st.write("### 📤 Upload PDFs")

        pdf_docs = st.file_uploader(
            "Upload your PDF Files", accept_multiple_files=True, type="pdf"
        )

        st.caption("Day 2 upgrade: page-aware chunking + metadata preserved")
        st.caption(f"Chunk size: {CHUNKING.chunk_size} | Overlap: {CHUNKING.chunk_overlap}")

        if st.button("Submit & Process", type="primary"):
            if pdf_docs:
                with st.spinner("Processing and indexing..."):
                    result = process_and_index_pdfs(
                        pdf_docs, st.session_state.current_collection
                    )
                    if result["ok"]:
                        st.success(
                            f"✅ Done! Processed {len(pdf_docs)} PDF(s) in collection '{st.session_state.current_collection}'"
                        )
                    else:
                        st.error(result["message"])
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

    # Footer
    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/PratikJadhav27" target="_blank">Pratik Jadhav</a> | Made with ❤️ | Multi-PDF RAG Chat App
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
