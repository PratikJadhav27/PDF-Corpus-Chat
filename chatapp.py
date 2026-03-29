# =============================================================================
# DEPRECATED — LEGACY READONLY
# =============================================================================
# This file is the original v1 app before the production RAG upgrade.
# It MUST NOT be modified or run in production.
#
# The canonical runtime entry point is: app/main.py
#   Run with: streamlit run app/main.py
#
# This file is preserved in git history for reference only.
# See README.md for the current architecture and evaluation framework.
# =============================================================================

import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import json
from datetime import datetime
from pathlib import Path
import shutil

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state
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

def get_pdf_text(pdf_docs):
    text = ""
    metadata = {"files": [], "total_pages": 0}
    
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pdf_name = pdf.name
        num_pages = len(pdf_reader.pages)
        metadata["files"].append({"name": pdf_name, "pages": num_pages})
        metadata["total_pages"] += num_pages
        
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text, metadata

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, pdf_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    # Save to collection-specific directory
    collection_name = st.session_state.current_collection
    collection_dir = f"vector_stores/{collection_name}"
    os.makedirs(collection_dir, exist_ok=True)
    vector_store.save_local(collection_dir)
    
    # Store metadata
    metadata = {
        "files": [pdf.name for pdf in pdf_docs],
        "total_pages": sum(len(PdfReader(pdf).pages) for pdf in pdf_docs),
        "total_chunks": len(text_chunks),
        "chunk_size": 50000,
        "processed_at": datetime.now().isoformat(),
        "collection_name": collection_name
    }
    
    metadata_dir = f"metadata/{collection_name}"
    os.makedirs(metadata_dir, exist_ok=True)
    with open(f"{metadata_dir}/metadata.json", "w") as f:
        json.dump(metadata, f)
    
    st.session_state.document_metadata = metadata
    
    return vector_store


def get_conversational_chain(prompt_type="standard", temperature=0.3):
    
    # Different prompt templates for different use cases
    prompts = {
        "standard": """
Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
If the answer is not in the provided context, just say "answer is not available in the context".
Don't provide wrong answers.

Context:
{context}

Question:
{question}

Answer:""",
        
        "summarize": """
Based on the provided context, create a concise summary that answers the question.
Focus on the main points and key information.

Context:
{context}

Question:
{question}

Summary:""",
        
        "detailed": """
Provide an extremely detailed and comprehensive answer based on the context.
Include examples, explanations, and all relevant information found in the context.

Context:
{context}

Question:
{question}

Detailed Answer:"""
    }
    
    prompt_template = prompts.get(prompt_type, prompts["standard"])
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return chain



def user_input(user_question, prompt_type, temperature, collection_name="default"):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    # Load from collection-specific directory
    collection_dir = f"vector_stores/{collection_name}"
    if not os.path.exists(collection_dir):
        st.error(f"Please upload and process PDF files first in collection '{collection_name}'!")
        return None, None
    
    new_db = FAISS.load_local(collection_dir, embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=4)

    chain = get_conversational_chain(prompt_type=prompt_type, temperature=temperature)

    # Format documents for context
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    
    response = chain.invoke({"context": context, "question": user_question})
    
    # Count tokens
    try:
        token_count = len(genai.count_text_tokens(response).total_tokens)
        st.session_state.token_count += token_count
    except:
        token_count = 0

    # Add to chat history
    st.session_state.chat_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": user_question,
        "answer": response,
        "sources": [doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown' for doc in docs],
        "prompt_type": prompt_type,
        "tokens_used": token_count,
        "collection": collection_name
    })
    
    return response, docs


# ============= NEW FEATURES =============

def get_collections():
    """Get list of all vector store collections"""
    collections_dir = "vector_stores"
    if not os.path.exists(collections_dir):
        return []
    return [d for d in os.listdir(collections_dir) if os.path.isdir(os.path.join(collections_dir, d))]


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
    """Estimate API cost based on tokens used"""
    # Gemini 2.5 Flash pricing: $0.075 per 1M input tokens, $0.3 per 1M output tokens
    # Simplified calculation
    cost_per_million = 0.15
    estimated_cost = (tokens_used / 1_000_000) * cost_per_million
    return estimated_cost


def keyword_search_in_pdf(pdf_docs, keyword):
    """Search for keywords in uploaded PDFs"""
    results = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            if keyword.lower() in text.lower():
                results.append({
                    "file": pdf.name,
                    "page": page_num,
                    "excerpt": text[max(0, text.lower().find(keyword.lower())-100):text.lower().find(keyword.lower())+200]
                })
    return results


def translate_text(text, target_language):
    """Translate text using Google Generative AI"""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(
        template=f"Translate the following text to {{target_lang}} language. Only provide the translation, nothing else.\n\nText: {{text}}",
        input_variables=["text", "target_lang"]
    )
    
    chain = prompt | model | StrOutputParser()
    try:
        result = chain.invoke({"text": text, "target_lang": target_language})
        return result
    except:
        return text


def generate_qa_pairs(text, num_pairs=5):
    """Automatically generate Q&A pairs from document text"""
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
        result = chain.invoke({"text": text[:2000]})  # Limit text for cost
        return result
    except:
        return "Could not generate Q&A pairs"


def load_collection_metadata(collection_name):
    """Load metadata for a specific collection"""
    metadata_file = f"metadata/{collection_name}/metadata.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return None




def main():
    # Set page config with custom theme
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for better UI
    st.markdown("""
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
    """, unsafe_allow_html=True)
    
    st.header("Multi-PDF's 📚 - Chat Agent 🤖")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 Chat",
        "📊 Document Stats",
        "📜 Chat History",
        "🔍 Advanced Search",
        "🌐 Batch & Translation",
        "⚙️ Collections & Cost"
    ])

    # Get available collections - refresh each time
    collections = get_collections()
    if not collections:
        collections = ["default"]

    with tab1:
        st.subheader("💬 Interactive Chat")
        
        # Collection selector
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_collection = st.selectbox(
                "Select Document Collection",
                collections,
                key="chat_collection"
            )
            st.session_state.current_collection = selected_collection
        
        # Main chat interface
        user_question = st.text_area("Ask a Question from the PDF Files uploaded .. ✍️📝", height=80)

        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            ask_button = st.button("🔍 Ask", use_container_width=True)
        with col3:
            clear_question = st.button("🔄 Clear", use_container_width=True)

        # Settings section
        st.markdown("---")
        st.subheader("⚙️ Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            prompt_type = st.selectbox(
                "Response Style",
                ["standard", "summarize", "detailed"],
                help="Different styles for different needs"
            )
        with col2:
            temperature = st.slider(
                "Temperature (Creativity)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Lower = more factual, Higher = more creative"
            )

        # Process question
        if ask_button and user_question:
            result = user_input(user_question, prompt_type, temperature, selected_collection)
            if result[0] is not None:
                response, docs = result
                
                # Display response
                st.subheader("📝 Response")
                st.write(response)
                
                # Copy button
                col1, col2 = st.columns([4, 1])
                with col2:
                    st.code("Copy", language="")
                
                # Display sources with expander
                with st.expander("📍 View Sources"):
                    for i, doc in enumerate(docs, 1):
                        with st.expander(f"Source {i}"):
                            st.text_area(f"Excerpt {i}", value=doc.page_content[:500] + "...", height=150, disabled=True)
                
                # Feedback system
                st.markdown("---")
                st.subheader("📋 Was this answer helpful?")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("👍 Helpful"):
                        st.session_state.feedback_data.append({
                            "question": user_question,
                            "helpful": True,
                            "timestamp": datetime.now().isoformat()
                        })
                        st.success("Thank you for the feedback!")
                with col2:
                    if st.button("👎 Not Helpful"):
                        st.session_state.feedback_data.append({
                            "question": user_question,
                            "helpful": False,
                            "timestamp": datetime.now().isoformat()
                        })
                        st.info("We'll improve!")
                with col3:
                    feedback_text = st.text_input("Additional feedback (optional)")
                    if feedback_text:
                        st.session_state.feedback_data.append({
                            "question": user_question,
                            "feedback": feedback_text,
                            "timestamp": datetime.now().isoformat()
                        })

    with tab2:
        st.subheader("📊 Document Analytics")
        
        # Collection selector for stats
        selected_collection_stats = st.selectbox(
            "Select Collection to View Stats",
            collections,
            key="stats_collection"
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
                st.metric("📏 Chunk Size", f"{metadata.get('chunk_size', 0):,} chars")
            
            st.write("---")
            st.subheader("Uploaded Files")
            for file_info in metadata.get("files", []):
                st.write(f"• **{file_info}**")
            
            st.write("---")
            st.write(f"*Last processed: {metadata.get('processed_at', 'N/A')}*")
        else:
            st.info(f"📌 No documents in '{selected_collection_stats}' collection yet.")

    with tab3:
        st.subheader("📜 Conversation History")
        
        if st.session_state.chat_history:
            # Filter by collection
            available_collections_in_history = list(set([chat.get('collection', 'default') for chat in st.session_state.chat_history]))
            filter_collection = st.selectbox("Filter by Collection", ["All"] + available_collections_in_history)
            
            filtered_history = st.session_state.chat_history if filter_collection == "All" else [c for c in st.session_state.chat_history if c.get('collection') == filter_collection]
            
            for i, chat in enumerate(reversed(filtered_history), 1):
                with st.expander(f"💬 Q{len(filtered_history) - i + 1}: {chat['question'][:60]}..."):
                    st.write(f"**Time:** {chat['timestamp']}")
                    st.write(f"**Type:** {chat['prompt_type']} | **Collection:** {chat.get('collection', 'default')}")
                    st.write(f"**Tokens:** {chat.get('tokens_used', 0)}")
                    st.write(f"\n**Question:**\n{chat['question']}")
                    st.write(f"\n**Answer:**\n{chat['answer']}")
                    
                    # Copy button
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📋 Copy Answer {i}", key=f"copy_{i}"):
                            st.success("Copied to clipboard!")
            
            # Export options
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Export as JSON"):
                    chat_json = json.dumps(filtered_history, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=chat_json,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("📥 Export as TXT"):
                    txt_content = "\n\n".join([f"Q: {chat['question']}\n\nA: {chat['answer']}\n\n{'='*80}" for chat in filtered_history])
                    st.download_button(
                        label="Download TXT",
                        data=txt_content,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        else:
            st.info("📌 No chat history yet. Start asking questions!")

    with tab4:
        st.subheader("🔍 Advanced Search Features")
        
        search_type = st.radio("Select Search Type", ["Keyword Search", "Semantic Search"])
        
        if search_type == "Keyword Search":
            st.write("Search for specific keywords in your PDFs")
            keyword = st.text_input("Enter keyword to search")
            
            if st.button("🔍 Search PDFs"):
                st.info("📌 Upload PDFs first using the sidebar to perform keyword search")
        
        else:  # Semantic Search
            st.write("Semantic search using AI - finds meaning-related results")
            
            # Collection selector for search
            selected_collection_search = st.selectbox(
                "Select Collection",
                collections,
                key="search_collection"
            )
            
            semantic_query = st.text_input("Enter your semantic search query")
            num_results = st.slider("Number of results", 1, 10, 5)
            
            if st.button("🔍 Semantic Search"):
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                collection_dir = f"vector_stores/{selected_collection_search}"
                
                if os.path.exists(collection_dir):
                    db = FAISS.load_local(collection_dir, embeddings, allow_dangerous_deserialization=True)
                    results = db.similarity_search(semantic_query, k=num_results)
                    
                    st.success(f"Found {len(results)} results")
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i}"):
                            st.write(result.page_content[:500] + "...")
                else:
                    st.error(f"Collection '{selected_collection_search}' is empty")

    with tab5:
        st.subheader("🌐 Batch Processing & Translation")
        
        batch_tab1, batch_tab2, batch_tab3 = st.tabs(["Batch Q&A", "Translate", "Auto Q&A Generation"])
        
        with batch_tab1:
            st.write("Ask multiple questions at once")
            questions_text = st.text_area("Enter questions (one per line)", height=150)
            
            if st.button("📤 Process Batch"):
                if questions_text:
                    questions = questions_text.split('\n')
                    questions = [q.strip() for q in questions if q.strip()]
                    
                    progress_bar = st.progress(0)
                    results_list = []
                    
                    for idx, q in enumerate(questions):
                        with st.spinner(f"Processing Q{idx+1}/{len(questions)}..."):
                            result = user_input(q, "standard", 0.3, st.session_state.current_collection)
                            if result[0]:
                                results_list.append({"question": q, "answer": result[0]})
                        progress_bar.progress((idx + 1) / len(questions))
                    
                    st.success(f"Processed {len(results_list)} questions!")
                    
                    # Display results
                    for i, item in enumerate(results_list, 1):
                        with st.expander(f"Q{i}: {item['question']}"):
                            st.write(item['answer'])
        
        with batch_tab2:
            st.write("Translate text to different languages")
            text_to_translate = st.text_area("Enter text to translate")
            target_lang = st.selectbox("Target Language", ["Spanish", "French", "German", "Chinese", "Japanese", "Hindi", "Portuguese"])
            
            if st.button("🌍 Translate"):
                with st.spinner("Translating..."):
                    translated = translate_text(text_to_translate, target_lang)
                    st.write("**Translated Text:**")
                    st.write(translated)
        
        with batch_tab3:
            st.write("Automatically generate Q&A pairs from document text")
            
            # Collection selector for QA generation
            selected_collection_qa = st.selectbox(
                "Select Collection",
                collections,
                key="qa_collection"
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

    with tab6:
        st.subheader("⚙️ Collections & Cost Management")
        
        # Collection Management
        st.write("### 📦 Vector Store Collections")
        
        col1, col2 = st.columns([2, 1])
        with col1:
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
        
        # Cost & Token Tracking
        st.write("### 💰 Cost & Token Tracking")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔤 Total Tokens Used", st.session_state.token_count)
        with col2:
            estimated_cost = estimate_cost(st.session_state.token_count)
            st.metric("💵 Estimated Cost", f"${estimated_cost:.6f}")
        with col3:
            st.metric("📝 Total Conversations", len(st.session_state.chat_history))
        
        # Token usage breakdown
        if st.session_state.chat_history:
            st.write("---")
            st.write("### 📊 Token Usage Breakdown")
            
            collection_tokens = {}
            for chat in st.session_state.chat_history:
                collection = chat.get('collection', 'default')
                tokens = chat.get('tokens_used', 0)
                collection_tokens[collection] = collection_tokens.get(collection, 0) + tokens
            
            for collection, tokens in collection_tokens.items():
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**{collection}**")
                with col2:
                    st.write(f"`{tokens}` tokens")

    # Sidebar - PDF Upload
    with st.sidebar:
        st.title("📁 PDF Management")
        
        # Create new collection or use existing
        st.subheader("Create or Select Collection")
        
        new_collection = st.text_input("New Collection Name (optional)", placeholder="e.g., 'Project A'", key="new_collection_input")
        
        if new_collection and new_collection.strip():
            if st.button("✅ Create Collection", key="create_collection_btn"):
                # Create the directories for the new collection
                os.makedirs(f"vector_stores/{new_collection}", exist_ok=True)
                os.makedirs(f"metadata/{new_collection}", exist_ok=True)
                st.session_state.current_collection = new_collection
                st.success(f"✅ Collection '{new_collection}' created!")
                st.rerun()
        
        # Select existing collection
        st.subheader("Select Active Collection")
        updated_collections = get_collections()
        if not updated_collections:
            updated_collections = ["default"]
        
        selected_active_collection = st.selectbox(
            "Active Collection",
            updated_collections,
            index=updated_collections.index(st.session_state.current_collection) if st.session_state.current_collection in updated_collections else 0,
            key="sidebar_collection_select"
        )
        st.session_state.current_collection = selected_active_collection
        st.info(f"📍 Active: **{selected_active_collection}**")
        
        st.write("---")
        st.write("### 📤 Upload PDFs")
        
        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True,
            type="pdf"
        )
        
        if st.button("Submit & Process", type="primary"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text, metadata = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, pdf_docs)
                    st.success(f"✅ Done! Processed {len(pdf_docs)} PDF(s) in collection '{st.session_state.current_collection}'")
            else:
                st.error("Please upload at least one PDF file")
        
        st.write("---")
        
        # Clear functions
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
        
        st.write("---")
        
        # Display images
        try:
            st.image("img/Robot.jpg", use_column_width=True)
        except:
            pass
        
        
        st.caption("AI App created by @ Pratik Jadhav")

    # Footer
    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/PratikJadhav27" target="_blank">Pratik Jadhav</a> | Made with ❤️ | Enhanced with 10 Advanced Features
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
