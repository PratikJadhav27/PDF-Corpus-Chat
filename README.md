# PDF Corpus Chat (Multi-PDF RAG)

A clean, production-style **Multi-PDF Chat App** built using **Retrieval-Augmented Generation (RAG)**.  
Upload multiple PDFs, organize them into collections, and chat with grounded answers based on the document corpus.

---

## 🚀 Getting Started (Windows)

### 1. Clone the repo

```bash
git clone https://github.com/PratikJadhav27/PDF-Corpus-Chat.git
cd PDF-Corpus-Chat
```

### 2. Setup (creates venv + installs deps)

.\scripts\setup.ps1

### 3. Add API key

Edit .env and set:

GOOGLE_API_KEY=your_api_key_here

### 4. Run the app

.\scripts\run.ps1

Also add Linux/Mac instructions if you want later, but Windows is enough since you’re on Windows.

---

## 🚀 Features

- 📄 Upload and chat with **multiple PDFs**
- 🗂️ **Collections** (separate knowledge bases per topic/use-case)
- 💾 Persistent **FAISS vector store** per collection
- 🔍 Retrieval-based context injection for more accurate answers
- 🧾 Metadata tracking (pages, chunks, processed time)

### Default Chunking

- Chunk size: **1500**
- Chunk overlap: **200**

---

## 🧱 Tech Stack

- **Streamlit** (UI)
- **LangChain** (RAG pipeline)
- **Google Gemini** (LLM + embeddings)
- **FAISS** (vector search)

---

## 🛠️ Setup

### 1) Create & activate virtual environment

python -m venv .venv

.venv\Scripts\Activate.ps1

### 2) Install dependencies

pip install -r requirements.txt

### 3) Add environment variables

Create a .env file in the root folder:

GOOGLE_API_KEY=your_api_key_here
⚠️ Do NOT commit .env to GitHub.

### ▶️ Run the app

streamlit run app/main.py

## 📁 Project Structure

text
Copy code
PDF-Corpus-Chat/
├── app/
│   ├── main.py
│   └── ui/
├── src/
│   ├── ingestion/
│   ├── rag/
│   └── utils/
├── tests/
├── assets/
├── requirements.txt
├── .env.example
└── README.md

## 🧭 Roadmap

This project is actively being developed with a clean GitHub workflow (issues → branches → PRs → merges).

Planned improvements:

- Fix chunking strategy (smaller chunks, better overlap)

- Add citations (PDF + page number) in answers

- Add MMR retrieval and top-k tuning

- Add scanned PDF detection (OCR warning)

- Add collection management (delete/reset)

- Add evaluation scripts + test cases

- Improve prompts for stricter grounded responses
  