# PDF Corpus Chat (Multi-PDF RAG)

A clean, production-style **Multi-PDF Chat App** built using **Retrieval-Augmented Generation (RAG)**.  
Upload multiple PDFs, organize them into collections, and chat with grounded answers based on your document corpus.

---

## ✨ Features

- 📄 Upload and chat with **multiple PDFs**
- 🗂️ Create and manage **Collections** (separate knowledge bases per topic/use-case)
- 💾 Persistent **FAISS vector store** per collection
- 🔍 Retrieval options: **Similarity** + **MMR** (more diverse context)
- 🎛️ Configurable **Top-K** chunk retrieval
- 📌 Grounded answers with **citations (source + page)**
- 📊 **Extraction stats** (extracted/empty/failed pages per PDF)
- 🧪 **Evaluation Mode**: run batch questions + export JSON reports

---

## 🧱 Tech Stack

- **Streamlit** — UI
- **LangChain** — RAG pipeline
- **Google Gemini** — LLM + embeddings
- **FAISS** — vector similarity search
- **PyPDF / PyPDF2** — PDF text extraction

---

## 📁 Project Structure

```text
PDF-Corpus-Chat/
├── app/
│   └── main.py
├── src/
│   ├── ingestion/
│   ├── rag/
│   ├── eval/
│   └── utils/
├── scripts/
│   ├── setup.ps1
│   └── run.ps1
├── tests/
├── assets/
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── .dockerignore
└── README.md

---

## 🚀 Getting Started (Windows)

1) Clone the repo
git clone https://github.com/PratikJadhav27/PDF-Corpus-Chat.git
cd PDF-Corpus-Chat

2) Setup (creates venv + installs dependencies)
.\scripts\setup.ps1

3) Add API key
Edit the .env file and set:

GOOGLE_API_KEY=your_api_key_here

4) Run the app
.\scripts\run.ps1

Open in browser:
👉 http://localhost:8501

## Getting Started (Mac/Linux)

If you're on Mac/Linux, you can run the app manually (PowerShell scripts are Windows-only).

1) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

2) Install dependencies
pip install -r requirements.txt

3) Add environment variables
Create a .env file in the root folder:

GOOGLE_API_KEY=your_api_key_here

4) Run the app
streamlit run app/main.py

---

## 🐳 Run with Docker

1) Add your API key
Create .env (or copy from .env.example) and set:

GOOGLE_API_KEY=your_api_key_here

2) Build & run
docker compose up --build

3) Open the app
👉 http://localhost:8501

To stop:

docker compose down

## ⚙️ Default Configuration

Chunking
Chunk size: 1500

Chunk overlap: 200

Retrieval
Default mode: similarity

Default Top-K: 4

## 🧪 Evaluation Mode
You can evaluate RAG quality using the Evaluation tab:

Enter multiple questions (one per line)

Choose retrieval mode (similarity / mmr)

Choose Top-K

Run evaluation

Export results as a JSON report

This helps track:

Citation rate

IDK rate

Latency per question

🔐 Environment Variables
Required:

GOOGLE_API_KEY — Gemini API Key

⚠️ Do NOT commit .env to GitHub.

## 🧭 Roadmap
This project is actively developed using a clean GitHub workflow:

Issues → Branches → PRs → Merge

Planned improvements:

Hybrid Search (BM25 + vector search)

Reranking for improved retrieval precision

OCR detection warning for scanned PDFs

Better UI for citations and sources

CI pipeline + unit tests

## 📜 License
This project is licensed under the MIT License.

