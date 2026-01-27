# PDF Corpus Chat (Multi-PDF RAG)

A clean, production-style **Multi-PDF Chat Application** built using **Retrieval-Augmented Generation (RAG)**.  
Upload multiple PDFs, organize them into collections, and chat with **grounded, citation-backed answers** based on your document corpus.

---

## 📋 Table of Contents

- [PDF Corpus Chat (Multi-PDF RAG)](#pdf-corpus-chat-multi-pdf-rag)
  - [📋 Table of Contents](#-table-of-contents)
  - [✨ Features](#-features)
  - [🧱 Tech Stack](#-tech-stack)
  - [📁 Project Structure](#-project-structure)
  - [🚀 Getting Started](#-getting-started)
    - [Windows](#windows)
    - [Mac / Linux](#mac--linux)
    - [Docker](#docker)
    - [Docker](#docker-1)
  - [⚙️ Configuration](#️-configuration)
    - [Chunking](#chunking)
    - [Retrieval](#retrieval)
  - [🧪 Evaluation Mode](#-evaluation-mode)
  - [🔐 Environment Variables](#-environment-variables)
    - [Required](#required)
  - [🧭 Roadmap](#-roadmap)
    - [Planned Improvements](#planned-improvements)
  - [📜 License](#-license)

---

## ✨ Features

- 📄 Upload and chat with **multiple PDFs**
- 🗂️ Create and manage **Collections** (separate knowledge bases per topic/use-case)
- 💾 Persistent **FAISS vector store** per collection
- 🔍 Retrieval strategies: **Similarity** and **MMR** (diverse context retrieval)
- 🎛️ Configurable **Top-K** chunk selection
- 📌 **Citation-aware responses** (document + page)
- 📊 **PDF extraction statistics** (extracted / empty / failed pages)
- 🧪 **Evaluation Mode** for batch Q&A with JSON report export

---

## 🧱 Tech Stack

- **Streamlit** — User Interface  
- **LangChain** — RAG pipeline  
- **Google Gemini** — LLM + embeddings  
- **FAISS** — Vector similarity search  
- **PyPDF** — PDF text extraction  

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
```

---

## 🚀 Getting Started

### Windows

1. **Clone the repository**
   ```bash
   git clone https://github.com/PratikJadhav27/PDF-Corpus-Chat.git
   cd PDF-Corpus-Chat
   ```

2. **Setup** (creates virtual environment and installs dependencies)
   ```bash
   .\scripts\setup.ps1
   ```

3. **Add API key**
   
   Create a `.env` file in the project root and set:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

4. **Run the app**
   ```bash
   .\scripts\run.ps1
   ```
   
   Open in browser: http://localhost:8501

### Mac / Linux

> **Note:** PowerShell scripts are Windows-only. Mac/Linux users can run the app manually.

1. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add environment variables**
   
   Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

4. **Run the app**
   ```bash
   streamlit run app/main.py
   ```

### Docker
### Docker

1. **Add your API key**
   
   Create `.env` (or copy from `.env.example`) and set:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

2. **Build and run**
   ```bash
   docker compose up --build
   ```

3. **Open the app**
   
   Visit: http://localhost:8501

4. **Stop the container**
   ```bash
   docker compose down
   ```

---

## ⚙️ Configuration

### Chunking

- **Chunk size:** 1500 tokens
- **Chunk overlap:** 200 tokens

### Retrieval

- **Default mode:** Similarity search
- **Default Top-K:** 4 chunks

---

## 🧪 Evaluation Mode

The **Evaluation** tab allows you to assess RAG quality:

**Steps:**
1. Enter multiple questions (one per line)
2. Select retrieval mode (Similarity / MMR)
3. Configure Top-K
4. Run batch evaluation
5. Export results as a JSON report

**Tracked Metrics:**
- Citation rate
- IDK (no-answer) rate
- Latency per question

---

## 🔐 Environment Variables

### Required

- `GOOGLE_API_KEY` — Google Gemini API key

> ⚠️ **Important:** Do NOT commit `.env` to GitHub. Add it to `.gitignore`.

---

## 🧭 Roadmap

This project follows a clean GitHub workflow: **Issues → Branches → Pull Requests → Merge**

### Planned Improvements

- [ ] Hybrid Search (BM25 + vector search)
- [ ] Retrieval re-ranking
- [ ] OCR detection for scanned PDFs
- [ ] Improved citation visualization
- [ ] CI pipeline and automated tests

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.