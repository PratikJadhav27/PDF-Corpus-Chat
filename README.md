# PDF Corpus Chat (Multi-PDF RAG)

A production-grade **Multi-PDF Chat Application** built using **Retrieval-Augmented Generation (RAG)**.  
Upload multiple PDFs, organize them into collections, and chat with **grounded, citation-backed answers** based on your document corpus.

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🧱 Tech Stack](#-tech-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [⚙️ Configuration](#️-configuration)
- [🧪 Local Evaluation](#-local-evaluation)
- [🔁 CI Regression Gate](#-ci-regression-gate)
- [🔐 Environment Variables](#-environment-variables)
- [📜 License](#-license)

---

## ✨ Features

- 📄 Upload and chat with **multiple PDFs**
- 🗂️ Create and manage **Collections** (separate knowledge bases per topic/use-case)
- 💾 Persistent **FAISS vector store** per collection
- 🔍 **Hybrid retrieval**: BM25 keyword search + semantic vector search fused via **Reciprocal Rank Fusion (RRF)**
- 🎯 **Cross-encoder style re-ranking** of retrieved chunks
- 📌 **Versioned prompt templates** with strict citation enforcement
- ✅ **Citation validation + guarded fallback** (hallucinated citations are caught and rejected)
- 🕵️ **Observability tracing** — every request produces a structured JSONL trace with latency, scores, and prompt info
- 🧪 **Evaluation Mode** (UI + CLI) with LLM-as-judge metrics
- 🔁 **CI regression gate** to prevent metric degradation on PRs

---

## 🏗️ Architecture

```
Layer 1 — Ingestion:    PDFs → Parser → Chunker → Embedder → FAISS + BM25 corpus
Layer 2 — Retrieval:    Query → BM25 + Vector → RRF Fusion → Re-ranker → Top-K
Layer 3 — Generation:   Top-K chunks → Versioned prompt → LLM → Citation-validated answer
Layer 4 — Observability: JSONL trace per request (latency, scores, model, prompt version)
Layer 5 — Evaluation:   Golden dataset CI gate (faithfulness, citation validity thresholds)
```

---

## 🧱 Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| LLM + Embeddings | Google Gemini (gemini-2.5-flash, embedding-001) |
| Vector Store | FAISS |
| BM25 | Custom lightweight implementation (no extra deps) |
| Tracing | Local JSONL sink (Langfuse/LangSmith ready) |
| CI | GitHub Actions |
| Testing | pytest |

---

## 📁 Project Structure

```text
Multi-PDFs_ChatApp_AI-Agent/
├── app/
│   └── main.py                    # ← Canonical runtime entry point
├── src/
│   ├── config.py                  # Chunking, retrieval, observability config
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   └── chunking.py
│   ├── rag/
│   │   ├── bm25_index.py          # Lightweight BM25 implementation
│   │   ├── fusion.py              # Reciprocal Rank Fusion (RRF)
│   │   ├── reranker.py            # Lexical overlap reranker
│   │   ├── retriever.py           # Hybrid + standard retrieval
│   │   ├── vector_store.py        # FAISS persistence helpers
│   │   ├── prompt_registry.py     # Versioned prompt templates
│   │   ├── prompting.py           # Context + prompt builder
│   │   └── validators.py          # Citation validation + guard fallback
│   ├── eval/
│   │   ├── evaluator.py           # Heuristic + LLM-judge evaluation
│   │   └── metrics.py             # Faithfulness, relevance, completeness
│   ├── observability/
│   │   ├── tracing.py             # Structured trace record builder
│   │   └── store.py               # JSONL append sink
│   └── utils/
├── evals/
│   ├── golden_dataset.jsonl       # Golden Q&A pairs for CI evaluation
│   ├── fixtures/                  # (optional) small PDFs for CI ingestion
│   └── reports/                   # Auto-generated eval reports
├── scripts/
│   ├── run_eval.py                # CLI batch evaluation runner
│   ├── setup.ps1
│   └── run.ps1
├── tests/
│   ├── test_fusion.py
│   ├── test_validators.py
│   └── test_retriever.py
├── .github/workflows/
│   └── eval-gate.yml              # CI regression gate
├── conftest.py                    # pytest path setup
├── requirements.txt
└── chatapp.py                     # ⚠️ DEPRECATED — legacy readonly
```

---

## 🚀 Getting Started

### Windows

```powershell
git clone https://github.com/PratikJadhav27/PDF-Corpus-Chat.git
cd PDF-Corpus-Chat
.\scripts\setup.ps1
```

Create `.env`:
```
GOOGLE_API_KEY=your_api_key_here
```

Run the app:
```powershell
.\scripts\run.ps1
# or directly:
streamlit run app/main.py
```

### Mac / Linux

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your_api_key_here" > .env
streamlit run app/main.py
```

### Docker

```bash
cp .env.example .env  # add your GOOGLE_API_KEY
docker compose up --build
# Visit: http://localhost:8501
```

---

## ⚙️ Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNKING.chunk_size` | 1500 | Token chunk size |
| `CHUNKING.chunk_overlap` | 200 | Overlap between chunks |
| `RETRIEVAL.top_k` | 4 | Final top-K chunks returned |
| `RETRIEVAL.mode` | similarity | `similarity` / `mmr` / `hybrid` |
| `OBSERVABILITY.enabled` | False | Set `OBSERVABILITY_ENABLED=true` to enable tracing |
| `OBSERVABILITY.traces_path` | `logs/rag_traces.jsonl` | JSONL trace output path |

---

## 🧪 Local Evaluation

### Run Unit Tests

```bash
pytest tests/ -v
```

### Run Batch CLI Evaluation

The CLI runner evaluates the **golden dataset** against a live collection using LLM-as-judge metrics:

```bash
python scripts/run_eval.py \
  --collection my_collection \
  --mode hybrid \
  --top-k 4 \
  --dataset evals/golden_dataset.jsonl \
  --faithfulness-threshold 0.85 \
  --citation-validity-threshold 0.90
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--collection` | required | Collection name to evaluate against |
| `--mode` | hybrid | Retrieval mode: `similarity`, `mmr`, `hybrid` |
| `--top-k` | 4 | Final chunks per question |
| `--candidate-k` | 12 | Hybrid candidate pool before rerank |
| `--dataset` | `evals/golden_dataset.jsonl` | Path to golden Q&A file |
| `--output` | auto-timestamped | Output JSON report path |
| `--no-llm-metrics` | off | Skip LLM-judge (faster, heuristic only) |
| `--faithfulness-threshold` | 0.85 | Fail if avg faithfulness < value |
| `--citation-validity-threshold` | 0.90 | Fail if avg citation validity < value |

**Exit codes:** `0` = pass, `1` = metric threshold failure, `2` = setup error.

**Report format** (saved to `evals/reports/`):
```json
{
  "summary": {
    "avg_faithfulness": 0.92,
    "avg_citation_validity_ratio": 0.95,
    "avg_composite_score": 0.89,
    ...
  },
  "results": [...]
}
```

### Extend the Golden Dataset

Add records to `evals/golden_dataset.jsonl` in JSONL format:
```json
{"question": "What is X?", "expected_answer": "X is ...", "expected_sources": ["doc.pdf"], "tags": ["comprehension"]}
```

---

## 🔁 CI Regression Gate

The GitHub Actions workflow (`.github/workflows/eval-gate.yml`) runs on every PR:

1. **Unit tests** — `tests/test_fusion.py`, `tests/test_validators.py`, `tests/test_retriever.py`
2. **Eval gate** — if `evals/fixtures/*.pdf` exists, ingests and evaluates against `evals/golden_dataset.jsonl`
3. **Artifacts** — eval report uploaded to GitHub for post-merge debugging

**PR is blocked if:**
- Any unit test fails
- `avg_faithfulness < 0.85`
- `avg_citation_validity_ratio < 0.90`

To add your own fixture PDFs for CI:
```
evals/fixtures/sample_doc.pdf   # small representative PDF
```

Set `GOOGLE_API_KEY` in your GitHub repository secrets.

---

## 🔐 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ | Google Gemini API key |
| `OBSERVABILITY_ENABLED` | optional | `true` to enable JSONL tracing |
| `OBSERVABILITY_TRACES_PATH` | optional | Override default trace path |
| `EVAL_JUDGE_MODEL` | optional | Override LLM judge model (default: gemini-2.5-flash) |
| `EVAL_LLM_MODEL` | optional | Override inference model for eval CLI |

> ⚠️ **Never commit `.env`** — it is in `.gitignore`.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.