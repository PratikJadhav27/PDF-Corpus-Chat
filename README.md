# PDF Corpus Chat — Multi-Document RAG System

A **multi-document RAG (Retrieval-Augmented Generation) system** that lets you upload PDFs, organise them into collections, and get **citation-backed answers grounded in your documents**.

Built with a focus on measurable retrieval quality: hybrid BM25 + Qdrant vector search fused via RRF, lexical reranking, structured citation validation, LLM-as-judge evaluation, and a CI regression gate.

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
- 💾 Persistent **Qdrant vector store** per collection (local embedded, auto-persisted)
- 🔍 **Hybrid retrieval**: BM25 keyword search + semantic vector search fused via **Reciprocal Rank Fusion (RRF)**
- 🎯 **Lexical reranking** — retrieved candidates re-scored by token-overlap combined with RRF score
- 📌 **Versioned prompt templates** with strict citation enforcement
- ✅ **Citation validation + guarded fallback** — citations are verified against retrieved source/page pairs before the answer is returned
- 🕵️ **Observability tracing** via [Langfuse](https://langfuse.com) — every request produces a structured trace with latency, scores, and prompt info; local JSONL sink available as fallback
- 🧪 **Evaluation Mode** (UI + CLI) using [DSPy](https://dspy.ai)-powered LLM-as-judge metrics (faithfulness, context relevance, answer completeness)
- 🔁 **CI regression gate** — GitHub Actions blocks PRs on metric degradation

---

## 🏗️ Architecture

```
Layer 1 — Ingestion:    PDFs → Parser → Chunker → Embedder → Qdrant + BM25 corpus
Layer 2 — Retrieval:    Query → BM25 + Vector → RRF Fusion → CrossEncoder Reranker → Top-K
Layer 3 — Generation:   Top-K chunks → Versioned prompt → Gemini LLM → Citation-validated answer
Layer 4 — Observability: Langfuse trace per request (latency, scores, model, prompt version)
Layer 5 — Evaluation:   Deterministic Recall@5 gate + DSPy LLM-as-judge CI gate
```

---

## 🧱 Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| LLM + Embeddings | Google Gemini (gemini-2.5-flash, embedding-001) |
| Vector Store | [Qdrant](https://qdrant.tech) (local embedded, persistent) |
| Keyword Search | Custom BM25 implementation (no extra deps) |
| LLM Orchestration | [DSPy](https://dspy.ai) (LLM-as-judge evaluation signatures) |
| Observability | [Langfuse](https://langfuse.com) + local JSONL fallback |
| Containers | Docker + Docker Compose |
| CI | GitHub Actions |
| Testing | pytest |

---

## 📁 Project Structure

```text
PDF-Corpus-Chat/
├── app/
│   └── main.py                    # Streamlit entry point
├── src/
│   ├── config.py                  # Chunking, retrieval, observability config
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   └── chunking.py
│   ├── rag/
│   │   ├── bm25_index.py          # Lightweight BM25 — persisted to disk, in-memory cache
│   │   ├── embeddings.py          # Gemini embedding factory
│   │   ├── fusion.py              # Reciprocal Rank Fusion (RRF)
│   │   ├── reranker.py            # ms-marco-MiniLM-L-6-v2 CrossEncoder (singleton)
│   │   ├── retriever.py           # Hybrid + standard retrieval
│   │   ├── vector_store.py        # Qdrant persistence helpers
│   │   ├── prompt_registry.py     # Versioned prompt templates
│   │   ├── prompting.py           # Context + prompt builder
│   │   └── validators.py          # Citation validation + guard fallback
│   ├── eval/
│   │   ├── evaluator.py           # Heuristic + LLM-judge evaluation
│   │   ├── metrics.py             # DSPy signatures: faithfulness, relevance, completeness
│   │   └── retrieval_metrics.py   # Recall@K, Precision@K, MRR, nDCG@K, P50/P95
│   ├── observability/
│   │   ├── tracing.py             # Structured trace record builder
│   │   └── store.py               # JSONL append sink
│   └── utils/
├── evals/
│   ├── golden_dataset.jsonl       # Golden Q&A pairs for CI evaluation
│   ├── fixtures/                  # Small PDFs for CI ingestion
│   └── reports/                   # Auto-generated eval reports
├── scripts/
│   ├── run_eval.py                # CLI batch evaluation runner
│   ├── benchmark_retrieval.py     # Retrieval strategy benchmark (Recall@K, MRR, nDCG)
│   ├── download_fixtures.py       # Downloads arXiv PDFs for CI fixture collection
│   ├── setup.ps1
│   └── run.ps1
├── tests/
│   ├── test_fusion.py
│   ├── test_validators.py
│   ├── test_retriever.py
│   ├── test_smoke.py
│   ├── test_reranker.py
│   └── test_retrieval_metrics.py
├── evals/
│   ├── golden_dataset.jsonl       # 50 verified Q&A pairs (6 categories)
│   ├── baselines/latest.json      # Latest retrieval benchmark results
│   ├── fixtures/                  # PDF fixtures for CI ingestion
│   └── reports/                   # Auto-generated eval reports
├── .github/workflows/
│   └── eval-gate.yml              # CI regression gate
├── conftest.py
├── requirements.txt
└── docker-compose.yml
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

The CLI runner evaluates the **golden dataset** against a live collection using DSPy LLM-as-judge metrics:

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
    "avg_composite_score": 0.89
  },
  "results": [...]
}
```

### Extend the Golden Dataset

Add records to `evals/golden_dataset.jsonl` in JSONL format:
```json
{"question": "What is X?", "expected_answer": "X is ...", "expected_sources": ["doc.pdf"], "tags": ["fact_lookup"]}
```

### Run Retrieval Benchmark

Compare four strategies (similarity / mmr / bm25 / hybrid) on Recall@5, MRR, nDCG@5, and latency:

```bash
python scripts/benchmark_retrieval.py \
  --collection my_collection \
  --dataset evals/golden_dataset.jsonl \
  --output evals/baselines/latest.json \
  --top-k 5
```

Sample output (values from CI run on arXiv fixture PDFs):

| Strategy       | Recall@5 | Precision@5 | MRR    | nDCG@5  | P50 (s) | P95 (s) |
|----------------|----------|-------------|--------|---------|---------|---------|
| similarity     | —        | —           | —      | —       | —       | —       |
| mmr            | —        | —           | —      | —       | —       | —       |
| bm25           | —        | —           | —      | —       | —       | —       |
| hybrid         | —        | —           | —      | —       | —       | —       |

> Numbers will be filled in on the first CI run with fixture PDFs. The baseline is saved to
> `evals/baselines/latest.json` and tracked in version control.

**CI gate:** Recall@5 of the best strategy must be ≥ 0.50 or the workflow fails.

---

## 🔁 CI Regression Gate

The GitHub Actions workflow (`.github/workflows/eval-gate.yml`) runs on every PR:

1. **Unit tests** — `tests/test_fusion.py`, `tests/test_validators.py`, `tests/test_retriever.py`, `tests/test_smoke.py`
2. **Eval gate** — if `evals/fixtures/*.pdf` exists, ingests and evaluates against `evals/golden_dataset.jsonl`
3. **Artifacts** — eval report uploaded to GitHub for post-merge debugging

**PR is blocked if:**
- Any unit test fails
- `avg_faithfulness < 0.85`
- `avg_citation_validity_ratio < 0.65`
- Best `Recall@5 < 0.50` (deterministic, no LLM needed)

To add fixture PDFs for CI:
```
evals/fixtures/sample_doc.pdf   # small representative PDF (< 5MB)
```

Set `GOOGLE_API_KEY` in your GitHub repository secrets.

---

## 🔐 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ | Google Gemini API key |
| `LANGFUSE_PUBLIC_KEY` | optional | Langfuse project public key |
| `LANGFUSE_SECRET_KEY` | optional | Langfuse project secret key |
| `LANGFUSE_HOST` | optional | Langfuse host (default: cloud.langfuse.com) |
| `OBSERVABILITY_ENABLED` | optional | `true` to enable JSONL tracing fallback |
| `OBSERVABILITY_TRACES_PATH` | optional | Override default trace path |
| `EVAL_JUDGE_MODEL` | optional | Override LLM judge model (default: gemini-2.5-flash) |
| `LLM_MODEL_NAME` | optional | LLM model to use for generation (default: gemini-2.5-flash) |
| `COST_PER_MILLION` | optional | Approximate USD cost per million tokens (default: 0.15) |

> ⚠️ **Never commit `.env`** — it is in `.gitignore`.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.