# Multi-PDF ChatApp (RAG Pipeline) - Interview Preparation Guide

This document is a comprehensive end-to-end summary of the Multi-PDF ChatApp project. It is structured to help you discuss the architecture, technical decisions, and advanced features you've built during technical interviews.

## 1. High-Level System Overview

**Project**: A production-grade Retrieval-Augmented Generation (RAG) system that allows users to upload multiple PDFs, store them in isolated collections, and chat with the documents.
**Core Value**: Provides grounded, citation-backed answers with strict hallucination guards, observability, and automated evaluation to ensure response quality.

### Tech Stack

- **Core Framework**: LangChain + Custom implementations.
- **Frontend**: Streamlit (with pending redesign mockups via Stitch/React).
- **LLM & Embeddings**: Google Gemini 2.5 Flash & `embedding-001`.
- **Vector Database**: FAISS (Facebook AI Similarity Search) - Local persistent storage.
- **CI/CD**: GitHub Actions (Regression Evaluation Gates).
- **Testing**: Pytest for unit tests.

---

## 2. Ingestion Pipeline

**Goal**: Convert raw PDFs into searchable semantic and lexical indexes.

- **Document Parsing**: Uses a custom loader (e.g., PyMuPDF/PyPDF2) to perform safe page extraction with telemetry (tracking `total_pages`, `extracted_pages`, `empty_pages`, and `failed_pages`).
- **Chunking**: Employs `RecursiveCharacterTextSplitter`.
  - Configured with `chunk_size = 1500` and `chunk_overlap = 200`.
  - Stores metadata containing document origin, page number, and chunk index to accurately trace back answers to original sources.
- **Index Generation**:
  - **Dense Index**: Generates dense embeddings using `GoogleGenerativeAIEmbeddings` and stores them in a collection-specific FAISS index.
  - **Sparse Index**: Maintains a secondary BM25 keyword index for exact-keyword lookup.
  - Generates `metadata.json` for each collection to track ingestion statistics.

---

## 3. Advanced Retrieval Architecture

**Goal**: Retrieve the most semantically relevant and lexically accurate chunks for a query.

- **Multi-Mode Retrieval**: Supports `similarity` (vector search), `mmr` (Maximal Marginal Relevance to ensure diversity), and `hybrid`.
- **Hybrid Retrieval Implementation**:
  - **1. Dual Querying**: Queries both the FAISS (Dense) and BM25 (Sparse) indexes simultaneously.
  - **2. Reciprocal Rank Fusion (RRF)**: A custom algorithmic implementation that takes the ranking lists from both BM25 and Vector Search and computes a unified score to balance semantic matching with exact-term matching.
  - **3. Re-ranking**: Passes the combined RRF candidates through a cross-encoder style re-ranker (lexical overlap or lightweight LLM check) to determine the absolute final `Top-K` chunks.

---

## 4. Generation & Prompting

**Goal**: Instruct the LLM to generate high-quality answers strictly grounded in the retrieved context.

- **Prompt Registry**: System uses a version-controlled prompt registry. Different conversational styles ("standard", "detailed", "summarize") are easily maintained and toggled.
- **Grounded Prompt Building**: The system constructs the prompt by injecting the retrieved chunks along with their citations (`Source X | Page Y`). The LLM is strictly instructed: _"If the answer is not in the provided context, say 'answer is not available'."_

---

## 5. Defense & Validation Mechanisms

**Goal**: Catch and prevent AI hallucinations or fake citations.

- **Citation Validity Checking**: Custom validation logic (`guard_answer_with_citation_check` in `validators.py`) parses the LLM's response specifically looking for citation tags.
- It verifies that the cited source was actually present in the provided context. If the response contains completely hallucinated citations, the fallback kicks in, or the validation metric ratio dips.

---

## 6. Observability

**Goal**: Trace every query in production to debug and monitor latency and costs.

- **JSONL Tracing**: Instead of blind logs, every request creates a structured Trace Record (ready for ingestion into platforms like LangSmith or Datadog).
- **Trace Contents**:
  - `trace_id` & `phase` (ingestion vs. chat)
  - `retrieval_mode`, `prompt_version`, `temperature`
  - Granular `latency_sec` measuring time taken for: `retrieval_sec`, `llm_sec`, and `total_sec`.
  - Final calculated `citation_validity_ratio`.

---

## 7. Automated Evaluation & CI/CD Pipeline

**Goal**: Prevent degraded LLM configurations from deploying to production.

- **Golden Dataset**: A `.jsonl` file containing pairs of questions and known expected answers/sources.
- **LLM-as-a-Judge (`evals/evaluator.py`)**: Uses an LLM to assess generated answers against the golden dataset. It calculates:
  1. **Faithfulness**: Is the answer derived _only_ from the retrieved context?
  2. **Citation Validity**: Do the citations in the response correctly map to actual retrieved documents?
- **CI / Regression Gate**: `eval-gate.yml` runs on every GitHub Pull Request.
  - Runs normal unit tests (e.g., testing RRF math, validator regex).
  - Performs a test ingestion and evaluates the golden dataset.
  - **Blocks Merges** if `avg_faithfulness < 0.85` or `citation_validity_ratio < 0.90`.

---

## Interview Talking Points & "Why" Questions

**Q: Why use FAISS locally instead of a managed vector DB like Pinecone?**
_A: For cost-efficiency, data privacy (documents remain entirely on local hardware), and simplicity since we partition indexes dynamically per "Collection". Local FAISS allows rapid file-based separation for topic-specific contexts._

**Q: Why build Hybrid Search (Dense + BM25) and not just Dense Vector Search?**
_A: Dense embeddings are great at semantics but often fail on exact keyword or serial number lookups. By fusing them using Reciprocal Rank Fusion (RRF), we get the best of both worlds—understanding concepts while never losing exact matches._

**Q: How do you handle LLM Hallucinations?**
_A: Three-tier defense: First, strict grounding prompts. Second, LLM parameter tuning (`Temperature = 0.3`). Third, an explicit post-generation Regex validator (`validators.py`) that strictly checks if the citation strings generated by the LLM match the IDs/filenames of the retrieved context chunks._

**Q: How do you ensure prompt or chunking logic changes don't break the system?**
_A: I implemented a CI/CD regression gate. On every PR, a golden dataset is evaluated using an LLM-as-a-Judge pattern to quantify changes in "Faithfulness" and "Citation Validity". If the scores dip below 85% and 90% respectively, the build fails._
