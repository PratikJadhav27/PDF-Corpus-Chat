"""
app/helpers.py

Shared, stateless helpers used across multiple pages.
No Streamlit rendering calls — pure logic only.

Ingestion semantics (ISSUE-001)
--------------------------------
Chunks belong to a *source file*. Re-ingesting a PDF replaces that file's
chunks on BOTH indexes (Qdrant points and the BM25/JSONL corpus); chunks from
other files are preserved, and metadata.json is merged per file. This keeps
the two indexes consistent no matter how many upload batches a user performs.
"""
import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.config import CHUNKING
from src.ingestion.pdf_loader import load_pdf_pages_with_stats
from src.ingestion.chunking import chunk_pdf_pages
from src.rag.collection_names import (
    VECTOR_STORES_DIR,
    collection_path,
    contained_path,
    resolve_collection_name,
)
from src.rag.vector_store import evict_qdrant_client, upsert_chunks
from src.rag.bm25_index import forget_corpus_cache, merge_chunk_corpus

import streamlit as st  # only for st.session_state.document_metadata write


METADATA_DIR = "metadata"


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def get_collections() -> List[str]:
    """Return a list of existing collection names."""
    if not os.path.exists(VECTOR_STORES_DIR):
        return []
    return [
        d
        for d in os.listdir(VECTOR_STORES_DIR)
        if os.path.isdir(os.path.join(VECTOR_STORES_DIR, d))
    ]


def create_collection(collection_name: str) -> str:
    """Create a collection's store and metadata directories.

    Returns the name that was actually used: it is sanitised to
    ``[A-Za-z0-9_-]`` (or kept as-is when a legacy directory already exists),
    so a typed name like ``../etc`` can only ever create ``etc`` inside
    ``vector_stores`` (ISSUE-010).
    """
    safe = resolve_collection_name(collection_name)
    for base in (VECTOR_STORES_DIR, METADATA_DIR):
        os.makedirs(contained_path(base, safe), exist_ok=True)
    return safe


def delete_collection(collection_name: str) -> bool:
    """Delete a collection's vector store and metadata directories.

    The name is resolved to a contained path (ISSUE-010), so a name with path
    separators can never remove anything outside the two collection trees.
    """
    safe = resolve_collection_name(collection_name)
    # Close the shared Qdrant client first: a local-mode client keeps the
    # storage SQLite file open, which blocks removal on Windows.
    evict_qdrant_client(safe)
    forget_corpus_cache(safe)
    for base in (VECTOR_STORES_DIR, METADATA_DIR):
        path = contained_path(base, safe)
        if os.path.exists(path):
            shutil.rmtree(path)
    return True


def load_collection_metadata(collection_name: str) -> Optional[Dict[str, Any]]:
    """Load persisted metadata.json for a collection, or None if missing."""
    path = os.path.join(collection_path(METADATA_DIR, collection_name), "metadata.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def estimate_cost(tokens_used: int) -> float:
    """Estimate cost in USD for a given token count.

    The per‑million‑token rate is read from the ``COST_PER_MILLION``
    environment variable (default 0.15 USD) to allow easy tuning per
    model/provider.
    """
    cost_per_million = float(os.getenv("COST_PER_MILLION", "0.15"))
    return (tokens_used / 1_000_000) * cost_per_million


# ---------------------------------------------------------------------------
# Metadata merge helpers (per-file identity)
# ---------------------------------------------------------------------------

def _merge_file_entries(
    previous: List[Dict[str, Any]],
    current: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge per-file extraction-stat entries.

    Entries with the same ``file_name`` are replaced by the newer entry while
    preserving first-seen ordering, so re-uploading a document refreshes its
    stats without losing other files' rows.
    """
    by_name: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for entry in list(previous or []) + list(current or []):
        name = entry.get("file_name") or entry.get("name")
        if not name:
            continue
        if name not in by_name:
            order.append(name)
        by_name[name] = entry
    return [by_name[n] for n in order]


def _merge_totals(per_file: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Recompute overall extraction totals from merged per-file entries."""
    total = sum(f.get("total_pages", 0) for f in per_file)
    extracted = sum(f.get("extracted_pages", 0) for f in per_file)
    empty = sum(f.get("empty_pages", 0) for f in per_file)
    failed = sum(f.get("failed_pages", 0) for f in per_file)
    return {
        "total_pages": total,
        "extracted_pages": extracted,
        "empty_pages": empty,
        "failed_pages": failed,
        "success_rate_percent": round((extracted / total) * 100, 2)
        if total > 0
        else 0.0,
    }


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_chunks_with_stats(
    chunks: List[Dict[str, Any]],
    extraction_stats: List[Dict[str, Any]],
    collection_name: str,
    embeddings: Any = None,
) -> Dict[str, Any]:
    """Index chunks into a collection, keeping every index in sync.

    - Vector store: replaced sources' points are deleted, new chunks added
      (only new chunks are embedded — see ``upsert_chunks``).
    - BM25 + JSONL corpus: same per-source replace semantics (``merge_chunk_corpus``).
    - metadata.json: per-file stats merged, totals recomputed.

    Returns ``{"ok", "store_mode", "metadata", "chunks_ingested"}``.
    """
    safe_collection = resolve_collection_name(collection_name)

    # 1. Vector store (Qdrant) — per-source replace + append
    store_mode = upsert_chunks(safe_collection, chunks, embeddings)

    # 2. BM25 index + JSONL corpus — same per-source semantics
    merged_rows = merge_chunk_corpus(safe_collection, chunks)

    # 3. metadata.json — merge per file, recompute totals
    previous = load_collection_metadata(safe_collection) or {}
    per_file = _merge_file_entries(
        (previous.get("extraction_stats") or {}).get("per_file", []),
        extraction_stats,
    )
    overall = _merge_totals(per_file)

    metadata = {
        "files": [f.get("file_name", "?") for f in per_file],
        "total_pages": overall["total_pages"],
        "total_chunks": len(merged_rows),
        "chunk_size": CHUNKING.chunk_size,
        "chunk_overlap": CHUNKING.chunk_overlap,
        "processed_at": datetime.now().isoformat(),
        "collection_name": safe_collection,
        "extraction_stats": {
            "overall": overall,
            "per_file": per_file,
        },
    }

    meta_dir = contained_path(METADATA_DIR, safe_collection)
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    st.session_state.document_metadata = metadata

    return {
        "ok": True,
        "store_mode": store_mode,
        "metadata": metadata,
        "chunks_ingested": len(chunks),
    }


def process_and_index_pdfs(pdf_docs, collection_name: str) -> Dict[str, Any]:
    """
    Full ingestion pipeline:
      - Load pages from each PDF
      - Chunk with page-aware metadata
      - Delegate to ingest_chunks_with_stats() (Qdrant + BM25 + metadata.json,
        all with per-file replace/append semantics)

    Returns a result dict with keys: ok, message, stats.
    """
    all_pages: List[Dict[str, Any]] = []
    extraction_stats: List[Dict[str, Any]] = []

    total_pages_all = extracted_pages_all = empty_pages_all = failed_pages_all = 0

    for pdf in pdf_docs:
        pages, stats = load_pdf_pages_with_stats(pdf)
        all_pages.extend(pages)
        extraction_stats.append(stats)
        total_pages_all += stats["total_pages"]
        extracted_pages_all += stats["extracted_pages"]
        empty_pages_all += stats["empty_pages"]
        failed_pages_all += stats["failed_pages"]

    chunks = chunk_pdf_pages(all_pages)

    if not chunks:
        return {
            "ok": False,
            "message": (
                "No usable text extracted from PDFs. "
                "These may be scanned/image-based or corrupted PDFs."
            ),
            "stats": {
                "total_pages": total_pages_all,
                "extracted_pages": extracted_pages_all,
                "empty_pages": empty_pages_all,
                "failed_pages": failed_pages_all,
                "files": extraction_stats,
            },
        }

    ingestion = ingest_chunks_with_stats(
        chunks=chunks,
        extraction_stats=extraction_stats,
        collection_name=collection_name,
    )

    return {
        "ok": True,
        "message": f"Indexed {len(pdf_docs)} PDF(s) successfully.",
        "stats": ingestion["metadata"]["extraction_stats"],
    }
