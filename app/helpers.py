"""
app/helpers.py

Shared, stateless helpers used across multiple pages.
No Streamlit calls — pure logic only.
"""
import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.config import CHUNKING
from src.ingestion.pdf_loader import load_pdf_pages_with_stats
from src.ingestion.chunking import chunk_pdf_pages
from src.rag.vector_store import build_vector_store_from_chunks
from src.rag.bm25_index import save_chunk_corpus

import streamlit as st  # only for st.session_state.document_metadata write


VECTOR_STORES_DIR = "vector_stores"
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


def delete_collection(collection_name: str) -> bool:
    """Delete a collection's vector store and metadata directories."""
    for base in (VECTOR_STORES_DIR, METADATA_DIR):
        path = os.path.join(base, collection_name)
        if os.path.exists(path):
            shutil.rmtree(path)
    return True


def load_collection_metadata(collection_name: str) -> Optional[Dict[str, Any]]:
    """Load persisted metadata.json for a collection, or None if missing."""
    path = os.path.join(METADATA_DIR, collection_name, "metadata.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def estimate_cost(tokens_used: int) -> float:
    """Rough Gemini cost estimate (USD) based on token count."""
    cost_per_million = 0.15
    return (tokens_used / 1_000_000) * cost_per_million


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def process_and_index_pdfs(pdf_docs, collection_name: str) -> Dict[str, Any]:
    """
    Full ingestion pipeline:
      - Load pages from each PDF
      - Chunk with page-aware metadata
      - Build and persist Qdrant vector store
      - Build and pickle BM25 index
      - Save metadata.json

    Returns a result dict with keys: ok, message, stats.
    """
    safe_collection = collection_name.strip().replace(" ", "_") or "default"

    all_pages: List[Dict[str, Any]] = []
    extraction_stats: List[Dict[str, Any]] = []
    files_info: List[Dict[str, Any]] = []

    total_pages_all = extracted_pages_all = empty_pages_all = failed_pages_all = 0

    for pdf in pdf_docs:
        pages, stats = load_pdf_pages_with_stats(pdf)
        all_pages.extend(pages)
        extraction_stats.append(stats)
        files_info.append({"name": pdf.name, "pages": stats["total_pages"]})
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

    build_vector_store_from_chunks(chunks, safe_collection)
    save_chunk_corpus(safe_collection, chunks)

    overall_success_rate = (
        round((extracted_pages_all / total_pages_all) * 100, 2)
        if total_pages_all > 0
        else 0.0
    )

    metadata = {
        "files": [f["name"] for f in files_info],
        "total_pages": total_pages_all,
        "total_chunks": len(chunks),
        "chunk_size": CHUNKING.chunk_size,
        "chunk_overlap": CHUNKING.chunk_overlap,
        "processed_at": datetime.now().isoformat(),
        "collection_name": safe_collection,
        "extraction_stats": {
            "overall": {
                "total_pages": total_pages_all,
                "extracted_pages": extracted_pages_all,
                "empty_pages": empty_pages_all,
                "failed_pages": failed_pages_all,
                "success_rate_percent": overall_success_rate,
            },
            "per_file": extraction_stats,
        },
    }

    meta_dir = os.path.join(METADATA_DIR, safe_collection)
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    st.session_state.document_metadata = metadata

    return {
        "ok": True,
        "message": f"Indexed {len(pdf_docs)} PDF(s) successfully.",
        "stats": metadata["extraction_stats"],
    }
