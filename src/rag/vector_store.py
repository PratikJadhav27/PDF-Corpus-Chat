import os
from typing import List, Dict, Any, Optional

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.rag.embeddings import get_embeddings  # centralised embedding factory


# Base directory for vector stores. Each collection gets its own sub‑directory.
def get_collection_dir(collection_name: str) -> str:
    """Return the absolute path for a collection's Qdrant storage.

    The function sanitises the collection name (spaces → underscores) and
    ensures a directory exists under ``vector_stores``.
    """
    safe = (collection_name or "default").strip().replace(" ", "_")
    return f"vector_stores/{safe}"

# Legacy constant kept for backward compatibility – points to the default collection.
QDRANT_PATH = get_collection_dir("default")


def sanitize_collection_name(name: str) -> str:
    """Replace spaces with underscores and strip unsafe characters."""
    safe = name.strip().replace(" ", "_")
    return safe if safe else "default"


def load_vector_store(collection_name: str, embeddings) -> Optional[QdrantVectorStore]:
    safe_name = sanitize_collection_name(collection_name)
    collection_dir = get_collection_dir(collection_name)

    if not os.path.exists(collection_dir):
        return None

    client = QdrantClient(path=collection_dir)

    try:
        collections = client.get_collections().collections
        if not any(c.name == safe_name for c in collections):
            return None
    except Exception:
        return None

    return QdrantVectorStore(
        client=client,
        collection_name=safe_name,
        embedding=embeddings,
    )


def build_vector_store_from_chunks(chunks: List[Dict[str, Any]], collection_name: str = "default") -> QdrantVectorStore:
    embeddings = get_embeddings()
    safe_name = sanitize_collection_name(collection_name)
    collection_dir = get_collection_dir(collection_name)

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    os.makedirs(collection_dir, exist_ok=True)

    return QdrantVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        path=collection_dir,
        collection_name=safe_name,
    )
