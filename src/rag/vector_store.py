import os
from typing import List, Dict, Any, Optional

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.rag.embeddings import get_embeddings  # centralised embedding factory


def sanitize_collection_name(name: str) -> str:
    # Replace spaces with underscores and remove weird characters
    safe = name.strip().replace(" ", "_")
    return safe if safe else "default"


def get_collection_dir(collection_name: str) -> str:
    # Kept for compatibility but we use a single Qdrant DB path for all collections
    safe_name = sanitize_collection_name(collection_name)
    return f"vector_stores/{safe_name}"

QDRANT_PATH = "vector_stores/qdrant_db"


def save_vector_store(vector_store: Any, collection_name: str):
    # Qdrant persists automatically if path is provided, so this is a no-op
    pass


def load_vector_store(collection_name: str, embeddings) -> Optional[QdrantVectorStore]:
    safe_name = sanitize_collection_name(collection_name)
    
    if not os.path.exists(QDRANT_PATH):
        return None

    client = QdrantClient(path=QDRANT_PATH)
    
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

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    os.makedirs(QDRANT_PATH, exist_ok=True)

    return QdrantVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        path=QDRANT_PATH,
        collection_name=safe_name,
    )
