import os
from typing import Dict, List, Any, Optional

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from src.rag.embeddings import get_embeddings  # centralised embedding factory
from src.rag.collection_names import (  # single source of truth (ISSUE-010)
    VECTOR_STORES_DIR,
    collection_path,
    resolve_collection_name,
    sanitize_collection_name,  # re-exported: existing callers import it from here
)


# Base directory for vector stores. Each collection gets its own sub‑directory.
def get_collection_dir(collection_name: str) -> str:
    """Return the path for a collection's Qdrant storage.

    The name is sanitised to a single ``[A-Za-z0-9_-]`` component (an existing
    legacy directory keeps its name), so the result is always inside
    ``vector_stores`` — see ``src.rag.collection_names``.
    """
    return collection_path(VECTOR_STORES_DIR, collection_name)

# Legacy constant kept for backward compatibility – points to the default collection.
QDRANT_PATH = get_collection_dir("default")


# ---------------------------------------------------------------------------
# Process-wide client cache (ISSUE-011)
#
# Local-mode Qdrant opens the storage folder and takes an EXCLUSIVE lock on it,
# released only when the client is closed. Opening a fresh client per query was
# a fixed latency tax and a latent `RuntimeError: Storage folder ... already
# accessed` whenever two parts of a run overlapped. One client per collection
# path is shared for the life of the process and closed explicitly by
# `evict_qdrant_client()` (called when a collection is deleted).
# ---------------------------------------------------------------------------

_CLIENTS: Dict[str, QdrantClient] = {}


def _client_key(collection_name: str) -> str:
    return os.path.abspath(get_collection_dir(collection_name))


def get_qdrant_client(collection_name: str) -> QdrantClient:
    """Return the shared client for a collection's storage folder."""
    key = _client_key(collection_name)
    client = _CLIENTS.get(key)
    if client is None:
        client = QdrantClient(path=key)
        _CLIENTS[key] = client
    return client


def evict_qdrant_client(collection_name: str) -> None:
    """Close and forget the cached client for a collection.

    Required before deleting the storage folder: on Windows the open SQLite
    file cannot be removed while a client still holds it.
    """
    client = _CLIENTS.pop(_client_key(collection_name), None)
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


def close_all_qdrant_clients() -> None:
    """Close every cached client (test isolation / process shutdown)."""
    for key in list(_CLIENTS):
        client = _CLIENTS.pop(key, None)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def load_vector_store(collection_name: str, embeddings) -> Optional[QdrantVectorStore]:
    safe_name = resolve_collection_name(collection_name)
    collection_dir = get_collection_dir(safe_name)

    if not os.path.exists(collection_dir):
        return None

    client = get_qdrant_client(safe_name)

    try:
        if not client.collection_exists(safe_name):
            return None
    except Exception:
        return None

    return QdrantVectorStore(
        client=client,
        collection_name=safe_name,
        embedding=embeddings,
    )


def build_vector_store_from_chunks(
    chunks: List[Dict[str, Any]],
    collection_name: str = "default",
    embeddings: Any = None,
) -> QdrantVectorStore:
    if embeddings is None:
        embeddings = get_embeddings()
    safe_name = resolve_collection_name(collection_name)
    collection_dir = get_collection_dir(safe_name)

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    os.makedirs(collection_dir, exist_ok=True)

    # Create through the shared client rather than `from_texts(path=...)`, which
    # would open a second local client on the same folder and keep its exclusive
    # lock alive as long as the returned store object lives.
    client = get_qdrant_client(safe_name)
    if not client.collection_exists(safe_name):
        probe_text = texts[0] if texts else "dummy_text"
        dimension = len(embeddings.embed_documents([probe_text])[0])
        client.create_collection(
            collection_name=safe_name,
            vectors_config={
                QdrantVectorStore.VECTOR_NAME: models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                )
            },
        )

    store = QdrantVectorStore(
        client=client,
        collection_name=safe_name,
        embedding=embeddings,
    )
    if texts:
        store.add_texts(texts=texts, metadatas=metadatas)
    return store


# ---------------------------------------------------------------------------
# Per-source point management (ISSUE-001)
#
# Re-ingesting a PDF must REPLACE its previous chunks on the vector side so
# Qdrant stays consistent with the BM25/JSONL corpus (which is rewritten with
# the same per-source semantics by src.rag.bm25_index.merge_chunk_corpus).
#
# Local-mode Qdrant takes an EXCLUSIVE flock on the collection directory and
# raises RuntimeError when a second client opens the same path, so every
# helper below uses exactly one client and closes it before returning.
# ---------------------------------------------------------------------------

def _source_filter(sources) -> models.Filter:
    """Payload filter matching points whose metadata.source is in ``sources``."""
    source_list = sorted({str(s).strip() for s in sources if str(s).strip()})
    return models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.source",
                match=models.MatchAny(any=source_list),
            )
        ]
    )


def delete_points_for_sources(
    client: QdrantClient,
    collection_name: str,
    sources,
) -> int:
    """Delete every point whose ``metadata.source`` is in ``sources``.

    Returns the number of points that were matched (and deleted). Chunks
    without a ``source`` payload are never matched, so legacy data survives.
    """
    source_list = [s for s in {str(x).strip() for x in sources} if s]
    if not source_list:
        return 0

    safe_name = resolve_collection_name(collection_name)
    flt = _source_filter(source_list)

    try:
        deleted = int(client.count(collection_name=safe_name, count_filter=flt).count)
    except Exception:
        deleted = 0

    client.delete(collection_name=safe_name, points_selector=models.FilterSelector(filter=flt))
    return deleted


def upsert_chunks(
    collection_name: str,
    chunks: List[Dict[str, Any]],
    embeddings: Any = None,
) -> str:
    """Add chunks to a collection, replacing earlier chunks from the same source files.

    Guarantees:
    - First ingestion creates the collection (via ``build_vector_store_from_chunks``).
    - Re-ingestion deletes the replaced sources' existing points with a payload
      filter and then adds ONLY the new chunks (retained files are never
      re-embedded, so no embedding API cost for unchanged documents).
    - A single QdrantClient is used for delete + add and is always closed.

    Returns ``"created"`` for a fresh collection or ``"appended"`` when an
    existing collection was updated.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    safe_name = resolve_collection_name(collection_name)
    collection_dir = get_collection_dir(safe_name)
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    sources = {c.get("metadata", {}).get("source") for c in chunks}

    # Fast path: no collection directory yet -> fresh build (creates dir + collection).
    if not os.path.exists(collection_dir):
        build_vector_store_from_chunks(chunks, collection_name, embeddings)
        return "created"

    client = get_qdrant_client(safe_name)
    try:
        exists = client.collection_exists(safe_name)
    except Exception:
        exists = False

    if not exists:
        build_vector_store_from_chunks(chunks, collection_name, embeddings)
        return "created"

    delete_points_for_sources(client, safe_name, sources)
    store = QdrantVectorStore(
        client=client,
        collection_name=safe_name,
        embedding=embeddings,
    )
    if texts:
        store.add_texts(texts=texts, metadatas=metadatas)

    return "appended"
