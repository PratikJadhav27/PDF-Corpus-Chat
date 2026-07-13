from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langfuse import observe

from src.config import RETRIEVAL
from src.rag.bm25_index import bm25_search_collection
from src.rag.fusion import rrf_fuse
from src.rag.reranker import rerank_candidates


@observe(as_type="retrieval")
def retrieve_docs(
    db: QdrantVectorStore,
    query: str,
    mode: Optional[str] = None,
    k: Optional[int] = None,
) -> List[Document]:
    """
    Retrieve documents using similarity or MMR.
    """
    retrieval_mode = (mode or RETRIEVAL.mode).lower()
    top_k = k if k is not None else RETRIEVAL.top_k

    if retrieval_mode == "mmr":
        # MMR gives diverse chunks (less repetition)
        return db.max_marginal_relevance_search(query, k=top_k, fetch_k=max(10, top_k * 3))

    # Default similarity search
    return db.similarity_search(query, k=top_k)


def _vector_hits_with_scores(
    db: QdrantVectorStore,
    query: str,
    k: int,
) -> List[Tuple[Dict[str, Any], float]]:
    hits = db.similarity_search_with_score(query, k=k)
    rows: List[Tuple[Dict[str, Any], float]] = []
    for doc, score in hits:
        rows.append(
            (
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                },
                float(score),
            )
        )
    return rows


@observe(as_type="retrieval")
def retrieve_hybrid_reranked(
    db: QdrantVectorStore,
    collection_name: str,
    query: str,
    top_k: int = 4,
    candidate_k: int = 12,
) -> List[Document]:
    """
    Hybrid retrieval pipeline:
    vector (with score) + BM25 -> RRF fuse/dedupe -> rerank -> top_k Documents
    """
    vector_hits = _vector_hits_with_scores(db, query, k=candidate_k)
    bm25_hits = bm25_search_collection(collection_name, query, k=candidate_k)

    fused = rrf_fuse(vector_hits=vector_hits, bm25_hits=bm25_hits)
    reranked = rerank_candidates(query=query, candidates=fused, top_n=top_k)

    docs: List[Document] = []
    for row in reranked:
        md = dict(row.get("metadata", {}))
        md["retrieval"] = {
            "rrf_score": row.get("rrf_score"),
            "rerank_score": row.get("rerank_score"),
            "overlap_score": row.get("overlap_score"),
            "vector_score": row.get("vector_score"),
            "bm25_score": row.get("bm25_score"),
            "channels": row.get("channels", []),
        }
        docs.append(Document(page_content=row.get("text", ""), metadata=md))
    return docs
