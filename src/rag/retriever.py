from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config import RETRIEVAL


def retrieve_docs(
    db: FAISS,
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
