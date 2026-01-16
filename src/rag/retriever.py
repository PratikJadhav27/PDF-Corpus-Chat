from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from src.config import RETRIEVAL


def retrieve_docs(db: FAISS, query: str, k: int = None) -> List[Document]:
    """
    Retrieve top-k relevant chunks.
    """
    top_k = k if k is not None else RETRIEVAL.top_k
    return db.similarity_search(query, k=top_k)
