import os
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def save_vector_store(vector_store: FAISS, collection_name: str):
    collection_dir = f"vector_stores/{collection_name}"
    os.makedirs(collection_dir, exist_ok=True)
    vector_store.save_local(collection_dir)


def load_vector_store(collection_name: str, embeddings) -> Optional[FAISS]:
    collection_dir = f"vector_stores/{collection_name}"
    if not os.path.exists(collection_dir):
        return None

    return FAISS.load_local(
        collection_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_vector_store_from_chunks(chunks: List[Dict[str, Any]]) -> FAISS:
    embeddings = get_embeddings()

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
