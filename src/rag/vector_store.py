import os
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def sanitize_collection_name(name: str) -> str:
    # Replace spaces with underscores and remove weird characters
    safe = name.strip().replace(" ", "_")
    return safe if safe else "default"


def get_embeddings():
    import urllib.request
    import json
    import os
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read())
            
            for model in data.get("models", []):
                if "embedContent" in model.get("supportedGenerationMethods", []):
                    # Found an available embedding model
                    return GoogleGenerativeAIEmbeddings(model=model["name"])
        except Exception as e:
            print(f"Failed to dynamically fetch models: {e}")
            
    # Fallback default if API check fails or key is missing
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


def get_collection_dir(collection_name: str) -> str:
    safe_name = sanitize_collection_name(collection_name)
    return f"vector_stores/{safe_name}"


def save_vector_store(vector_store: FAISS, collection_name: str):
    collection_dir = get_collection_dir(collection_name)
    os.makedirs(collection_dir, exist_ok=True)
    vector_store.save_local(collection_dir)


def load_vector_store(collection_name: str, embeddings) -> Optional[FAISS]:
    collection_dir = get_collection_dir(collection_name)

    index_path = os.path.join(collection_dir, "index.faiss")
    store_path = os.path.join(collection_dir, "index.pkl")

    # If FAISS files don't exist, treat as empty collection
    if not (os.path.exists(index_path) and os.path.exists(store_path)):
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
