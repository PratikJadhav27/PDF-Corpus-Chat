"""
Embedding model factory.

Centralises provider selection so the rest of the codebase
never hard-codes a specific embedding class.

Supported providers (set EMBEDDING_PROVIDER in .env):
  - google   (default) → GoogleGenerativeAIEmbeddings
  - openai             → OpenAIEmbeddings
  - anthropic          → Not natively supported for embeddings;
                         falls back to OpenAI-compatible endpoint via openai sdk
"""

from __future__ import annotations

import os
from typing import Any


def get_embeddings() -> Any:
    """
    Return a LangChain-compatible embedding object based on the
    EMBEDDING_PROVIDER environment variable.

    Supported values: "google" (default), "openai"
    """
    provider = os.environ.get("EMBEDDING_PROVIDER", "google").lower().strip()

    if provider == "openai":
        return _get_openai_embeddings()

    # Default: Google Gemini embeddings
    return _get_google_embeddings()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_google_embeddings() -> Any:
    """
    Google Generative AI embeddings.

    Requires:
      GOOGLE_API_KEY set in environment.

    Tries to auto-discover the first available embedding model from the
    Gemini API; falls back to 'models/text-embedding-004'.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        try:
            import urllib.request, json

            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            with urllib.request.urlopen(urllib.request.Request(url)) as resp:
                data = json.loads(resp.read())

            for model in data.get("models", []):
                if "embedContent" in model.get("supportedGenerationMethods", []):
                    return GoogleGenerativeAIEmbeddings(model=model["name"])
        except Exception as exc:
            print(f"[embeddings] Dynamic model discovery failed: {exc}. Using default.")

    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


def _get_openai_embeddings() -> Any:
    """
    OpenAI embeddings.

    Requires:
      OPENAI_API_KEY set in environment.
      OPENAI_EMBEDDING_MODEL (optional, default: text-embedding-3-small)
    """
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for OpenAI embeddings. "
            "Install it with: pip install langchain-openai"
        ) from exc

    model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)
