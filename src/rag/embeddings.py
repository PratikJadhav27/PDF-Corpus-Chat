"""
Embedding model factory.

Centralises provider selection so the rest of the codebase
never hard-codes a specific embedding class.

Supported providers (set EMBEDDING_PROVIDER in .env):
  - google   (default) → GoogleGenerativeAIEmbeddings
  - openai             → OpenAIEmbeddings
  - anthropic          → Not natively supported for embeddings;
                         falls back to OpenAI-compatible endpoint via openai sdk

The factory is cached (ISSUE-011): building the Google embeddings object used
to perform a live HTTPS model-discovery call on every invocation — once per
chat query, per semantic search and per ingestion. The discovery result and
the constructed object are now kept for the process lifetime.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


# provider+model key -> embeddings instance
_EMBEDDINGS_CACHE: Dict[str, Any] = {}

# Sentinel for "discovery not attempted yet"; None means "attempted, no model".
_UNSET = object()
_discovered_google_model: Any = _UNSET


def get_embeddings(force_refresh: bool = False) -> Any:
    """
    Return a LangChain-compatible embedding object based on the
    EMBEDDING_PROVIDER environment variable.

    Supported values: "google" (default), "openai"

    The instance is cached per provider/model, because constructing it may
    involve a network round-trip and every query needs the same model anyway.
    """
    provider = os.environ.get("EMBEDDING_PROVIDER", "google").lower().strip()
    key = f"{provider}:{_model_override(provider)}"

    if not force_refresh:
        cached = _EMBEDDINGS_CACHE.get(key)
        if cached is not None:
            return cached

    if provider == "openai":
        embeddings = _get_openai_embeddings()
    else:
        embeddings = _get_google_embeddings()

    _EMBEDDINGS_CACHE[key] = embeddings
    return embeddings


def clear_embeddings_cache() -> None:
    """Forget cached instances and the discovery result (tests, config changes)."""
    global _discovered_google_model
    _EMBEDDINGS_CACHE.clear()
    _discovered_google_model = _UNSET


def _model_override(provider: str) -> str:
    if provider == "openai":
        return os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return os.environ.get("GOOGLE_EMBEDDING_MODEL", "")


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

    model = _discover_google_model()
    return GoogleGenerativeAIEmbeddings(model=model)


def _discover_google_model() -> str:
    """Resolve the Google embedding model name once per process.

    Prefers an explicit model, otherwise asks the API which models support
    ``embedContent`` and falls back to ``text-embedding-004``. The result —
    including a failed attempt — is memoised so queries do not repeat the
    request.
    """
    global _discovered_google_model

    pinned = os.environ.get("GOOGLE_EMBEDDING_MODEL")
    if pinned:
        return pinned

    if _discovered_google_model is not _UNSET:
        return _discovered_google_model or "models/text-embedding-004"

    _discovered_google_model = None  # "attempted" even if it fails below
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        try:
            import json
            import urllib.parse
            import urllib.request

            url = (
                "https://generativelanguage.googleapis.com/v1beta/models?"
                + urllib.parse.urlencode({"key": api_key})
            )
            with urllib.request.urlopen(urllib.request.Request(url), timeout=5) as resp:
                data = json.loads(resp.read())

            for model in data.get("models", []):
                if "embedContent" in model.get("supportedGenerationMethods", []):
                    _discovered_google_model = model["name"]
                    break
        except Exception as exc:
            print(f"[embeddings] Dynamic model discovery failed: {exc}. Using default.")

    return _discovered_google_model or "models/text-embedding-004"


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
