"""
BM25 index with persistence.

Key design decisions:
- The JSONL chunk corpus (chunks.jsonl) stores raw text + metadata and is
  written at ingestion time by save_chunk_corpus().
- The BM25 index (bm25_index.pkl) is built once at ingestion time and
  pickled alongside the corpus. At query time it is loaded from disk and
  cached in memory per collection — no re-tokenisation on every request.
- The in-memory cache (_INDEX_CACHE) is keyed by collection name. It is
  invalidated whenever save_bm25_index() is called for that collection,
  so a re-ingestion always forces a fresh load on the next query.
"""

import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# In-memory cache: collection_name -> SimpleBM25 instance
_INDEX_CACHE: Dict[str, "SimpleBM25"] = {}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _collection_dir(collection_name: str) -> str:
    safe = (collection_name or "default").strip().replace(" ", "_")
    return f"vector_stores/{safe}"


def get_chunk_corpus_path(collection_name: str) -> str:
    return os.path.join(_collection_dir(collection_name), "chunks.jsonl")


def _get_index_path(collection_name: str) -> str:
    return os.path.join(_collection_dir(collection_name), "bm25_index.pkl")


# ---------------------------------------------------------------------------
# Tokeniser (shared by SimpleBM25 and reranker)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


# ---------------------------------------------------------------------------
# BM25 implementation
# ---------------------------------------------------------------------------

@dataclass
class BM25DocScore:
    index: int
    score: float


class SimpleBM25:
    """Lightweight BM25 index — no extra runtime dependencies."""

    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens = [_tokenize(d) for d in docs]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))

        self.tf: List[Counter] = [Counter(toks) for toks in self.doc_tokens]
        self.df: Dict[str, int] = defaultdict(int)
        for toks in self.doc_tokens:
            for term in set(toks):
                self.df[term] += 1

        self.n_docs = len(docs)

    def _idf(self, term: str) -> float:
        n_qi = self.df.get(term, 0)
        return math.log(1 + ((self.n_docs - n_qi + 0.5) / (n_qi + 0.5)))

    def search(self, query: str, k: int = 10) -> List[BM25DocScore]:
        query_terms = _tokenize(query)
        if not query_terms or not self.docs:
            return []

        scored: List[BM25DocScore] = []
        for idx, tf_counter in enumerate(self.tf):
            dl = self.doc_lens[idx]
            score = 0.0
            for term in query_terms:
                f_qi_d = tf_counter.get(term, 0)
                if f_qi_d == 0:
                    continue
                idf = self._idf(term)
                denom = f_qi_d + self.k1 * (1 - self.b + self.b * (dl / max(1e-9, self.avgdl)))
                score += idf * ((f_qi_d * (self.k1 + 1)) / max(1e-9, denom))
            if score > 0.0:
                scored.append(BM25DocScore(index=idx, score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]


# ---------------------------------------------------------------------------
# Corpus persistence (JSONL)
# ---------------------------------------------------------------------------

def save_chunk_corpus(collection_name: str, chunks: List[Dict[str, Any]]) -> None:
    """Persist raw chunks to JSONL and build + pickle the BM25 index."""
    dir_path = _collection_dir(collection_name)
    os.makedirs(dir_path, exist_ok=True)

    # 1. Write JSONL corpus (text + metadata for post-hit lookup)
    corpus_path = get_chunk_corpus_path(collection_name)
    with open(corpus_path, "w", encoding="utf-8") as f:
        for c in chunks:
            row = {"text": c.get("text", ""), "metadata": c.get("metadata", {})}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 2. Build BM25 index from the corpus and persist it
    texts = [c.get("text", "") for c in chunks]
    save_bm25_index(collection_name, texts)


def load_chunk_corpus(collection_name: str) -> List[Dict[str, Any]]:
    """Load raw chunks from the JSONL corpus file."""
    path = get_chunk_corpus_path(collection_name)
    if not os.path.exists(path):
        return []

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ---------------------------------------------------------------------------
# BM25 index persistence (pickle)
# ---------------------------------------------------------------------------

def save_bm25_index(collection_name: str, texts: List[str]) -> None:
    """Build a SimpleBM25 from texts, pickle it, and invalidate the cache."""
    index = SimpleBM25(texts)
    index_path = _get_index_path(collection_name)
    os.makedirs(_collection_dir(collection_name), exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump(index, f)
    # Invalidate in-memory cache so the next query loads the fresh index
    _INDEX_CACHE.pop(collection_name, None)


def load_bm25_index(collection_name: str) -> Optional[SimpleBM25]:
    """
    Load BM25 index for a collection.

    Lookup order:
    1. In-memory cache (fastest — avoids disk on repeat queries)
    2. Pickle file on disk (loads once per process lifetime, then caches)
    3. Returns None if neither exists (collection not yet ingested)
    """
    if collection_name in _INDEX_CACHE:
        return _INDEX_CACHE[collection_name]

    index_path = _get_index_path(collection_name)
    if not os.path.exists(index_path):
        return None

    with open(index_path, "rb") as f:
        index = pickle.load(f)

    _INDEX_CACHE[collection_name] = index
    return index


# ---------------------------------------------------------------------------
# Public search API (used by retriever.py)
# ---------------------------------------------------------------------------

def bm25_search_collection(
    collection_name: str,
    query: str,
    k: int = 10,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Search the BM25 index for a collection.

    Uses the persisted + cached index — O(query_terms * n_docs) at search
    time only, with no corpus re-loading or re-tokenisation per query.

    Falls back gracefully to an empty list if the collection has not been
    ingested yet or the index file is missing.
    """
    index = load_bm25_index(collection_name)

    # Fallback: if no pickled index exists (e.g. legacy collection ingested
    # before this change), rebuild from JSONL corpus on-the-fly and cache.
    if index is None:
        corpus = load_chunk_corpus(collection_name)
        if not corpus:
            return []
        texts = [row.get("text", "") for row in corpus]
        index = SimpleBM25(texts)
        _INDEX_CACHE[collection_name] = index

    # We still need the corpus for metadata lookup after hits
    corpus = load_chunk_corpus(collection_name)
    if not corpus:
        return []

    hits = index.search(query, k=k)

    results: List[Tuple[Dict[str, Any], float]] = []
    for h in hits:
        if h.index >= len(corpus):
            continue
        row = corpus[h.index]
        result_doc = {"text": row.get("text", ""), "metadata": row.get("metadata", {})}
        results.append((result_doc, float(h.score)))
    return results
