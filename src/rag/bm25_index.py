import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def get_chunk_corpus_path(collection_name: str) -> str:
    safe = (collection_name or "default").strip().replace(" ", "_")
    return f"vector_stores/{safe}/chunks.jsonl"


def save_chunk_corpus(collection_name: str, chunks: List[Dict[str, Any]]) -> None:
    os.makedirs(f"vector_stores/{collection_name}", exist_ok=True)
    path = get_chunk_corpus_path(collection_name)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            row = {"text": c.get("text", ""), "metadata": c.get("metadata", {})}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_chunk_corpus(collection_name: str) -> List[Dict[str, Any]]:
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


@dataclass
class BM25DocScore:
    index: int
    score: float


class SimpleBM25:
    """
    Lightweight BM25 index to avoid additional runtime dependencies.
    """

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


def bm25_search_collection(
    collection_name: str,
    query: str,
    k: int = 10,
) -> List[Tuple[Dict[str, Any], float]]:
    corpus = load_chunk_corpus(collection_name)
    if not corpus:
        return []

    texts = [row.get("text", "") for row in corpus]
    bm25 = SimpleBM25(texts)
    hits = bm25.search(query, k=k)

    results: List[Tuple[Dict[str, Any], float]] = []
    for h in hits:
        row = corpus[h.index]
        result_doc = {"text": row.get("text", ""), "metadata": row.get("metadata", {})}
        results.append((result_doc, float(h.score)))
    return results
