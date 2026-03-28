import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.config import OBSERVABILITY
from src.observability.store import append_jsonl


def is_tracing_enabled() -> bool:
    env = os.getenv("OBSERVABILITY_ENABLED")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return OBSERVABILITY.enabled


def new_trace_id() -> str:
    return str(uuid.uuid4())


def _preview(text: str, max_len: int = 400) -> str:
    if not text:
        return ""
    t = text.strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def serialize_retrieved_chunks(docs: List[Document]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in docs or []:
        md = d.metadata or {}
        row: Dict[str, Any] = {
            "source": md.get("source", "Unknown"),
            "page_number": md.get("page_number"),
            "chunk_index": md.get("chunk_index"),
            "text_preview": _preview(d.page_content or ""),
        }
        r = md.get("retrieval")
        if isinstance(r, dict):
            row["retrieval"] = {
                "rrf_score": r.get("rrf_score"),
                "rerank_score": r.get("rerank_score"),
                "overlap_score": r.get("overlap_score"),
                "vector_score": r.get("vector_score"),
                "bm25_score": r.get("bm25_score"),
                "channels": r.get("channels"),
            }
        out.append(row)
    return out


def append_rag_trace(record: Dict[str, Any]) -> None:
    if not is_tracing_enabled():
        return
    path = os.getenv("OBSERVABILITY_TRACES_PATH") or OBSERVABILITY.traces_path
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    append_jsonl(path, payload)


def build_trace_record(
    *,
    trace_id: str,
    phase: str,
    collection: str,
    query: str,
    retrieval_mode: str,
    prompt_version: str,
    model_name: str,
    temperature: float,
    top_k: Optional[int] = None,
    candidate_k: Optional[int] = None,
    rerank_top_n: Optional[int] = None,
    docs: List[Document],
    latency_sec: Dict[str, float],
    citation_validity_ratio: Optional[float] = None,
    answer_preview: Optional[str] = None,
    error: Optional[str] = None,
    prompt_style: Optional[str] = None,
    eval_index: Optional[int] = None,
    eval_total: Optional[int] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "trace_id": trace_id,
        "phase": phase,
        "collection": collection,
        "query": query,
        "retrieval_mode": retrieval_mode,
        "prompt_version": prompt_version,
        "model": model_name,
        "temperature": temperature,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "rerank_top_n": rerank_top_n,
        "retrieved_chunks": serialize_retrieved_chunks(docs),
        "latency_sec": latency_sec,
        "citation_validity_ratio": citation_validity_ratio,
        "answer_preview": answer_preview,
        "error": error,
    }
    if prompt_style is not None:
        record["prompt_style"] = prompt_style
    if eval_index is not None:
        record["eval_index"] = eval_index
    if eval_total is not None:
        record["eval_total"] = eval_total
    return record
