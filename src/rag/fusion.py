from typing import Any, Dict, List, Tuple


def chunk_key(metadata: Dict[str, Any]) -> str:
    source = metadata.get("source", "unknown")
    page = metadata.get("page_number", "?")
    chunk_index = metadata.get("chunk_index", "?")
    return f"{source}::{page}::{chunk_index}"


def rrf_fuse(
    vector_hits: List[Tuple[Dict[str, Any], float]],
    bm25_hits: List[Tuple[Dict[str, Any], float]],
    k_constant: int = 60,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) over two ranked lists.
    Returns merged rows with stable dedupe by chunk key.
    """
    merged: Dict[str, Dict[str, Any]] = {}

    def ingest(
        hits: List[Tuple[Dict[str, Any], float]],
        channel: str,
    ) -> None:
        for rank, (row, raw_score) in enumerate(hits, start=1):
            md = row.get("metadata", {})
            key = chunk_key(md)
            rrf_score = 1.0 / (k_constant + rank)

            if key not in merged:
                merged[key] = {
                    "key": key,
                    "text": row.get("text", ""),
                    "metadata": md,
                    "rrf_score": 0.0,
                    "vector_score": None,
                    "bm25_score": None,
                    "channels": [],
                }

            merged[key]["rrf_score"] += rrf_score
            merged[key]["channels"].append(channel)
            if channel == "vector":
                merged[key]["vector_score"] = raw_score
            elif channel == "bm25":
                merged[key]["bm25_score"] = raw_score

    ingest(vector_hits, "vector")
    ingest(bm25_hits, "bm25")

    rows = list(merged.values())
    rows.sort(key=lambda x: x["rrf_score"], reverse=True)
    return rows
