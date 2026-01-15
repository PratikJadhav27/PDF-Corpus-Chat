from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 1500
    chunk_overlap: int = 200


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 4


CHUNKING = ChunkingConfig()
RETRIEVAL = RetrievalConfig()
