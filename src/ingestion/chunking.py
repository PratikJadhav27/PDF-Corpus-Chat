from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNKING


def chunk_pdf_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Splits PDF pages into smaller chunks while keeping metadata.

    Returns list of chunks:
      {
        "text": "...",
        "metadata": {
            "source": "filename.pdf",
            "page_number": 1,
            "chunk_index": 0
        }
      }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNKING.chunk_size,
        chunk_overlap=CHUNKING.chunk_overlap,
    )

    chunks: List[Dict[str, Any]] = []

    for page in pages:
        page_text = page.get("text", "")
        if not page_text:
            continue

        splits = splitter.split_text(page_text)
        for idx, chunk_text in enumerate(splits):
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "source": page.get("source", "unknown.pdf"),
                        "page_number": page.get("page_number", None),
                        "chunk_index": idx,
                    },
                }
            )

    return chunks
