from typing import List
from langchain_core.documents import Document
from src.rag.prompt_registry import get_prompt_template


def build_context_with_citations(docs: List[Document]) -> str:
    """
    Converts retrieved docs into a context block that includes citation tags.
    Each chunk becomes:
      [source=... | page=...]
      chunk text
    """
    parts = []
    for d in docs:
        source = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page_number", "?")
        parts.append(f"[{source} | page {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def build_grounded_prompt(context: str, question: str, version: str = "v1_strict") -> str:
    template = get_prompt_template(version)
    return template.format(context=context, question=question)

