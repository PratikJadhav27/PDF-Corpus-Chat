from typing import List
from langchain_core.documents import Document


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
        parts.append(f"[source={source} | page={page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def build_grounded_prompt(context: str, question: str) -> str:
    """
    Strict prompt to reduce hallucinations.
    Forces citations and grounded answers only.
    """
    return f"""
You are a strict document-grounded assistant.

RULES:
1) Answer ONLY using the context below.
2) If the answer is not in the context, say exactly:
"I don't know based on the provided documents."
3) When you use a fact, add a citation at the end of the sentence like:
[source.pdf | page 3]
4) Do NOT use outside knowledge.
5) Keep the answer clear and structured.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()
