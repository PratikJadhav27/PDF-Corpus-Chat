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
        parts.append(f"[{source} | page {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def build_grounded_prompt(context: str, question: str) -> str:
    return f"""
You are a strict document-grounded assistant.

You MUST follow these rules:
- Use ONLY the context provided.
- If the answer is not explicitly present in the context, reply exactly:
I don't know based on the provided documents.
- Every sentence in your answer MUST end with a citation in this exact format:
[<source> | page <page_number>]
- Do NOT add citations that are not in the context.
- Do NOT use outside knowledge.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (with citations at end of every sentence):
""".strip()

