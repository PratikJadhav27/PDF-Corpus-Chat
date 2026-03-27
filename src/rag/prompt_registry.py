from typing import Dict


PROMPTS: Dict[str, str] = {
    "v1_strict": """
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
""".strip(),
    "v2_concise": """
You are a grounded assistant.

Rules:
1) Use only context.
2) If missing, reply exactly: I don't know based on the provided documents.
3) Keep the answer concise and factual.
4) End each sentence with one citation: [<source> | page <page_number>]
5) Never invent citations.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip(),
}


DEFAULT_PROMPT_VERSION = "v1_strict"


def get_prompt_template(version: str) -> str:
    return PROMPTS.get(version, PROMPTS[DEFAULT_PROMPT_VERSION])
