"""
src/eval/metrics.py

LLM-as-a-judge evaluation metrics for the RAG pipeline.

Metrics:
  - faithfulness: Does the answer rely only on the retrieved context?
  - context_relevance: Does the retrieved context actually contain the answer?
  - answer_completeness: Does the answer fully address the question?

Each metric returns a float in [0.0, 1.0] and a brief explanation.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gemini-2.5-flash")
_JUDGE_TEMPERATURE = 0.0  # deterministic scoring


def _build_judge_model() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=_JUDGE_MODEL, temperature=_JUDGE_TEMPERATURE)


def _extract_score(text: str) -> Optional[float]:
    """Parse the first float/int in [0.0, 1.0] from the judge's response."""
    match = re.search(r"\b([01](?:\.\d+)?)\b", text)
    if match:
        val = float(match.group(1))
        return round(min(max(val, 0.0), 1.0), 4)
    return None


def _call_judge(prompt: str) -> Dict[str, Any]:
    """Call the LLM judge and return {score, explanation, raw_response}."""
    try:
        model = _build_judge_model()
        raw = model.invoke(prompt).content or ""
        score = _extract_score(raw)
        # Everything after the score line is the explanation
        lines = raw.strip().splitlines()
        explanation = " ".join(lines).strip()
        return {
            "score": score if score is not None else 0.0,
            "explanation": explanation[:500],
            "raw_response": raw[:800],
            "error": None,
        }
    except Exception as exc:
        return {
            "score": 0.0,
            "explanation": "",
            "raw_response": "",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Metric 1: Faithfulness
# ---------------------------------------------------------------------------
_FAITHFULNESS_PROMPT = """You are an evaluation judge for a Retrieval-Augmented Generation (RAG) system.

TASK: Evaluate whether the ANSWER is grounded exclusively in the CONTEXT provided.
- Score 1.0 if every claim in the answer can be directly attributed to the context.
- Score 0.5 if most claims are grounded but there are minor unsupported additions.
- Score 0.0 if the answer contains significant information NOT present in the context.

Respond with ONLY a score between 0.0 and 1.0 on the first line, then a one-sentence explanation.

CONTEXT:
{context}

ANSWER:
{answer}

SCORE (0.0-1.0):"""


def evaluate_faithfulness(answer: str, docs: List[Document]) -> Dict[str, Any]:
    """Measure whether the answer is grounded in the retrieved context."""
    if not answer or not docs:
        return {"score": 0.0, "explanation": "No answer or context provided.", "error": None}

    context_text = "\n\n---\n\n".join(
        f"[{d.metadata.get('source', 'Unknown')} | page {d.metadata.get('page_number', '?')}]\n{d.page_content}"
        for d in docs
    )
    prompt = _FAITHFULNESS_PROMPT.format(
        context=context_text[:6000],
        answer=answer[:2000],
    )
    result = _call_judge(prompt)
    result["metric"] = "faithfulness"
    return result


# ---------------------------------------------------------------------------
# Metric 2: Context Relevance
# ---------------------------------------------------------------------------
_CONTEXT_RELEVANCE_PROMPT = """You are an evaluation judge for a Retrieval-Augmented Generation (RAG) system.

TASK: Evaluate whether the CONTEXT is relevant to answer the QUESTION.
- Score 1.0 if the context directly contains information needed to answer the question.
- Score 0.5 if the context is partially relevant but missing some key details.
- Score 0.0 if the context is irrelevant or unable to support answering the question.

Respond with ONLY a score between 0.0 and 1.0 on the first line, then a one-sentence explanation.

QUESTION:
{question}

CONTEXT:
{context}

SCORE (0.0-1.0):"""


def evaluate_context_relevance(question: str, docs: List[Document]) -> Dict[str, Any]:
    """Measure whether the retrieved context is actually relevant to the question."""
    if not question or not docs:
        return {"score": 0.0, "explanation": "No question or context provided.", "error": None}

    context_text = "\n\n---\n\n".join(
        f"[{d.metadata.get('source', 'Unknown')} | page {d.metadata.get('page_number', '?')}]\n{d.page_content}"
        for d in docs
    )
    prompt = _CONTEXT_RELEVANCE_PROMPT.format(
        question=question[:1000],
        context=context_text[:6000],
    )
    result = _call_judge(prompt)
    result["metric"] = "context_relevance"
    return result


# ---------------------------------------------------------------------------
# Metric 3: Answer Completeness
# ---------------------------------------------------------------------------
_COMPLETENESS_PROMPT = """You are an evaluation judge for a Retrieval-Augmented Generation (RAG) system.

TASK: Evaluate whether the ANSWER fully addresses the QUESTION.
- Score 1.0 if the answer completely and thoroughly answers the question.
- Score 0.5 if the answer partially addresses the question but misses important aspects.
- Score 0.0 if the answer fails to address the question or only gives a generic IDK response.

Respond with ONLY a score between 0.0 and 1.0 on the first line, then a one-sentence explanation.

QUESTION:
{question}

ANSWER:
{answer}

SCORE (0.0-1.0):"""


def evaluate_answer_completeness(question: str, answer: str) -> Dict[str, Any]:
    """Measure whether the answer fully addresses the question asked."""
    if not question or not answer:
        return {"score": 0.0, "explanation": "No question or answer provided.", "error": None}

    prompt = _COMPLETENESS_PROMPT.format(
        question=question[:1000],
        answer=answer[:2000],
    )
    result = _call_judge(prompt)
    result["metric"] = "answer_completeness"
    return result


# ---------------------------------------------------------------------------
# Composite: run all metrics for a single Q&A turn
# ---------------------------------------------------------------------------
def evaluate_all_metrics(
    question: str,
    answer: str,
    docs: List[Document],
) -> Dict[str, Any]:
    """
    Run all three LLM-judge metrics for a single question/answer/context tuple.
    Returns a dict with each metric's score and explanation, plus a composite mean.
    """
    faithfulness = evaluate_faithfulness(answer, docs)
    context_relevance = evaluate_context_relevance(question, docs)
    completeness = evaluate_answer_completeness(question, answer)

    scores = [
        faithfulness["score"],
        context_relevance["score"],
        completeness["score"],
    ]
    composite = round(sum(scores) / len(scores), 4)

    return {
        "faithfulness": faithfulness,
        "context_relevance": context_relevance,
        "answer_completeness": completeness,
        "composite_score": composite,
    }
