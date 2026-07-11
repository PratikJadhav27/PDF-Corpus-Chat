"""
src/eval/metrics.py

LLM-as-a-judge evaluation metrics for the RAG pipeline using DSPy.

Metrics:
  - faithfulness: Does the answer rely only on the retrieved context?
  - context_relevance: Does the retrieved context actually contain the answer?
  - answer_completeness: Does the answer fully address the question?

Each metric returns a float in [0.0, 1.0] and a brief explanation.
"""

import os
import re
from typing import Any, Dict, List

import dspy
from langchain_core.documents import Document

_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gemini-2.5-flash")

# Configure DSPy to use the Gemini model
# DSPy 2.4+ uses litellm for gemini/ providers or google native.
gemini_lm = dspy.LM(f"gemini/{_JUDGE_MODEL}")
dspy.configure(lm=gemini_lm)

def _extract_score(text: str) -> float:
    """Parse the first float/int in [0.0, 1.0] from the judge's response."""
    if not text:
        return 0.0
    match = re.search(r"\b([01](?:\.\d+)?)\b", text)
    if match:
        val = float(match.group(1))
        return round(min(max(val, 0.0), 1.0), 4)
    return 0.0


# ---------------------------------------------------------------------------
# Metric 1: Faithfulness
# ---------------------------------------------------------------------------
class AssessFaithfulness(dspy.Signature):
    """Evaluate whether the ANSWER is grounded exclusively in the CONTEXT provided.
    - Score 1.0 if every claim in the answer can be directly attributed to the context.
    - Score 0.5 if most claims are grounded but there are minor unsupported additions.
    - Score 0.0 if the answer contains significant information NOT present in the context.
    """
    context = dspy.InputField(desc="The context retrieved for the question")
    answer = dspy.InputField(desc="The answer to evaluate")
    explanation = dspy.OutputField(desc="One sentence explanation for the score")
    score = dspy.OutputField(desc="A float between 0.0 and 1.0")

def evaluate_faithfulness(answer: str, docs: List[Document]) -> Dict[str, Any]:
    """Measure whether the answer is grounded in the retrieved context."""
    if not answer or not docs:
        return {"score": 0.0, "explanation": "No answer or context provided.", "error": None}
    
    context_text = "\n\n---\n\n".join(
        f"[{d.metadata.get('source', 'Unknown')} | page {d.metadata.get('page_number', '?')}]\n{d.page_content}"
        for d in docs
    )
    
    judge = dspy.ChainOfThought(AssessFaithfulness)
    try:
        res = judge(context=context_text[:6000], answer=answer[:2000])
        score_val = _extract_score(str(res.score))
        return {
            "score": score_val,
            "explanation": str(res.explanation)[:500],
            "raw_response": f"Reasoning: {res.reasoning}\nScore: {res.score}\nExplanation: {res.explanation}",
            "metric": "faithfulness",
            "error": None
        }
    except Exception as exc:
        return {"score": 0.0, "explanation": "", "raw_response": "", "metric": "faithfulness", "error": str(exc)}


# ---------------------------------------------------------------------------
# Metric 2: Context Relevance
# ---------------------------------------------------------------------------
class AssessContextRelevance(dspy.Signature):
    """Evaluate whether the CONTEXT is relevant to answer the QUESTION.
    - Score 1.0 if the context directly contains information needed to answer the question.
    - Score 0.5 if the context is partially relevant but missing some key details.
    - Score 0.0 if the context is irrelevant or unable to support answering the question.
    """
    question = dspy.InputField()
    context = dspy.InputField()
    explanation = dspy.OutputField(desc="One sentence explanation for the score")
    score = dspy.OutputField(desc="A float between 0.0 and 1.0")

def evaluate_context_relevance(question: str, docs: List[Document]) -> Dict[str, Any]:
    """Measure whether the retrieved context is actually relevant to the question."""
    if not question or not docs:
        return {"score": 0.0, "explanation": "No question or context provided.", "error": None}
        
    context_text = "\n\n---\n\n".join(
        f"[{d.metadata.get('source', 'Unknown')} | page {d.metadata.get('page_number', '?')}]\n{d.page_content}"
        for d in docs
    )
    
    judge = dspy.ChainOfThought(AssessContextRelevance)
    try:
        res = judge(question=question[:1000], context=context_text[:6000])
        score_val = _extract_score(str(res.score))
        return {
            "score": score_val,
            "explanation": str(res.explanation)[:500],
            "raw_response": f"Reasoning: {res.reasoning}\nScore: {res.score}\nExplanation: {res.explanation}",
            "metric": "context_relevance",
            "error": None
        }
    except Exception as exc:
        return {"score": 0.0, "explanation": "", "raw_response": "", "metric": "context_relevance", "error": str(exc)}


# ---------------------------------------------------------------------------
# Metric 3: Answer Completeness
# ---------------------------------------------------------------------------
class AssessAnswerCompleteness(dspy.Signature):
    """Evaluate whether the ANSWER fully addresses the QUESTION.
    - Score 1.0 if the answer completely and thoroughly answers the question.
    - Score 0.5 if the answer partially addresses the question but misses important aspects.
    - Score 0.0 if the answer fails to address the question or only gives a generic IDK response.
    """
    question = dspy.InputField()
    answer = dspy.InputField()
    explanation = dspy.OutputField(desc="One sentence explanation for the score")
    score = dspy.OutputField(desc="A float between 0.0 and 1.0")

def evaluate_answer_completeness(question: str, answer: str) -> Dict[str, Any]:
    """Measure whether the answer fully addresses the question asked."""
    if not question or not answer:
        return {"score": 0.0, "explanation": "No question or answer provided.", "error": None}
        
    judge = dspy.ChainOfThought(AssessAnswerCompleteness)
    try:
        res = judge(question=question[:1000], answer=answer[:2000])
        score_val = _extract_score(str(res.score))
        return {
            "score": score_val,
            "explanation": str(res.explanation)[:500],
            "raw_response": f"Reasoning: {res.reasoning}\nScore: {res.score}\nExplanation: {res.explanation}",
            "metric": "answer_completeness",
            "error": None
        }
    except Exception as exc:
        return {"score": 0.0, "explanation": "", "raw_response": "", "metric": "answer_completeness", "error": str(exc)}


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
