"""
tests/test_entailment.py

Unit tests for src/eval/entailment.py.

All tests mock out the DSPy judge so no API key is required.
We verify:
  - Citation parsing from answer text
  - Chunk lookup from retrieved docs
  - Score aggregation / entailment_rate computation
  - Edge cases: empty answer, no citations, missing chunk, all-fail
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.eval.entailment import (
    _parse_citations,
    _find_cited_chunk,
    evaluate_citation_entailment,
    _parse_entailment_score,
)


# ---------------------------------------------------------------------------
# _parse_citations
# ---------------------------------------------------------------------------
class TestParseCitations:
    def test_single_citation(self):
        answer = "The model uses 8 heads [attention.pdf | page 3]."
        cits = _parse_citations(answer)
        assert len(cits) == 1
        assert cits[0]["source"] == "attention.pdf"
        assert cits[0]["page"] == "3"

    def test_multiple_citations_in_one_sentence(self):
        answer = "Revenue grew [report.pdf | page 1] and costs fell [report.pdf | page 2]."
        cits = _parse_citations(answer)
        assert len(cits) == 2

    def test_multiple_sentences(self):
        answer = (
            "The model has 6 layers [bert.pdf | page 5]. "
            "It uses masked attention [bert.pdf | page 6]."
        )
        cits = _parse_citations(answer)
        assert len(cits) == 2

    def test_no_citations_returns_empty(self):
        answer = "The model is very large."
        assert _parse_citations(answer) == []

    def test_empty_answer_returns_empty(self):
        assert _parse_citations("") == []

    def test_none_answer_returns_empty(self):
        assert _parse_citations(None) == []

    def test_claim_has_citation_bracket_stripped(self):
        answer = "Attention uses queries [paper.pdf | page 3]."
        cits = _parse_citations(answer)
        assert "[paper.pdf" not in cits[0]["claim"]
        assert "Attention uses queries" in cits[0]["claim"]

    def test_source_is_lowercased(self):
        answer = "Result [Attention_Is_All_You_Need.PDF | page 1]."
        cits = _parse_citations(answer)
        assert cits[0]["source"] == "attention_is_all_you_need.pdf"

    def test_source_is_basename_only(self):
        answer = "Result [/some/path/to/paper.pdf | page 4]."
        cits = _parse_citations(answer)
        assert cits[0]["source"] == "paper.pdf"


# ---------------------------------------------------------------------------
# _find_cited_chunk
# ---------------------------------------------------------------------------
class TestFindCitedChunk:
    def _make_doc(self, source, page, content):
        return Document(
            page_content=content,
            metadata={"source": source, "page_number": page},
        )

    def test_exact_match_returns_content(self):
        docs = [self._make_doc("report.pdf", 3, "Revenue grew 10%.")]
        result = _find_cited_chunk("report.pdf", "3", docs)
        assert result == "Revenue grew 10%."

    def test_no_match_returns_none(self):
        docs = [self._make_doc("report.pdf", 3, "Revenue grew 10%.")]
        assert _find_cited_chunk("other.pdf", "3", docs) is None

    def test_page_mismatch_returns_none(self):
        docs = [self._make_doc("report.pdf", 3, "Revenue grew 10%.")]
        assert _find_cited_chunk("report.pdf", "5", docs) is None

    def test_case_insensitive_source_match(self):
        docs = [self._make_doc("/path/to/Report.PDF", 1, "Big revenue.")]
        # _find_cited_chunk lowercases both sides
        result = _find_cited_chunk("report.pdf", "1", docs)
        assert result == "Big revenue."

    def test_empty_docs_returns_none(self):
        assert _find_cited_chunk("report.pdf", "1", []) is None


# ---------------------------------------------------------------------------
# _parse_entailment_score
# ---------------------------------------------------------------------------
class TestParseEntailmentScore:
    def test_score_10(self):
        assert _parse_entailment_score("1.0") == 1.0

    def test_score_05(self):
        assert _parse_entailment_score("0.5") == 0.5

    def test_score_00(self):
        assert _parse_entailment_score("0.0") == 0.0

    def test_score_embedded_in_text(self):
        assert _parse_entailment_score("The score is 1.0 because it matches.") == 1.0

    def test_integer_1(self):
        assert _parse_entailment_score("1") == 1.0

    def test_integer_0(self):
        assert _parse_entailment_score("0") == 0.0

    def test_empty_returns_zero(self):
        assert _parse_entailment_score("") == 0.0

    def test_none_returns_zero(self):
        assert _parse_entailment_score(None) == 0.0


# ---------------------------------------------------------------------------
# evaluate_citation_entailment — mocked judge
# ---------------------------------------------------------------------------

def _make_doc(source, page, content):
    return Document(
        page_content=content,
        metadata={"source": source, "page_number": page},
    )


def _mock_judge_result(score="1.0", explanation="Fully supported."):
    result = MagicMock()
    result.entailment_score = score
    result.explanation = explanation
    result.reasoning = "Step by step reasoning."
    return result


class TestEvaluateCitationEntailment:
    def test_empty_answer_returns_zero_rate(self):
        result = evaluate_citation_entailment("", [])
        assert result["entailment_rate"] == 0.0
        assert result["error"] == "Empty answer"

    def test_no_citations_returns_vacuous_true(self):
        answer = "The model is large."
        result = evaluate_citation_entailment(answer, [])
        assert result["entailment_rate"] == 1.0
        assert result["citation_count"] == 0
        assert result["error"] is None

    def test_citation_not_in_docs_scores_zero(self):
        answer = "The model uses 8 heads [paper.pdf | page 5]."
        docs = [_make_doc("other.pdf", 1, "Unrelated text.")]

        with patch("src.eval.entailment._ensure_dspy_configured"):
            with patch("src.eval.entailment.dspy.ChainOfThought"):
                result = evaluate_citation_entailment(answer, docs)

        assert result["not_found_count"] == 1
        assert result["citations"][0]["score"] == 0.0

    def test_fully_entailed_citation(self):
        answer = "BERT uses masked LM [bert.pdf | page 2]."
        docs = [_make_doc("bert.pdf", 2, "BERT uses masked language modelling for pre-training.")]

        mock_judge_instance = MagicMock()
        mock_judge_instance.return_value = _mock_judge_result("1.0", "Supported by passage.")

        with patch("src.eval.entailment._ensure_dspy_configured"):
            with patch("src.eval.entailment.dspy.ChainOfThought", return_value=mock_judge_instance):
                result = evaluate_citation_entailment(answer, docs)

        assert result["entailment_rate"] == 1.0
        assert result["entailed_count"] == 1
        assert result["citation_count"] == 1
        assert result["error"] is None

    def test_partial_entailment(self):
        answer = "BERT uses masked LM [bert.pdf | page 2]."
        docs = [_make_doc("bert.pdf", 2, "BERT uses various pre-training techniques.")]

        mock_judge_instance = MagicMock()
        mock_judge_instance.return_value = _mock_judge_result("0.5", "Partially supported.")

        with patch("src.eval.entailment._ensure_dspy_configured"):
            with patch("src.eval.entailment.dspy.ChainOfThought", return_value=mock_judge_instance):
                result = evaluate_citation_entailment(answer, docs)

        assert result["entailment_rate"] == 0.5
        assert result["entailed_count"] == 1   # >= 0.5 counts as entailed

    def test_not_entailed(self):
        answer = "Revenue was $10B [paper.pdf | page 1]."
        docs = [_make_doc("paper.pdf", 1, "The weather was sunny.")]

        mock_judge_instance = MagicMock()
        mock_judge_instance.return_value = _mock_judge_result("0.0", "Not supported.")

        with patch("src.eval.entailment._ensure_dspy_configured"):
            with patch("src.eval.entailment.dspy.ChainOfThought", return_value=mock_judge_instance):
                result = evaluate_citation_entailment(answer, docs)

        assert result["entailment_rate"] == 0.0
        assert result["entailed_count"] == 0

    def test_multiple_citations_averages_scores(self):
        answer = (
            "BERT uses 12 heads [bert.pdf | page 1]. "
            "GPT uses 16 heads [gpt.pdf | page 2]."
        )
        docs = [
            _make_doc("bert.pdf", 1, "BERT uses 12 attention heads."),
            _make_doc("gpt.pdf", 2, "GPT-2 uses 16 attention heads."),
        ]

        call_count = [0]
        def mock_judge_factory(*args, **kwargs):
            instance = MagicMock()
            def side_effect(**kw):
                call_count[0] += 1
                # First call: score 1.0, second: score 0.0
                score = "1.0" if call_count[0] == 1 else "0.0"
                return _mock_judge_result(score)
            instance.side_effect = side_effect
            return instance

        with patch("src.eval.entailment._ensure_dspy_configured"):
            with patch("src.eval.entailment.dspy.ChainOfThought", side_effect=mock_judge_factory):
                result = evaluate_citation_entailment(answer, docs)

        assert result["citation_count"] == 2
        assert result["entailment_rate"] == pytest.approx(0.5)

    def test_judge_exception_scores_zero(self):
        answer = "The model has 512 dimensions [paper.pdf | page 3]."
        docs = [_make_doc("paper.pdf", 3, "Some content here.")]

        mock_judge_instance = MagicMock()
        mock_judge_instance.side_effect = RuntimeError("API timeout")

        with patch("src.eval.entailment._ensure_dspy_configured"):
            with patch("src.eval.entailment.dspy.ChainOfThought", return_value=mock_judge_instance):
                result = evaluate_citation_entailment(answer, docs)

        assert result["citations"][0]["score"] == 0.0
        assert result["citations"][0]["error"] is not None
        assert result["entailment_rate"] == 0.0

    def test_result_keys_always_present(self):
        result = evaluate_citation_entailment("", [])
        required_keys = {
            "entailment_rate", "citation_count", "entailed_count",
            "not_found_count", "citations", "error"
        }
        assert required_keys.issubset(result.keys())
