"""
tests/test_validators.py

Unit tests for src/rag/validators.py

Tests cover:
- Citation parsing regex
- Allowed citation set construction
- Citation validity ratio calculation
- Guard fallback behavior
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from src.rag.validators import (
    CITATION_PATTERN,
    FALLBACK,
    parse_citations,
    build_allowed_citations,
    citation_validity,
    guard_answer_with_citation_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(source: str, page) -> Document:
    return Document(
        page_content="some content",
        metadata={"source": source, "page_number": page},
    )


# ---------------------------------------------------------------------------
# parse_citations
# ---------------------------------------------------------------------------

class TestParseCitations:
    def test_single_citation(self):
        answer = "The sky is blue. [report.pdf | page 3]"
        result = parse_citations(answer)
        assert result == [("report.pdf", "3")]

    def test_multiple_citations(self):
        answer = "Point A [a.pdf | page 1] and point B [b.pdf | page 5]."
        result = parse_citations(answer)
        assert ("a.pdf", "1") in result
        assert ("b.pdf", "5") in result

    def test_no_citations(self):
        assert parse_citations("No citations here.") == []

    def test_empty_string(self):
        assert parse_citations("") == []

    def test_none_input(self):
        assert parse_citations(None) == []

    def test_case_insensitive_page_keyword(self):
        answer = "[doc.pdf | PAGE 7]"
        result = parse_citations(answer)
        assert result == [("doc.pdf", "7")]

    def test_citation_with_extra_spaces(self):
        answer = "[myfile.pdf |   page   4]"
        result = parse_citations(answer)
        assert result == [("myfile.pdf", "4")]

    def test_non_numeric_page_not_matched(self):
        # The regex requires digits for page number
        answer = "[doc.pdf | page xyz]"
        result = parse_citations(answer)
        assert result == []

    def test_duplicate_citations_are_all_returned(self):
        answer = "[a.pdf | page 1] [a.pdf | page 1]"
        result = parse_citations(answer)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# build_allowed_citations
# ---------------------------------------------------------------------------

class TestBuildAllowedCitations:
    def test_single_doc(self):
        docs = [_doc("report.pdf", 2)]
        allowed = build_allowed_citations(docs)
        assert ("report.pdf", "2") in allowed

    def test_multiple_docs(self):
        docs = [_doc("a.pdf", 1), _doc("b.pdf", 3)]
        allowed = build_allowed_citations(docs)
        assert ("a.pdf", "1") in allowed
        assert ("b.pdf", "3") in allowed

    def test_empty_docs(self):
        assert build_allowed_citations([]) == set()

    def test_source_stripped(self):
        d = Document(page_content="", metadata={"source": "  file.pdf  ", "page_number": 1})
        allowed = build_allowed_citations([d])
        assert ("file.pdf", "1") in allowed

    def test_missing_metadata_uses_defaults(self):
        d = Document(page_content="", metadata={})
        allowed = build_allowed_citations([d])
        assert ("Unknown", "?") in allowed


# ---------------------------------------------------------------------------
# citation_validity
# ---------------------------------------------------------------------------

class TestCitationValidity:
    def test_all_valid(self):
        docs = [_doc("report.pdf", 3)]
        answer = "Claims made here. [report.pdf | page 3]"
        result = citation_validity(answer, docs)
        assert result["citation_count"] == 1
        assert result["valid_citation_count"] == 1
        assert result["validity_ratio"] == 1.0

    def test_all_invalid(self):
        docs = [_doc("report.pdf", 3)]
        answer = "Claims. [other.pdf | page 99]"
        result = citation_validity(answer, docs)
        assert result["citation_count"] == 1
        assert result["valid_citation_count"] == 0
        assert result["validity_ratio"] == 0.0

    def test_mixed_validity(self):
        docs = [_doc("a.pdf", 1)]
        answer = "[a.pdf | page 1] [fake.pdf | page 5]"
        result = citation_validity(answer, docs)
        assert result["citation_count"] == 2
        assert result["valid_citation_count"] == 1
        assert result["validity_ratio"] == 0.5

    def test_no_citations_in_answer(self):
        docs = [_doc("a.pdf", 1)]
        result = citation_validity("No citations here.", docs)
        assert result["citation_count"] == 0
        assert result["validity_ratio"] == 0.0

    def test_empty_answer(self):
        docs = [_doc("a.pdf", 1)]
        result = citation_validity("", docs)
        assert result["citation_count"] == 0

    def test_empty_docs(self):
        result = citation_validity("[a.pdf | page 1]", [])
        assert result["citation_count"] == 1
        assert result["valid_citation_count"] == 0
        assert result["validity_ratio"] == 0.0


# ---------------------------------------------------------------------------
# guard_answer_with_citation_check
# ---------------------------------------------------------------------------

class TestGuardAnswer:
    def test_valid_answer_passes_through(self):
        docs = [_doc("report.pdf", 1)]
        answer = "The sky is blue. [report.pdf | page 1]"
        result = guard_answer_with_citation_check(answer, docs)
        assert result == answer

    def test_answer_with_no_citations_returns_fallback(self):
        docs = [_doc("report.pdf", 1)]
        answer = "The sky is blue."
        result = guard_answer_with_citation_check(answer, docs)
        assert result == FALLBACK

    def test_answer_below_threshold_returns_fallback(self):
        docs = [_doc("report.pdf", 1)]
        # One valid + one invalid → ratio 0.5 < 0.8 threshold
        answer = "[report.pdf | page 1] [fake.pdf | page 99]"
        result = guard_answer_with_citation_check(answer, docs, min_validity_ratio=0.8)
        assert result == FALLBACK

    def test_answer_exactly_at_threshold_passes(self):
        docs = [_doc("report.pdf", 1)]
        answer = "[report.pdf | page 1]"  # 100% valid
        result = guard_answer_with_citation_check(answer, docs, min_validity_ratio=1.0)
        assert result == answer

    def test_idk_style_answer_with_no_citations_returns_fallback(self):
        docs = [_doc("report.pdf", 1)]
        answer = "I don't know based on the provided documents."
        result = guard_answer_with_citation_check(answer, docs)
        assert result == FALLBACK

    def test_custom_threshold_0_always_passes_if_any_citation_exists(self):
        docs = [_doc("report.pdf", 1)]
        # Even completely invalid citation should pass at threshold 0.0
        answer = "[totally_wrong.pdf | page 999]"
        result = guard_answer_with_citation_check(answer, docs, min_validity_ratio=0.0)
        # citation_count=1, valid=0, ratio=0.0 >= 0.0 → should pass
        assert result == answer
