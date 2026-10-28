"""
Regression tests for the citation pills shown inside an answer.

The answer is built as one HTML string and then split into paragraphs on blank
lines. The pill tooltip embeds the matched chunk's excerpt with a ``\\n\\n``
separator, so a raw newline inside the ``<span data-tooltip="...">`` attribute
used to split the tag in half: the browser then rendered the tooltip text as
part of the answer (``…47 secon…">[1]``) and the pill disappeared.

Live symptom reproduced in the app before the fix; these tests pin the markup
shape so it cannot come back.
"""
import re

from langchain_core.documents import Document

from app.views.chat import _process_citations


def _doc(source="aurora.pdf", page=1, text="Excerpt text from the document."):
    return Document(
        page_content=text,
        metadata={"source": source, "page_number": page},
    )


def _paragraphs(html_out: str):
    return re.findall(r"<p [^>]*>(.*?)</p>", html_out, re.S)


def test_answer_html_never_contains_raw_newlines():
    html_out, citations = _process_citations(
        "The window is 47 seconds [aurora.pdf | page 1].",
        [_doc()],
    )

    assert citations, "citation should be registered"
    assert "\n" not in html_out
    assert "\r" not in html_out


def test_tooltip_newline_is_entity_encoded():
    html_out, _ = _process_citations(
        "The window is 47 seconds [aurora.pdf | page 1].",
        [_doc()],
    )

    assert "&#10;" in html_out


def test_pill_tag_is_not_split_across_paragraphs():
    html_out, _ = _process_citations(
        "First fact [aurora.pdf | page 1].\n\nSecond fact [aurora.pdf | page 1].",
        [_doc()],
    )

    paras = _paragraphs(html_out)
    assert len(paras) == 2

    for para in paras:
        assert para.count("<sup>") == para.count("</sup>")
        assert para.count('class="dm-cite-pill"') == para.count("</span></sup>")


def test_tooltip_keeps_source_and_excerpt():
    html_out, _ = _process_citations(
        "Fact [aurora.pdf | page 1].",
        [_doc(text="Battery pack rated for 900 cycles.")],
    )

    assert "aurora.pdf" in html_out
    assert "page 1" in html_out
    # the excerpt travels in the tooltip, not in the visible answer text
    assert "Battery pack rated for 900 cycles." not in _paragraphs(html_out)[0].split("<sup>")[0]
    assert "Battery pack rated for 900 cycles." in html_out


def test_tooltip_cannot_close_its_own_attribute():
    html_out, _ = _process_citations(
        "Fact [aurora.pdf | page 1].",
        [_doc(text='he said "quit" <script>alert(1)</script>')],
    )

    # The document excerpt cannot introduce a quote, nor a tag, of its own.
    assert 'data-tooltip="📄 aurora.pdf  ·  page 1&#10;&#10;' in html_out
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_answer_without_citations_is_still_paragraph_wrapped():
    html_out, citations = _process_citations("No sources here.", [])

    assert citations == []
    assert html_out.startswith("<p ")
    assert html_out.endswith("</p>")
