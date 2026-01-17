from typing import List, Dict, Any
from pypdf import PdfReader


def load_pdf_pages(pdf_file) -> List[Dict[str, Any]]:
    """
    Loads PDF pages safely.
    If a page fails extraction (bad fonts etc), it won't crash the app.
    """
    reader = PdfReader(pdf_file)

    pages: List[Dict[str, Any]] = []
    source_name = getattr(pdf_file, "name", "unknown.pdf")

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            # Some PDFs break due to missing font metadata like bbox
            text = ""

        pages.append(
            {
                "text": text.strip(),
                "source": source_name,
                "page_number": i + 1,
            }
        )

    return pages
