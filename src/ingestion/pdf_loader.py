from typing import List, Dict, Any
from pypdf import PdfReader


def load_pdf_pages(pdf_file) -> List[Dict[str, Any]]:
    """
    Input: Streamlit uploaded file object (or file-like object)
    Output: List of dicts, one per page:
      {
        "text": "...",
        "source": "filename.pdf",
        "page_number": 1
      }
    """
    reader = PdfReader(pdf_file)

    pages: List[Dict[str, Any]] = []
    source_name = getattr(pdf_file, "name", "unknown.pdf")

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(
            {
                "text": text.strip(),
                "source": source_name,
                "page_number": i + 1,
            }
        )

    return pages
