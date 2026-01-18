from typing import List, Dict, Any, Tuple
from pypdf import PdfReader


def load_pdf_pages_with_stats(pdf_file) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Loads PDF pages safely + returns extraction stats.

    Returns:
      pages: List of dicts
      stats: dict with extraction info
    """
    reader = PdfReader(pdf_file)

    source_name = getattr(pdf_file, "name", "unknown.pdf")

    pages: List[Dict[str, Any]] = []

    total_pages = len(reader.pages)
    extracted_pages = 0
    empty_pages = 0
    failed_pages = 0

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            text = text.strip()

            if text:
                extracted_pages += 1
            else:
                empty_pages += 1

        except Exception:
            # Bad PDFs (fonts missing bbox etc.)
            text = ""
            failed_pages += 1

        pages.append(
            {
                "text": text,
                "source": source_name,
                "page_number": i + 1,
            }
        )

    stats = {
        "file_name": source_name,
        "total_pages": total_pages,
        "extracted_pages": extracted_pages,
        "empty_pages": empty_pages,
        "failed_pages": failed_pages,
        "extraction_success_rate": round((extracted_pages / total_pages) * 100, 2)
        if total_pages > 0
        else 0.0,
    }

    return pages, stats
