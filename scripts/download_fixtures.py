#!/usr/bin/env python
"""
scripts/download_fixtures.py

Downloads fixture PDFs used by the golden dataset evaluation.

PDFs are NOT committed to the repository to avoid copyright concerns.
This script fetches them from their official public sources at CI/local
setup time and places them in evals/fixtures/.

Usage:
    python scripts/download_fixtures.py

The script is idempotent — it skips files that already exist and have
the expected size.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "evals" / "fixtures"

# ---------------------------------------------------------------------------
# Fixture registry
# Each entry: (filename, url, min_size_bytes, description)
# All PDFs are open-access papers freely distributable under their licenses.
# ---------------------------------------------------------------------------
FIXTURES = [
    (
        "attention_is_all_you_need.pdf",
        "https://arxiv.org/pdf/1706.03762",
        500_000,  # ~700KB expected
        "Vaswani et al. 2017 — Attention Is All You Need (arXiv:1706.03762)",
    ),
    (
        "bert_pretraining.pdf",
        "https://arxiv.org/pdf/1810.04805",
        400_000,  # ~600KB expected
        "Devlin et al. 2018 — BERT: Pre-training of Deep Bidirectional Transformers (arXiv:1810.04805)",
    ),
]


def download(url: str, dest: Path, min_size: int, description: str) -> bool:
    """Download url to dest. Returns True if downloaded, False if skipped."""
    if dest.exists() and dest.stat().st_size >= min_size:
        print(f"  [SKIP] {dest.name} already exists ({dest.stat().st_size:,} bytes)")
        return False

    print(f"  [DOWN] {dest.name}")
    print(f"         {description}")
    print(f"         {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except Exception as exc:
        print(f"  [FAIL] Could not download {url}: {exc}", file=sys.stderr)
        return False

    if len(data) < min_size:
        print(
            f"  [FAIL] Downloaded file too small ({len(data):,} bytes, expected >= {min_size:,})",
            file=sys.stderr,
        )
        return False

    dest.write_bytes(data)
    print(f"  [OK]   Saved {dest.name} ({len(data):,} bytes)")
    return True


def main() -> int:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fixture directory: {FIXTURES_DIR}\n")

    downloaded = 0
    skipped = 0
    failed = 0

    for filename, url, min_size, description in FIXTURES:
        dest = FIXTURES_DIR / filename
        result = download(url, dest, min_size, description)
        if result is True:
            downloaded += 1
        elif result is False and dest.exists():
            skipped += 1
        else:
            failed += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed.")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
