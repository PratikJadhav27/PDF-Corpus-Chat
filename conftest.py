# conftest.py — ensures the project root is on sys.path for all pytest runs
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
