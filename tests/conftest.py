"""
tests/conftest.py

Shared test isolation.

The chat log / feedback store (ISSUE-009) defaults to ``data/chat_history/``
in the working copy. Without redirecting it, every AppTest run would read the
developer's real conversation log — making results depend on local state — and
a test that clicks 👍/👎 would append to real feedback data.

Qdrant local storage allows exactly one client per folder per process
(ISSUE-011 caches that client), so the cached clients are closed after each
test: otherwise a test's temporary store stays locked until the process exits
and Windows refuses to clean the tmp directory up.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolated_chat_log(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_LOG_DIR", str(tmp_path / "chat_log"))


@pytest.fixture(autouse=True)
def _release_shared_qdrant_clients():
    yield
    from src.rag.vector_store import close_all_qdrant_clients

    close_all_qdrant_clients()
