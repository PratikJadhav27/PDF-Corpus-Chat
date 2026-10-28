"""
tests/conftest.py

Shared test isolation.

The chat log / feedback store (ISSUE-009) defaults to ``data/chat_history/``
in the working copy. Without redirecting it, every AppTest run would read the
developer's real conversation log — making results depend on local state — and
a test that clicks 👍/👎 would append to real feedback data.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolated_chat_log(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_LOG_DIR", str(tmp_path / "chat_log"))
