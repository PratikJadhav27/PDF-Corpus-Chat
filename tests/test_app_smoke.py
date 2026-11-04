"""ISSUE-012: one CI check that the app actually boots.

The suite has plenty of focused AppTests, but none of them promises that
``app/main.py`` itself starts: a broken import, a missing session default or a
crashing top-level view would surface as a pile of unrelated failures. This
test only answers "does the shell render", so the cause is obvious.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


def test_app_boots_and_renders_its_shell(monkeypatch, tmp_path):
    # Keep the app out of the real chat log / trace sinks.
    monkeypatch.setenv("CHAT_LOG_DIR", str(tmp_path / "chat_log"))
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "0")

    at = AppTest.from_file(str(APP_MAIN), default_timeout=120)
    at.run()

    assert not at.exception, at.exception
    # ``at.tabs`` is flat and interleaves the nested History/Semantic Search
    # tabs, so assert the four shell tabs are all present rather than their
    # position in that flattened list.
    labels = [tab.label for tab in at.tabs]
    for expected in ("💬 Chat", "📊 Analytics", "📜 History", "⚙️ Collections"):
        assert expected in labels, labels
    # Session defaults exist, so no view has to guard for a missing key.
    assert at.session_state["token_count"] == 0
    assert at.session_state["chat_history"] == []
