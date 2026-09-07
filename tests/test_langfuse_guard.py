"""
tests/test_langfuse_guard.py

Unit tests for src/observability/langfuse_guard.py.

Tests verify behaviour in BOTH the enabled and disabled paths without
requiring real Langfuse keys or network access. Monkeypatching is used to
simulate env var presence/absence.
"""

import importlib
import sys
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_guard(public_key: str = "", secret_key: str = ""):
    """
    Reload langfuse_guard with controlled env vars.
    Returns the freshly-imported module so tests can inspect its state.
    """
    env = {
        "LANGFUSE_PUBLIC_KEY": public_key,
        "LANGFUSE_SECRET_KEY": secret_key,
    }
    # Remove cached module so importlib.import_module re-executes it
    mod_name = "src.observability.langfuse_guard"
    sys.modules.pop(mod_name, None)

    with patch.dict("os.environ", env, clear=False):
        mod = importlib.import_module(mod_name)
    return mod


# ---------------------------------------------------------------------------
# No-op path (keys absent)
# ---------------------------------------------------------------------------

class TestNoOpPath:
    """Verify that all exports are no-ops when keys are not set."""

    def setup_method(self):
        self.guard = _reload_guard(public_key="", secret_key="")

    def test_langfuse_enabled_is_false(self):
        assert self.guard.LANGFUSE_ENABLED is False

    def test_observe_returns_function_unchanged(self):
        def my_fn():
            return 42
        decorated = self.guard.observe()(my_fn)
        assert decorated is my_fn
        assert decorated() == 42

    def test_observe_bare_decorator_style(self):
        """@observe (without parentheses) must also work."""
        def my_fn():
            return "hello"
        result = self.guard.observe(my_fn)
        assert result is my_fn

    def test_callback_handler_instantiates_without_error(self):
        handler = self.guard.CallbackHandler()
        assert handler is not None

    def test_callback_handler_accepts_arbitrary_args(self):
        # Should not raise even with unknown keyword args
        handler = self.guard.CallbackHandler(session_id="abc", user_id="xyz")
        assert handler is not None

    def test_langfuse_flush_is_callable_no_error(self):
        # Should silently do nothing
        self.guard.langfuse_flush()

    def test_observe_decorated_function_runs_normally(self):
        @self.guard.observe()
        def add(a, b):
            return a + b
        assert add(2, 3) == 5


# ---------------------------------------------------------------------------
# Enabled path (keys present, langfuse mocked)
# ---------------------------------------------------------------------------

class TestEnabledPath:
    """When keys are present, guard should use real langfuse symbols."""

    def test_langfuse_enabled_true_when_both_keys_set(self):
        # Mock langfuse at the package level so no network calls happen
        mock_langfuse_module = MagicMock()
        mock_langfuse_module.observe = lambda *a, **kw: (a[0] if a and callable(a[0]) else lambda fn: fn)
        mock_lf_langchain = MagicMock()

        with patch.dict(sys.modules, {
            "langfuse": mock_langfuse_module,
            "langfuse.langchain": mock_lf_langchain,
        }):
            guard = _reload_guard(public_key="pk-live-xxx", secret_key="sk-live-yyy")

        assert guard.LANGFUSE_ENABLED is True

    def test_no_keys_means_disabled_even_if_langfuse_installed(self):
        mock_langfuse_module = MagicMock()
        with patch.dict(sys.modules, {"langfuse": mock_langfuse_module, "langfuse.langchain": MagicMock()}):
            guard = _reload_guard(public_key="", secret_key="")
        assert guard.LANGFUSE_ENABLED is False

    def test_only_public_key_not_enough(self):
        mock_langfuse_module = MagicMock()
        with patch.dict(sys.modules, {"langfuse": mock_langfuse_module, "langfuse.langchain": MagicMock()}):
            guard = _reload_guard(public_key="pk-xxx", secret_key="")
        assert guard.LANGFUSE_ENABLED is False

    def test_only_secret_key_not_enough(self):
        mock_langfuse_module = MagicMock()
        with patch.dict(sys.modules, {"langfuse": mock_langfuse_module, "langfuse.langchain": MagicMock()}):
            guard = _reload_guard(public_key="", secret_key="sk-yyy")
        assert guard.LANGFUSE_ENABLED is False


# ---------------------------------------------------------------------------
# Guard is the single import point — direct langfuse imports removed
# ---------------------------------------------------------------------------

class TestImportSurface:
    """Verify that the main source files no longer directly import langfuse."""

    def _file_contains_direct_import(self, path: str) -> bool:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # We allow 'langfuse' to appear in comments/strings but not as
        # a bare `from langfuse import` or `import langfuse` statement.
        import re
        return bool(re.search(r"^\s*(?:from langfuse|import langfuse)\b", content, re.MULTILINE))

    def test_retriever_has_no_direct_langfuse_import(self):
        assert not self._file_contains_direct_import(
            "src/rag/retriever.py"
        )

    def test_evaluator_has_no_direct_langfuse_import(self):
        assert not self._file_contains_direct_import(
            "src/eval/evaluator.py"
        )

    def test_chat_page_has_no_direct_langfuse_import(self):
        assert not self._file_contains_direct_import(
            "app/pages/chat.py"
        )
