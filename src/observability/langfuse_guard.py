"""
src/observability/langfuse_guard.py

Conditional Langfuse initialisation.

Problem
-------
Langfuse logs an AUTH ERROR to stderr every time `Langfuse()` or
`CallbackHandler()` is created without LANGFUSE_PUBLIC_KEY.  In CI this
floods the output with 50+ identical lines per eval run, obscuring real
errors and making logs unreadable.

Solution
--------
This module exports the same symbols that consumers need
(`observe`, `CallbackHandler`, `langfuse_flush`) but only activates real
Langfuse when BOTH required env vars are present:

    LANGFUSE_PUBLIC_KEY   — Langfuse project public key
    LANGFUSE_SECRET_KEY   — Langfuse project secret key

When keys are absent every exported symbol is a no-op:
  - `observe`           — identity decorator (function is returned unchanged)
  - `CallbackHandler`   — stub class whose instances silently do nothing
  - `langfuse_flush`    — no-op function

Import pattern
--------------
Instead of:
    from langfuse import observe, Langfuse
    from langfuse.langchain import CallbackHandler

Use:
    from src.observability.langfuse_guard import observe, CallbackHandler, langfuse_flush

The guard is the ONLY place in the codebase that directly imports langfuse,
which makes it easy to test and easy to swap out later.
"""

import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Check whether both required keys are available
# ---------------------------------------------------------------------------

_PUBLIC_KEY  = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
_SECRET_KEY  = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
LANGFUSE_ENABLED = bool(_PUBLIC_KEY and _SECRET_KEY)


# ---------------------------------------------------------------------------
# Live path — real Langfuse symbols
# ---------------------------------------------------------------------------

if LANGFUSE_ENABLED:
    try:
        from langfuse import observe as _real_observe          # type: ignore
        from langfuse import Langfuse as _Langfuse             # type: ignore
        from langfuse.langchain import CallbackHandler         # type: ignore

        observe = _real_observe

        def langfuse_flush() -> None:
            """Flush buffered Langfuse events to the server."""
            try:
                _Langfuse().flush()
            except Exception as exc:
                logger.debug("Langfuse flush failed: %s", exc)

        logger.info("Langfuse enabled (public_key configured).")

    except ImportError:
        # langfuse not installed — fall through to no-op path
        LANGFUSE_ENABLED = False


# ---------------------------------------------------------------------------
# No-op path — when keys are absent or langfuse not installed
# ---------------------------------------------------------------------------

if not LANGFUSE_ENABLED:
    def observe(*args: Any, **kwargs: Any) -> Any:          # type: ignore[misc]
        """No-op decorator — returns the wrapped function unchanged."""
        # Handles both @observe  and  @observe()
        if args and callable(args[0]):
            return args[0]
        def decorator(fn: Callable) -> Callable:
            return fn
        return decorator

    class CallbackHandler:                                   # type: ignore[no-redef]
        """Stub that silently absorbs any construction args and does nothing."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    def langfuse_flush() -> None:                           # type: ignore[misc]
        """No-op flush when Langfuse is not configured."""
        pass
