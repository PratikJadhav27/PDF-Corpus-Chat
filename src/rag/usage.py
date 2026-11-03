"""Token accounting for LLM replies (ISSUE-013).

The Collections tab's "Usage & Cost" panel reads ``st.session_state.token_count``
and turns it into an estimated price, but nothing ever wrote that key — every
session showed "Total Tokens 0" and "$0.000000" no matter how many questions
were asked. Every Gemini reply carries its own accounting, so the pipeline now
adds it to a running total.

Deliberately import-free of both ``langchain`` and ``streamlit``: any object
exposing ``usage_metadata`` works, and the caller passes the session state in.
That keeps the accumulator usable from the CLI and testable without a running
app.
"""
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


TOKEN_COUNT_KEY = "token_count"


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported by one model call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return self.total_tokens <= 0


def _as_count(value: Any) -> int:
    """Coerce a reported count to a non-negative int (0 when unusable)."""
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _first(mapping: Dict[str, Any], *keys: str) -> Any:
    """First key present with a non-None value — providers name these differently."""
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _usage_mappings(message: Any) -> Iterable[Dict[str, Any]]:
    """Every place a LangChain message may keep usage counts.

    ``AIMessage.usage_metadata`` is the standard location; ``response_metadata``
    is where provider-specific payloads survive, and older integrations put the
    raw Gemini ``usageMetadata`` / OpenAI-style ``token_usage`` there.
    """
    direct = getattr(message, "usage_metadata", None)
    if isinstance(direct, dict):
        yield direct

    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        for key in ("usage_metadata", "token_usage"):
            nested = response_metadata.get(key)
            if isinstance(nested, dict):
                yield nested


def usage_from_message(message: Any) -> Optional[TokenUsage]:
    """Extract the token counts from one model response, or ``None``.

    ``None`` means "the provider did not report usage" — callers must not treat
    that as a zero-token call, and must not overwrite the running total with it.
    """
    for counts in _usage_mappings(message):
        prompt = _as_count(_first(counts, "input_tokens", "prompt_tokens"))
        completion = _as_count(
            _first(counts, "output_tokens", "completion_tokens")
        )
        total = _as_count(_first(counts, "total_tokens"))
        if total <= 0:
            # Some providers omit the sum; it is always the two parts.
            total = prompt + completion
        if total > 0:
            return TokenUsage(prompt, completion, total)
    return None


def _read_count(state: Any) -> int:
    """Read the running total from a Streamlit session or a plain mapping."""
    try:
        stored = state[TOKEN_COUNT_KEY]
    except (KeyError, TypeError):
        stored = getattr(state, TOKEN_COUNT_KEY, 0)
    return _as_count(stored)


def _write_count(state: Any, value: int) -> None:
    try:
        state[TOKEN_COUNT_KEY] = value
    except (TypeError, AttributeError):
        setattr(state, TOKEN_COUNT_KEY, value)


def accumulate_usage(state: Any, usage: Optional[TokenUsage]) -> int:
    """Add ``usage`` to ``state["token_count"]`` and return the new total.

    A missing/non-numeric stored value counts as 0, and an unreported usage
    leaves the total untouched: a metrics bug must never break the answer the
    call already paid for.
    """
    total = _read_count(state)
    if usage is None or usage.is_empty:
        return total
    total += usage.total_tokens
    _write_count(state, total)
    return total
