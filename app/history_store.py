"""
app/history_store.py

Durable conversation + feedback store (ISSUE-009, decision D3).

Two append-only JSONL files live under ``$CHAT_LOG_DIR`` (default
``data/chat_history/``, already gitignored):

  chat_history.jsonl   one record per completed Q&A turn
  feedback.jsonl       one record per 👍/👎 click, linked by ``entry_id``

Both are logs, never rewritten. An interrupted write can at worst leave a torn
final line, which the readers skip; feedback is reduced to the newest click per
entry on read, so the UI always shows the current verdict while the click
history stays available for later analysis.

Records are JSON, so retrieved ``Document`` objects are stored as
``{"page_content", "metadata"}`` and rebuilt on load — the rest of the app can
keep assuming ``chat["docs"]`` holds LangChain documents.
"""
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.observability.store import append_jsonl

DEFAULT_LOG_DIR = "data/chat_history"
HISTORY_FILE = "chat_history.jsonl"
FEEDBACK_FILE = "feedback.jsonl"


def log_dir() -> str:
    """Directory holding the log files (overridable for tests / deployments)."""
    return os.getenv("CHAT_LOG_DIR") or DEFAULT_LOG_DIR


def _path(filename: str) -> str:
    return os.path.join(log_dir(), filename)


def new_entry_id() -> str:
    """Stable identifier for one conversation turn.

    Feedback is linked by this id rather than by question text: the same
    question may legitimately be asked (and rated) more than once.
    """
    return uuid.uuid4().hex


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Parse a JSONL log, tolerating blank and truncated (torn) lines."""
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _serialize_doc(doc: Any) -> Dict[str, Any]:
    """Reduce a Document (or an already-plain doc dict) to JSON-safe fields."""
    if isinstance(doc, dict):
        return {
            "page_content": doc.get("page_content", doc.get("text", "")) or "",
            "metadata": doc.get("metadata") or {},
        }
    return {
        "page_content": getattr(doc, "page_content", "") or "",
        "metadata": dict(getattr(doc, "metadata", None) or {}),
    }


def _deserialize_doc(raw: Any) -> Optional[Document]:
    if isinstance(raw, Document):
        return raw
    if not isinstance(raw, dict):
        return None
    metadata = raw.get("metadata")
    return Document(
        page_content=raw.get("page_content", "") or "",
        metadata=metadata if isinstance(metadata, dict) else {},
    )


# ---------------------------------------------------------------------------
# Conversation turns
# ---------------------------------------------------------------------------

def append_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Persist one Q&A turn and return the record as written.

    The returned record carries the ``id`` used for feedback, so callers can
    render the turn and link its thumbs to the same record.
    """
    record: Dict[str, Any] = {k: v for k, v in entry.items() if k != "docs"}
    record["id"] = record.get("id") or new_entry_id()
    record.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    record["docs"] = [_serialize_doc(d) for d in (entry.get("docs") or [])]
    append_jsonl(_path(HISTORY_FILE), record)
    return record


def load_entries() -> List[Dict[str, Any]]:
    """All persisted turns, oldest first, with ``docs`` back as Documents."""
    entries: List[Dict[str, Any]] = []
    for index, row in enumerate(_read_jsonl(_path(HISTORY_FILE))):
        entry = dict(row)
        # Rows written by this module always carry an id. A hand-edited row
        # falls back to its position — stable for an append-only log — so
        # feedback linked to it still resolves after a reload.
        entry["id"] = entry.get("id") or f"legacy_{index}"
        entry["docs"] = [
            doc
            for doc in (_deserialize_doc(raw) for raw in (row.get("docs") or []))
            if doc is not None
        ]
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def append_feedback(
    entry_id: str,
    helpful: bool,
    *,
    question: str = "",
    collection: str = "",
) -> Dict[str, Any]:
    """Record one 👍/👎 click against a conversation turn."""
    record = {
        "entry_id": entry_id,
        "helpful": bool(helpful),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "collection": collection,
    }
    append_jsonl(_path(FEEDBACK_FILE), record)
    return record


def load_feedback() -> Dict[str, Dict[str, Any]]:
    """Map ``entry_id`` → the newest feedback record for that turn."""
    latest: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(_path(FEEDBACK_FILE)):
        entry_id = row.get("entry_id")
        if entry_id:
            latest[str(entry_id)] = row
    return latest


def feedback_summary(feedback: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, int]:
    """Count rated turns, newest click per turn (a change of mind doesn't count twice)."""
    rows = load_feedback() if feedback is None else feedback
    helpful = sum(1 for row in rows.values() if row.get("helpful"))
    return {"helpful": helpful, "not_helpful": len(rows) - helpful}


def clear_log() -> None:
    """Delete both logs (used by the sidebar's "Clear Chat")."""
    for name in (HISTORY_FILE, FEEDBACK_FILE):
        path = _path(name)
        if os.path.exists(path):
            os.remove(path)
