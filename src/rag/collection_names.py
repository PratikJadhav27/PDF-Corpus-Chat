"""Single source of truth for collection names (ISSUE-010).

A collection name becomes a directory under ``vector_stores/`` and
``metadata/``, so it must never be able to leave those trees. Only
``[A-Za-z0-9_-]`` survives; every other run of characters collapses to ``_``.
That makes ``../../etc`` resolve to ``etc`` inside the store instead of the
parent of the store.

Names that already exist on disk are used verbatim. They were created by an
older build that only replaced spaces, so a legacy ``paper.v2`` directory must
keep working (and keep its Qdrant collection name) rather than silently
vanishing behind a ``paper_v2`` that no longer exists. Entries returned by
``os.listdir`` are single path components, so accepting one cannot escape the
base directory.

Resolution always consults ``vector_stores/`` — the canonical set of
collections — so ``metadata/`` and the vector store can never disagree about
what a name means.
"""
import os
import re

VECTOR_STORES_DIR = "vector_stores"

_UNSAFE_RUN = re.compile(r"[^A-Za-z0-9_-]+")


def sanitize_collection_name(name: str) -> str:
    """Reduce ``name`` to a safe, single path component.

    Spaces become underscores (the behaviour the UI has always had), everything
    outside the allowlist collapses to ``_``, and an empty result falls back to
    ``"default"``.
    """
    safe = _UNSAFE_RUN.sub("_", (name or "").strip()).strip("_-")
    return safe or "default"


def _existing_dirs() -> set:
    try:
        return {
            d
            for d in os.listdir(VECTOR_STORES_DIR)
            if os.path.isdir(os.path.join(VECTOR_STORES_DIR, d))
        }
    except OSError:
        return set()


def resolve_collection_name(name: str) -> str:
    """Name to actually use on disk for ``name``.

    An existing directory wins (legacy names keep working); otherwise the name
    is sanitised.
    """
    raw = (name or "").strip()
    if raw in _existing_dirs():
        return raw
    return sanitize_collection_name(raw)


def contained_path(base_dir: str, component: str) -> str:
    """Join an already-resolved name under ``base_dir``, asserting containment.

    Use this once a name has been resolved, so a second lookup cannot change
    what the name means (deleting the store directory would otherwise make
    ``resolve_collection_name`` sanitise a legacy name differently and point
    the ``metadata/`` path somewhere else).
    """
    base_abs = os.path.abspath(base_dir)
    path_abs = os.path.abspath(os.path.join(base_abs, component))
    if os.path.commonpath([base_abs, path_abs]) != base_abs:
        raise ValueError(f"Unsafe collection name: {component!r}")
    return os.path.join(base_dir, component)


def collection_path(base_dir: str, name: str) -> str:
    """Path for collection ``name`` under ``base_dir``, guaranteed contained."""
    return contained_path(base_dir, resolve_collection_name(name))
