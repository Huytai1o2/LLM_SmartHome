"""Shared vector store singleton.

Provides a single ``get_vector_store()`` entry point used by retriever tools.
On first call it tries to load a persisted index from disk; if none exists it
falls back to building one from scratch.

The loaded store is cached in memory so subsequent calls are instant.

Typical usage
-------------
    from app.vectore_store.store import get_vector_store

    vs = get_vector_store()
    results = vs.similarity_search("What is PEFT?", k=5)
"""

from __future__ import annotations

import logging

from langchain_community.vectorstores import FAISS

from app.vectore_store.builder import build_and_save
from app.vectore_store.loader import index_exists, load_vector_store

logger = logging.getLogger(__name__)

_store: FAISS | None = None


def get_vector_store() -> FAISS:
    """
    Return the shared FAISS vector store (lazy singleton).

    Load order:
    1. Return cached in-memory instance if already initialised.
    2. Load from local disk if ``faiss_index/`` exists.
    3. Build from scratch (downloads HuggingFace dataset) and save to disk.
    """
    global _store

    if _store is not None:
        return _store

    if index_exists():
        _store = load_vector_store()
    else:
        logger.warning(
            "No local FAISS index found — building from scratch. "
            "This may take a few minutes."
        )
        _store = build_and_save()

    return _store


def reset_vector_store() -> None:
    """Clear the in-memory cache (useful for testing or after re-indexing)."""
    global _store
    _store = None
