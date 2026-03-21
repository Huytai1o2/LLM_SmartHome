"""Shared embedding model.

Uses HuggingFace sentence-transformers so no external API key is needed.
The model is instantiated once and reused across the application.

Typical usage
-------------
    from app.vectore_store.embeddings import get_embeddings

    embeddings = get_embeddings()
"""

from __future__ import annotations

import os

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME: str = os.environ.get("EMBEDDING_MODEL_NAME", "thenlper/gte-small")

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared HuggingFaceEmbeddings instance (lazy singleton)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings
