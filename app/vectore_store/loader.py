"""Vector store loader.

Loads a previously built FAISS index from local disk.

Typical usage
-------------
    from app.vectore_store.loader import load_vector_store

    vs = load_vector_store()             # loads from default FAISS_INDEX_PATH
    vs = load_vector_store("/my/path")   # loads from a custom directory
"""

from __future__ import annotations

import logging
import os

from langchain_community.vectorstores import FAISS

from app.vectore_store.embeddings import get_embeddings

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH: str = os.environ.get("FAISS_INDEX_PATH", "faiss_index")


def index_exists(path: str = FAISS_INDEX_PATH) -> bool:
    """Return True if a persisted FAISS index exists at *path*."""
    return os.path.isfile(os.path.join(path, "index.faiss"))


def load_vector_store(path: str = FAISS_INDEX_PATH) -> FAISS:
    """
    Load and return a FAISS vector store from a local directory.

    Parameters
    ----------
    path:
        Directory that contains ``index.faiss`` and ``index.pkl``.
        Defaults to the ``FAISS_INDEX_PATH`` environment variable
        or ``faiss_index/`` at the project root.

    Raises
    ------
    FileNotFoundError
        If the expected index files are not found at *path*.
    """
    if not index_exists(path):
        raise FileNotFoundError(
            f"No FAISS index found at '{path}'. "
            "Run the builder first: "
            "`from app.vectore_store.builder import build_and_save; build_and_save()`"
        )

    logger.info("Loading FAISS index from '%s'...", path)
    vector_store = FAISS.load_local(
        path,
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS index loaded successfully.")
    return vector_store
