"""Document sources for the RAG knowledge base.

This is the single place to define what gets indexed into the vector store.
Add, remove, or replace sources here — the builder will pick them up
automatically on the next run.

Current sources
---------------
- HuggingFace docs dataset (``m-ric/huggingface_doc``)

To add your own files later, load them here and extend the returned list::

    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    dir_loader = DirectoryLoader("knowledge_base/files/", glob="**/*.txt",
                                 loader_cls=TextLoader)
    local_docs = dir_loader.load()
    return hf_docs + local_docs

Typical usage
-------------
    from knowledge_base.sources import load_documents

    docs = load_documents()
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _load_hf_dataset() -> List[Document]:
    """Download and return documents from the HuggingFace docs dataset."""
    import datasets

    logger.info("Loading HuggingFace docs dataset (m-ric/huggingface_doc)...")
    knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
    docs = [
        Document(
            page_content=doc["text"],
            metadata={"source": doc["source"].split("/")[1]},
        )
        for doc in knowledge_base
    ]
    logger.info("Loaded %d documents from HuggingFace dataset.", len(docs))
    return docs


def load_documents() -> List[Document]:
    """
    Return the full list of LangChain Documents to be indexed.

    This is the entrypoint called by the vector store builder. Extend this
    function when you want to add new document sources.
    """
    docs: List[Document] = []

    # --- Source 1: HuggingFace docs dataset (replace with your own sources) ---
    docs.extend(_load_hf_dataset())

    # --- Source 2: Local files (uncomment and adjust when ready) -------------
    # from langchain_community.document_loaders import DirectoryLoader, TextLoader
    # from langchain_community.document_loaders import PyPDFLoader
    # from langchain_community.document_loaders import UnstructuredMarkdownLoader
    # loaders = [
    #     DirectoryLoader("knowledge_base/files/", glob="**/*.txt",  loader_cls=TextLoader),
    #     DirectoryLoader("knowledge_base/files/", glob="**/*.pdf",  loader_cls=PyPDFLoader),
    #     DirectoryLoader("knowledge_base/files/", glob="**/*.md",   loader_cls=UnstructuredMarkdownLoader),
    # ]
    # for loader in loaders:
    #     docs.extend(loader.load())

    logger.info("Total documents collected: %d", len(docs))
    return docs
