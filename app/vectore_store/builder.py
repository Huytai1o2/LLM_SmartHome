"""Vector store builder.

Loads raw documents, splits them into chunks, embeds them with the shared
embedding model, and persists the resulting FAISS index to disk.

Typical usage
-------------
    from app.vectore_store.builder import build_and_save

    build_and_save()           # builds from HuggingFace dataset
    build_and_save(docs=my_docs)  # builds from provided LangChain Documents
"""

from __future__ import annotations

import logging
import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from transformers import AutoTokenizer

from app.vectore_store.embeddings import EMBEDDING_MODEL_NAME, get_embeddings
from knowledge_base.sources import load_documents

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH: str = os.environ.get("FAISS_INDEX_PATH", "faiss_index")
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "200"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "20"))


def _split_documents(source_docs: List[Document]) -> List[Document]:
    """Split documents into chunks, removing duplicates."""
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME),
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    logger.info("Splitting %d documents...", len(source_docs))
    processed: List[Document] = []
    seen: dict[str, bool] = {}
    for doc in tqdm(source_docs, desc="Splitting"):
        for chunk in text_splitter.split_documents([doc]):
            if chunk.page_content not in seen:
                seen[chunk.page_content] = True
                processed.append(chunk)

    logger.info("Produced %d unique chunks.", len(processed))
    return processed


def build_and_save(docs: List[Document] | None = None) -> FAISS:
    """
    Build a FAISS vector store from documents and persist it to disk.

    Parameters
    ----------
    docs:
        Optional list of LangChain Documents to index. When omitted, the
        HuggingFace docs dataset is downloaded automatically.

    Returns
    -------
    FAISS
        The in-memory vector store that was saved to FAISS_INDEX_PATH.
    """
    if docs is None:
        docs = load_documents()

    chunks = _split_documents(docs)

    logger.info("Embedding %d chunks...", len(chunks))
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        distance_strategy=DistanceStrategy.COSINE,
    )

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vector_store.save_local(FAISS_INDEX_PATH)
    logger.info("FAISS index saved to '%s'.", FAISS_INDEX_PATH)

    return vector_store
