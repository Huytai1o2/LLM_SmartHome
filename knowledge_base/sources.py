"""Document sources for the IoT Smart Home RAG knowledge base.

This is the single place to define what gets indexed into the static FAISS
vector store. Add, remove, or replace sources here — the builder will pick
them up automatically on the next run.

Static knowledge (embedded once)
---------------------------------
These four sources map directly to the "Static — embed once" tier in the
architecture diagram:

- ``device_registry.txt``   — all registered smart-home devices and their capabilities
- ``sensor_knowledge.txt``  — sensor types, measurement ranges, and interpretation rules
- ``rules.txt``             — automation rules (triggers, conditions, actions)
- ``demonstration.txt``     — worked examples of agent commands and responses

Note: Sensor logs (CSV) and conversation history are handled separately —
they are NOT embedded here:
  • Sensor logs   → read live by ``app.vectore_store.sensor_logs``
  • Conv. history → asynchronously embedded by ``app.vectore_store.conversation_memory``

Typical usage
-------------
    from knowledge_base.sources import load_documents

    docs = load_documents()
"""

from __future__ import annotations

import logging
from typing import List

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

IOT_KNOWLEDGE_PATH: str = "knowledge_base/iot_knowledge"

# Map each knowledge file to a human-readable source tag used in metadata
_STATIC_SOURCES: dict[str, str] = {
    "device_registry.txt": "device_registry",
    "sensor_knowledge.txt": "sensor_knowledge",
    "rules.txt": "rules",
    "demonstration.txt": "demonstration",
    "device_sensor_types.txt": "device_sensor_types",
}


def _load_iot_static() -> List[Document]:
    """Load the four static IoT knowledge files and return LangChain Documents."""
    import os

    docs: List[Document] = []
    for filename, source_tag in _STATIC_SOURCES.items():
        filepath = os.path.join(IOT_KNOWLEDGE_PATH, filename)
        if not os.path.isfile(filepath):
            logger.warning("Static knowledge file not found, skipping: '%s'", filepath)
            continue
        loader = TextLoader(filepath, encoding="utf-8")
        file_docs = loader.load()
        for doc in file_docs:
            doc.metadata["source"] = source_tag
        docs.extend(file_docs)
        logger.info("Loaded %d document(s) from '%s'.", len(file_docs), filename)

    return docs


def load_documents() -> List[Document]:
    """
    Return all LangChain Documents to be indexed into the static FAISS store.

    Called by the vector store builder. Extend this function to add more
    static knowledge sources.
    """
    docs: List[Document] = []

    # --- IoT static knowledge (device registry, sensor knowledge, rules, demonstrations) ---
    docs.extend(_load_iot_static())

    logger.info("Total static documents collected: %d", len(docs))
    return docs
