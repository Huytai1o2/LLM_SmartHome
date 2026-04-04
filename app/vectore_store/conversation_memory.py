"""VectorStore-Backed Conversation Memory.

Uses LangChain's ``VectorStoreRetrieverMemory`` to asynchronously embed and
persist conversation history in a dedicated FAISS vector store. This lets the
retriever agent query semantically similar past exchanges when answering a new
question.

Architecture role: "Async embedding — conversation history" in the RAG store.

How it works
------------
1. After every chat turn the caller runs ``async_save_conversation()``.
2. That coroutine delegates the heavy embedding work to a thread-pool executor
   so the FastAPI event loop is never blocked.
3. The background thread calls ``VectorStoreRetrieverMemory.save_context()``,
   which embeds the turn and adds it to the FAISS index.
4. The updated index is flushed to ``CONVERSATION_MEMORY_PATH`` on disk so it
   survives restarts.
5. At retrieval time, ``load_conversation_context()`` calls
   ``memory.load_memory_variables()`` which does a similarity search and
   returns the *k* most relevant past turns as formatted text.

Typical usage
-------------
    # In chat router — after every turn:
    from app.vectore_store.conversation_memory import async_save_conversation

    await async_save_conversation(
        inputs={"input": user_message},
        outputs={"output": assistant_reply},
    )

    # In retrieval tool:
    from app.vectore_store.conversation_memory import load_conversation_context

    context = load_conversation_context(query="What temperature did I set last time?")
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict

from langchain_classic.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.vectore_store.embeddings import get_embeddings

logger = logging.getLogger(__name__)

CONVERSATION_MEMORY_PATH: str = os.environ.get(
    "CONVERSATION_MEMORY_PATH", "faiss_index/conversation_memory"
)

# Module-level singletons (lazy-initialised)
_memory_store: FAISS | None = None
_memory: VectorStoreRetrieverMemory | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_or_create_store() -> FAISS:
    """Return the FAISS store for conversation history (lazy singleton)."""
    global _memory_store

    if _memory_store is not None:
        return _memory_store

    embeddings = get_embeddings()
    index_file = os.path.join(CONVERSATION_MEMORY_PATH, "index.faiss")

    if os.path.isfile(index_file):
        logger.info(
            "Loading conversation memory store from '%s'.", CONVERSATION_MEMORY_PATH
        )
        _memory_store = FAISS.load_local(
            CONVERSATION_MEMORY_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        logger.info("Initialising new in-memory conversation history store.")
        # Bootstrap with a single placeholder document so the store is valid
        _memory_store = FAISS.from_documents(
            [
                Document(
                    page_content="Conversation history store initialised.",
                    metadata={"type": "init"},
                )
            ],
            embedding=embeddings,
        )

    return _memory_store


def get_conversation_memory() -> VectorStoreRetrieverMemory:
    """Return the shared ``VectorStoreRetrieverMemory`` (lazy singleton)."""
    global _memory

    if _memory is not None:
        return _memory

    store = _get_or_create_store()
    retriever = store.as_retriever(search_kwargs={"k": 5})
    _memory = VectorStoreRetrieverMemory(
        retriever=retriever,
        memory_key="conversation_history",
        return_docs=False,
    )
    return _memory


# ---------------------------------------------------------------------------
# Sync / async save
# ---------------------------------------------------------------------------


def _sync_save_conversation(
    inputs: Dict[str, Any],
    outputs: Dict[str, str],
) -> None:
    """Embed one conversation turn and persist the updated index to disk."""
    try:
        memory = get_conversation_memory()
        memory.save_context(inputs, outputs)

        # Persist the updated index so it survives restarts
        store = _get_or_create_store()
        os.makedirs(CONVERSATION_MEMORY_PATH, exist_ok=True)
        store.save_local(CONVERSATION_MEMORY_PATH)
        logger.debug(
            "Conversation turn embedded and saved to '%s'.", CONVERSATION_MEMORY_PATH
        )
    except Exception as exc:
        # Non-fatal: log and continue so a memory failure never breaks the chat
        logger.error("Failed to save conversation turn to memory store: %s", exc)


async def async_save_conversation(
    inputs: Dict[str, Any],
    outputs: Dict[str, str],
) -> None:
    """
    Asynchronously embed a conversation turn in the background.

    The embedding call is offloaded to the default thread-pool executor so the
    FastAPI event loop is never blocked while waiting for the model inference.

    Parameters
    ----------
    inputs:
        Dict with at least ``{"input": "<user message>"}`` — the format expected
        by ``VectorStoreRetrieverMemory.save_context()``.
    outputs:
        Dict with at least ``{"output": "<assistant reply>"}`` — the
        complementary output side of the conversation turn.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _sync_save_conversation, inputs, outputs)


# ---------------------------------------------------------------------------
# Retrieval helper
# ---------------------------------------------------------------------------


def load_conversation_context(query: str) -> str:
    """
    Retrieve the most semantically relevant past conversation turns for *query*.

    Parameters
    ----------
    query:
        The current user input to match against stored conversation history.

    Returns
    -------
    str
        Formatted string containing up to *k* relevant past conversation turns,
        or a message indicating no history is available.
    """
    memory = get_conversation_memory()
    variables = memory.load_memory_variables({"prompt": query})
    history: str = variables.get("conversation_history", "")

    if not history or history.strip() == "Conversation history store initialised.":
        return "No relevant conversation history found."

    return f"Relevant past conversations:\n{history}"
