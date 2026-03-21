"""
Retriever tool backed by the shared FAISS vector store.

The vector store is managed by app.vectore_store.store and loaded lazily on
first query — either from the persisted faiss_index/ on disk, or built from
scratch via the builder if no index exists yet.
"""

from __future__ import annotations

import logging

from smolagents import Tool

from app.vectore_store.store import get_vector_store

logger = logging.getLogger(__name__)


class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Using semantic similarity, retrieves some documents from the knowledge base "
        "that have the closest embeddings to the input query."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "The query to perform. This should be semantically close to your target "
                "documents. Use the affirmative form rather than a question."
            ),
        }
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = get_vector_store().similarity_search(query, k=3)
        return "\nRetrieved documents:\n" + "".join(
            f"===== Document {i} =====\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )


huggingface_doc_retriever_tool = RetrieverTool()
