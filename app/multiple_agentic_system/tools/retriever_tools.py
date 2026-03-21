"""
Retriever tools backed by FAISS vector stores.

- RetrieverTool : searches the indexed HuggingFace documentation.

The knowledge base is built LAZILY on first query and cached in memory.
Import is fast; startup time is unaffected.
"""

from __future__ import annotations

import logging
from smolagents import Tool
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)

# Module-level cache – populated on first forward() call
_vector_db: VectorStore | None = None


def _build_vector_db() -> VectorStore:
    """Download dataset, split, embed, and return FAISS index."""
    import datasets
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores.utils import DistanceStrategy

    logger.info("Loading HuggingFace docs dataset...")
    knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

    source_docs = [
        Document(
            page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]}
        )
        for doc in knowledge_base
    ]

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained("thenlper/gte-small"),
        chunk_size=200,
        chunk_overlap=20,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    logger.info("Splitting documents...")
    docs_processed = []
    unique_texts: dict[str, bool] = {}
    for doc in tqdm(source_docs, desc="Splitting"):
        for new_doc in text_splitter.split_documents([doc]):
            if new_doc.page_content not in unique_texts:
                unique_texts[new_doc.page_content] = True
                docs_processed.append(new_doc)

    logger.info("Embedding %d document chunks...", len(docs_processed))
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    vector_db = FAISS.from_documents(
        documents=docs_processed,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )
    logger.info("FAISS index ready (%d chunks).", len(docs_processed))
    return vector_db


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
        global _vector_db
        assert isinstance(query, str), "Your search query must be a string"

        # Lazy build on first call
        if _vector_db is None:
            _vector_db = _build_vector_db()

        docs = _vector_db.similarity_search(query, k=7)
        return "\nRetrieved documents:\n" + "".join(
            f"===== Document {i} =====\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )


huggingface_doc_retriever_tool = RetrieverTool()
