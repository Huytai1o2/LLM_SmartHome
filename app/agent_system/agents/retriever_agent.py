"""
Managed Retriever Agent

Equipped with:
  - RetrieverToolHFDocs    : searches the indexed HuggingFace documentation.
  - RetrieverToolPEFTIssues: searches the indexed PEFT GitHub issues knowledge base.

Both tools use FAISS vector stores with sentence-transformer embeddings.

Uses CodeAgent (Python code format) instead of ToolCallingAgent (JSON format)
so that small local models like qwen3:1.7b can reliably call tools without
failing to produce strict JSON blobs.
"""

from smolagents import CodeAgent

from app.agent_system.model import model
from app.agent_system.tools.retriever_tools import (
    huggingface_doc_retriever_tool,
)

retriever_agent = CodeAgent(
    tools=[huggingface_doc_retriever_tool],
    model=model,
    max_steps=2,  # retrieve once, summarise once
    verbosity_level=1,
    stream_outputs=True,  # stream token deltas as they arrive
    name="retriever_agent",
    description="Retrieves documents from the knowledge base for you that are close to the input query. Give it your query as an argument. The knowledge base includes Hugging Face documentation and PEFT issues.",
)
