"""
Managed Retriever Agent

Equipped with:
  - RetrieverToolHFDocs    : searches the indexed HuggingFace documentation.
  - RetrieverToolPEFTIssues: searches the indexed PEFT GitHub issues knowledge base.

Both tools use FAISS vector stores with sentence-transformer embeddings.
"""

from smolagents import ToolCallingAgent

from app.agent_system.model import model
from app.agent_system.tools.retriever_tools import (
    huggingface_doc_retriever_tool,
)

retriever_agent = ToolCallingAgent(
    tools=[huggingface_doc_retriever_tool],
    model=model,
    max_steps=2,  # retrieve once, summarise once
    verbosity_level=1,
    name="retriever_agent",
    description="Retrieves documents from the knowledge base for you that are close to the input query. Give it your query as an argument. The knowledge base includes Hugging Face documentation and PEFT issues.",
)
