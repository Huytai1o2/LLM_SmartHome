"""
Managed Retriever Agent

Equipped with two RAG store tools:
  - RetrieverTool             : semantic search over static IoT knowledge
                                (device registry, sensor knowledge, rules, demonstrations).
  - ConversationHistoryTool   : semantic search over async-embedded conversation history.

Uses ToolCallingAgent (JSON format) for structured tool invocation.
"""

from smolagents import ToolCallingAgent

from app.agent_system.model import model
from app.agent_system.tools.retriever_tools import (
    huggingface_doc_retriever_tool,
    conversation_history_tool,
)

retriever_agent = ToolCallingAgent(
    tools=[
        huggingface_doc_retriever_tool,  # static knowledge (FAISS)
        conversation_history_tool,  # conversation history (async-embedded FAISS)
    ],
    model=model,
    max_steps=3,  # retrieve from one or more sources, then summarise
    verbosity_level=1,
    stream_outputs=True,
    name="retriever_agent",
    description=(
        "Retrieves information from the IoT smart home knowledge base. "
        "Can query static knowledge (device registry, sensor knowledge, rules, demonstrations) "
        "and past conversation history. "
        "Pass a natural language query describing what information you need."
    ),
)
