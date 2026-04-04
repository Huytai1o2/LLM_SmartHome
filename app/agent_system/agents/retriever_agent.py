"""
Managed Retriever Agent

Equipped with two tools:
  - RetrieverTool           : semantic search over static IoT knowledge
                              (device registry, device_sensor_types, rules, demonstrations).
  - ConversationHistoryTool : semantic search over async-embedded conversation history.

Purpose: look up device_id and sensor_type from the knowledge base for a given query,
so the manager can pass that info to smart_home_agent to fetch the live reading.

Uses ToolCallingAgent (JSON format) for structured tool invocation.
"""

from smolagents import ToolCallingAgent

from app.agent_system.model import model
from app.agent_system.tools.retriever_tools import (
    huggingface_doc_retriever_tool,
    conversation_history_tool,
)

_RETRIEVER_INSTRUCTIONS = """
You are a knowledge assistant for an IoT smart home system.
Answer questions from the knowledge base only. Do NOT invent sensor values or readings.

## General knowledge questions (automation rules, device specs, capabilities):
Search the knowledge base with `huggingface_doc_retriever` and return the relevant information.

## Conversation history questions ("what did I ask before?", "previous sessions"):
Use `conversation_history_retriever`.
"""

retriever_agent = ToolCallingAgent(
    tools=[
        huggingface_doc_retriever_tool,  # static knowledge (FAISS)
        conversation_history_tool,  # conversation history (async-embedded FAISS)
    ],
    model=model,
    max_steps=3,
    verbosity_level=1,
    stream_outputs=True,
    name="retriever_agent",
    description=(
        "Answers general knowledge questions about the smart home: automation rules, "
        "device capabilities, specifications, and conversation history lookups. "
        "Does NOT handle sensor readings or control commands — use smart_home_agent for those."
    ),
    instructions=_RETRIEVER_INSTRUCTIONS,
)
