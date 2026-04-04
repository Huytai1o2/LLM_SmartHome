"""
Managed Clarification Agent

Handles ambiguous user queries that are missing required details (e.g. device location).

Equipped with:
  - RetrieverTool             : looks up what devices and locations exist in the smart home.
  - ConversationHistoryTool   : checks past conversation turns to see if the user already
                                mentioned a location earlier in the session.

This lets it give the user concrete options rather than a generic "please specify",
and avoids asking again if the location was already stated before.
"""

from smolagents import ToolCallingAgent

from app.agent_system.model import model
from app.agent_system.tools.retriever_tools import (
    huggingface_doc_retriever_tool,
    conversation_history_tool,
)

_CLARIFICATION_INSTRUCTIONS = """
You are a clarification assistant for an IoT smart home system.
Your ONLY job is to return a SHORT question asking the user which room they mean.

Steps:
1. Use `conversation_history_retriever` to check if the user already mentioned a location
   earlier in the conversation. If a location is found, return it as plain text so the
   caller can proceed — do NOT ask again.
2. If no prior location is found, use `retriever` to look up which rooms have that device type.
3. Return ONLY a single short question with the available rooms listed.
   Do NOT resolve the command. Do NOT explain. Do NOT add any preamble.

Good output example:
  "Which speaker did you mean? I found one in: living room. Please specify the room."
  "Which light did you mean? Available rooms: living room, bedroom, kitchen."

Bad output — NEVER do this:
  "The speaker is in the living room so the command would be POST /api/devices/..."
"""

clarification_agent = ToolCallingAgent(
    tools=[huggingface_doc_retriever_tool, conversation_history_tool],
    model=model,
    max_steps=3,  # history check + retrieval + return question
    verbosity_level=1,
    stream_outputs=True,
    name="clarification_agent",
    description=(
        "Handles ambiguous device control requests that are missing a location. "
        "Looks up available devices from the knowledge base and asks the user a targeted "
        "follow-up question listing the specific rooms/locations they can choose from. "
        "Use this whenever the user refers to a device by type only without specifying a room."
    ),
    instructions=_CLARIFICATION_INSTRUCTIONS,
)
