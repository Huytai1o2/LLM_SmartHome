"""
Managed Smart Home Agent

Equipped with:
  - SmartHomeControlTool : sends control commands to the hub API.
  - RetrieverTool        : looks up device IDs and capabilities from the IoT knowledge base.

Having both tools lets the agent resolve the correct device_id itself before sending the command,
without needing the manager to do a separate retrieval step.

Uses ToolCallingAgent (JSON format) for structured tool invocation.
"""

from smolagents import ToolCallingAgent

from app.agent_system.model import model
from app.agent_system.tools.smart_home_control_tool import smart_home_control_tool
from app.agent_system.tools.retriever_tools import huggingface_doc_retriever_tool

smart_home_agent = ToolCallingAgent(
    tools=[huggingface_doc_retriever_tool, smart_home_control_tool],
    model=model,
    max_steps=5,
    verbosity_level=1,
    stream_outputs=True,
    name="smart_home_agent",
    description=(
        "Controls smart home devices. Given a natural language instruction, it looks up "
        "the correct device_id and capabilities from the IoT knowledge base, then sends "
        "the appropriate command to the hub API. "
        "Examples: 'Turn on the bedroom ceiling light', 'Set living room speaker volume to 60', "
        "'Lock the front door', 'Set thermostat to 22°C in heat mode'."
    ),
)
