"""
Managed Smart Home Agent

Equipped with:
  - SensorLogsTool       : fetches live sensor readings from the hub API.
  - SmartHomeControlTool : sends control commands to the hub API.
  - RetrieverTool        : looks up device IDs and capabilities from the IoT knowledge base.

Handles two responsibilities:
  1. SENSOR READS   — given device_id + sensor_type (resolved by retriever_agent),
                      calls sensor_logs_reader to fetch the live value from the hub.
  2. CONTROL COMMANDS — given a natural language instruction,
                        sends the appropriate command to the hub API.

Uses ToolCallingAgent (JSON format) for structured tool invocation.
"""

from smolagents import ToolCallingAgent

from app.agent_system.model import model
from app.agent_system.tools.smart_home_control_tool import smart_home_control_tool
from app.agent_system.tools.sensor_logs_tool import sensor_logs_tool
from app.agent_system.tools.retriever_tools import huggingface_doc_retriever_tool

_SMART_HOME_INSTRUCTIONS = """
You interact with the smart home hub via your tools.

## For SENSOR READ tasks (questions about current values):
You will receive a natural language question like "how bright is the light in the living room?" or "what is the temperature in the hallway?".

Steps:
1. Call `huggingface_doc_retriever` with the question to look up:
   - The correct `device_id` from the device registry.
   - The correct `sensor_type` from the device_sensor_types knowledge.
2. Call `sensor_logs_reader(device_id=<found_id>, sensor_type=<found_type>)` to get the live reading.
3. Return the reading with device_id, sensor_type, value, unit, and timestamp.

Example — "how bright is the light in the living room?":
- `huggingface_doc_retriever` returns: device_id=light_living_01, sensor_type=brightness
- `sensor_logs_reader(device_id="light_living_01", sensor_type="brightness")` returns live brightness

## For CONTROL COMMAND tasks:
You will receive a command like "Turn on the bedroom light" or "Set thermostat to 22°C".

Steps:
1. If you need the device_id, use `huggingface_doc_retriever` to look it up.
2. Call `smart_home_control` with the correct device_id, action, and parameters.
"""

smart_home_agent = ToolCallingAgent(
    tools=[sensor_logs_tool, smart_home_control_tool, huggingface_doc_retriever_tool],
    model=model,
    max_steps=5,
    verbosity_level=1,
    stream_outputs=True,
    name="smart_home_agent",
    description=(
        "Handles all smart home hub interactions. "
        "Use for: (1) SENSOR READS — current temperature, brightness, lock state, volume, CO2, humidity, etc. "
        "It looks up the correct device_id and sensor_type from the knowledge base then fetches the live value from the hub. "
        "(2) CONTROL COMMANDS — turn lights on/off, adjust brightness, set thermostat, lock/unlock door, toggle plugs, move blinds, control speakers."
    ),
    instructions=_SMART_HOME_INSTRUCTIONS,
)
