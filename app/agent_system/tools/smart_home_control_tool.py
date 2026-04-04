"""
SmartHomeControlTool — Send control commands to smart home devices.

Makes HTTP POST requests to the smart home hub API to control actuators
such as lights, thermostats, locks, smart plugs, blinds, and speakers.

API format (from demonstration knowledge):
    POST {SMART_HOME_API_BASE}/api/devices/{device_id}/control
    Body: {"action": "<action>", ...extra_params}

The base URL is configured via the SMART_HOME_API_BASE environment variable
(defaults to http://localhost:8123).

Supported device IDs and their actions come from the IoT device registry:
  - light_living_01, light_bedroom_01, light_kitchen_01
      actions: turn_on, turn_off, set
      params:  brightness (0-100), color_temperature (2700-6500)

  - thermostat_01
      actions: set_temperature, set_mode
      params:  setpoint (16-30), mode (heat/cool/auto/off)

  - door_front_01
      actions: lock, unlock
      params:  duration_seconds (optional, for timed unlock)

  - plug_tv_01, plug_fridge_01
      actions: turn_on, turn_off

  - blind_living_01
      actions: set_position, open, close
      params:  position (0-100), tilt (0-100)

  - speaker_living_01
      actions: play, pause, set_volume
      params:  volume (0-100), media_source
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx
from smolagents import Tool


class SmartHomeControlTool(Tool):
    name = "smart_home_control"
    description = (
        "Sends a control command to a smart home device via the hub API. "
        "Use this when the user wants to control a device — turn lights on/off, "
        "adjust brightness or colour temperature, set the thermostat temperature or mode, "
        "lock or unlock the front door, toggle smart plugs, or change blind positions. "
        "The device_id must match a registered device (e.g. 'light_living_01', "
        "'thermostat_01', 'door_front_01'). "
        "Always retrieve device capabilities from the retriever_agent first if unsure."
    )
    inputs = {
        "device_id": {
            "type": "string",
            "description": (
                "The ID of the device to control. Must be one of the registered device IDs, "
                "e.g. 'light_living_01', 'light_bedroom_01', 'light_kitchen_01', "
                "'thermostat_01', 'door_front_01', 'plug_tv_01', 'plug_fridge_01', "
                "'blind_living_01', 'speaker_living_01'."
            ),
        },
        "action": {
            "type": "string",
            "description": (
                "The action to perform on the device. Examples: "
                "'turn_on', 'turn_off', 'set', 'set_temperature', 'set_mode', "
                "'lock', 'unlock', 'set_position', 'open', 'close', "
                "'play', 'pause', 'set_volume'."
            ),
        },
        "parameters": {
            "type": "object",
            "description": (
                "Optional dictionary of extra parameters for the action. Examples: "
                '{"brightness": 80} for lights, '
                '{"setpoint": 22, "mode": "heat"} for the thermostat, '
                '{"position": 50} for blinds, '
                '{"volume": 100} for speakers, '
                '{"duration_seconds": 300} for a timed door unlock. '
                "Pass null or omit if the action needs no extra parameters."
            ),
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        device_id: str,
        action: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> str:
        base_url = os.getenv("SMART_HOME_API_BASE", "http://localhost:8123").rstrip("/")
        url = f"{base_url}/api/devices/{device_id}/control"

        payload: dict[str, Any] = {"action": action}

        if parameters:
            if not isinstance(parameters, dict):
                return f"Error: parameters must be a dictionary, got: {parameters!r}"
            payload.update(parameters)

        try:
            response = httpx.post(url, json=payload, timeout=10.0)
            response.raise_for_status()
            return (
                f"Command sent to {device_id}: action='{action}', payload={payload}.\n"
                f"Hub response ({response.status_code}): {response.text}"
            )
        except httpx.HTTPStatusError as exc:
            return (
                f"Device API error for '{device_id}': "
                f"HTTP {exc.response.status_code} — {exc.response.text}"
            )
        except httpx.RequestError as exc:
            return (
                f"Could not reach the smart home hub at {base_url}. "
                f"Check that SMART_HOME_API_BASE is set correctly. Details: {exc}"
            )


smart_home_control_tool = SmartHomeControlTool()

__all__ = ["SmartHomeControlTool", "smart_home_control_tool"]
