"""
Pydantic schemas for structured LLM output throughout the IoT pipeline.

These schemas force the LLM to output valid JSON instead of generating Python code,
which is far more reliable for small local models like gemma4:e2b.

Pipeline:
  UserIntent    → Master Agent  (intent extraction from user message)
  DeviceAction  → Retriever     (device selection from YAML subset)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UserIntent(BaseModel):
    """
    Structured output from the Master Agent's intent extraction step.

    The LLM is asked to extract room_name and type_device from the user's
    natural-language message. Both fields are Optional — None signals that
    the information was not present in the message and further clarification
    is needed.
    """

    room_name: Optional[str] = Field(
        default=None,
        description=(
            "The room name in snake_case English, e.g. 'living_room', 'kitchen', "
            "'bedroom'. None if the user did not mention a room."
        ),
    )
    type_device: Optional[str] = Field(
        default=None,
        description=(
            "The device category, e.g. 'smart_light' or 'smart_fan'. "
            "None if the user did not mention a device type."
        ),
    )


class DeviceAction(BaseModel):
    """
    One device to act on, as selected by the Retriever step.

    Matches the output contract expected by iot_action_agent:
        name_device   — human-readable device name from YAML
        token         — CoreIoT device token (NEVER invented — copied from YAML)
        room          — room name from YAML
        shared_attribute — key:value pairs where:
            value = True/False  → control (set attribute)
            value = None        → read (query current state)
    """

    name_device: str
    token: str
    device_id: Optional[str] = Field(
        default=None,
        description="CoreIoT device UUID — required for Server-Side RPC control.",
    )
    room: str
    type_device: Optional[str] = Field(
        default=None,
        description="Device category from YAML e.g. 'smart_light', 'smart_fan'.",
    )
    shared_attribute: Dict[str, Any] = Field(
        description=(
            "E.g. {'led': True} to turn on, {'led': False} to turn off, "
            "{'led': None} to read current value, {'brightness': 80} to set level."
        )
    )


class DeviceActionList(BaseModel):
    """Wrapper so the LLM can output a JSON object instead of a bare array."""

    devices: List[DeviceAction]
