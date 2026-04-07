"""
Clarification Agent — resolves missing room/device context.

No LLM, no code generation — fully deterministic:
  1. Check Buffer Window Memory for cached device/room this session.
  2. If still missing → list candidate rooms from YAML and ask one question.
"""

from __future__ import annotations

import json
import re

from app.agent_system.tools.buffer_window_tools import check_buffer_window_tool
from app.agent_system.tools.yaml_iterator import iterate_smart_home_yaml_tool


_DEVICE_TYPE_MAP = {
    "đèn": "smart_light",
    "bóng đèn": "smart_light",
    "đèn trần": "smart_light",
    "đèn ngủ": "smart_light",
    "đèn bếp": "smart_light",
    "quạt": "smart_fan",
    "quạt trần": "smart_fan",
    "light": "smart_light",
    "fan": "smart_fan",
}


def _infer_device_type(user_message: str) -> str | None:
    msg = user_message.lower()
    for keyword, dtype in _DEVICE_TYPE_MAP.items():
        if keyword in msg:
            return dtype
    return None


class ClarificationAgent:
    """Deterministic clarification — no LLM, no code generation."""

    def run(self, user_message: str) -> str:
        # Step 1 — Check Buffer Window Memory
        cached = check_buffer_window_tool.forward(user_message)
        if cached != "[]":
            try:
                hit = json.loads(cached)[0]
                return (
                    f"CACHED: device_name={hit['device_name']}, "
                    f"room={hit['room']}, token={hit['token']}"
                )
            except (json.JSONDecodeError, IndexError, KeyError):
                pass

        # Step 2 — List rooms from YAML and ask one question
        device_type = _infer_device_type(user_message)
        yaml_result = iterate_smart_home_yaml_tool.forward(
            room_name="", type_device=device_type or ""
        )
        rooms = re.findall(r"name:\s*(\S+)", yaml_result)
        # Filter out device names (rooms are snake_case like living_room, kitchen)
        room_names = [r for r in rooms if "_" in r or r in ("kitchen", "bedroom")]
        if not room_names:
            room_names = list(dict.fromkeys(rooms))  # deduplicate, preserve order

        room_list = ", ".join(room_names) if room_names else "không rõ"
        return f"Bạn muốn điều khiển ở phòng nào? Các phòng hiện có: {room_list}."


clarification_agent = ClarificationAgent()
