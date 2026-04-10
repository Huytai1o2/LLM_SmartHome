"""
Clarification Agent — resolves missing room/device context.

No LLM, no code generation — fully deterministic:
  1. Check Buffer Window Memory for cached device/room this session.
  2. If still missing → asks specifically for the missing info (room or device type).
"""

from __future__ import annotations

import json
import re

from app.agent_system.tools.buffer_window_tools import check_buffer_window_tool
from app.agent_system.memory.buffer_window import get_current_buffer
from app.agent_system.tools.yaml_iterator import iterate_smart_home_yaml_tool, list_available_rooms, get_room_and_device_types


class ClarificationAgent:
    """Deterministic clarification — no LLM, no code generation."""

    def run(self, user_message: str, extracted_intent=None) -> str:
        # Step 1 — Check Buffer Window Memory
        cached = check_buffer_window_tool.forward(user_message)
        if cached != "[]":
            try:
                hit = json.loads(cached)[0]
                return (
                    f"CACHED: device_name={hit.get('device_name')}, "
                    f"room={hit.get('room')}, token={hit.get('token')}"
                )
            except (json.JSONDecodeError, IndexError, KeyError):
                pass
                
        # Use extracted intent directly
        has_room = False
        device_type = None
        if extracted_intent:
            has_room = bool(extracted_intent.room_name)
            device_type = extracted_intent.type_device

        # If LLM extracted "all"
        if device_type == "all" and has_room:
            pass

        # Build suggestions from Buffer Window
        suggestion_msg = ""
        buf = get_current_buffer()
        if buf and buf.all():
            recent = buf.all()[-1] # the most recent action
            suggestion_msg = f" Hay là bạn muốn điều khiển '{recent.device_name}' ở '{recent.room}' như trước đó?"

        # Step 2 — Missing Room
        if not has_room:
            # We filter rooms that actually have the requested device type
            room_names = list_available_rooms()
            if device_type and device_type != "all":
                # Only offer rooms that actually contain the requested device
                yaml_result = iterate_smart_home_yaml_tool.forward(
                    room_name="", type_device=device_type
                )
                rooms = re.findall(r"name:\s*(\S+)", yaml_result)
                # Keep rooms that appear in yaml_result
                room_names = [r for r in room_names if r in rooms]
            
            room_list = ", ".join(room_names) if room_names else "không rõ"
            return f"Bạn muốn điều khiển ở phòng nào? Các phòng hiện có: {room_list}.{suggestion_msg}"
            
        # Step 3 - Missing Device
        if not device_type:
            # Build a string like: phòng khách (smart_light, smart_fan), phòng bếp (smart_fan)
            room_to_types = get_room_and_device_types()
            details = []
            for r, t_list in room_to_types.items():
                if has_room and extracted_intent and r != extracted_intent.room_name and r not in (extracted_intent.room_name or []):
                    continue
                types_str = ", ".join(t_list)
                details.append(f"{r} ({types_str})")
                
            avail_str = "; ".join(details) if details else "không rõ"
            return f"Bạn muốn điều khiển thiết bị nào? (Các thiết bị hiện có: {avail_str} hoặc 'tất cả thiết bị').{suggestion_msg}"
            
        return "Bạn hãy cung cấp rõ hơn thông tin thiết bị hoặc phòng nhé."


clarification_agent = ClarificationAgent()
