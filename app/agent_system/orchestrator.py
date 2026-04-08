"""
Orchestrator – Deterministic IoT Pipeline

Replaces the previous CodeAgent master agent with a Python function that:
  1. Calls the LLM once with JSON structured output (Pydantic) to extract intent
  2. Checks Buffer Window Memory
  3. Delegates to clarification_agent if room/device info is missing
  4. Calls iterate_smart_home_yaml() deterministically to get the YAML subset
  5. Calls the LLM once with JSON structured output to select the target device(s)
  6. Delegates code generation + API execution to iot_action_agent (CodeAgent)

Why no CodeAgent for orchestration?
  Small local models (gemma4:e2b, qwen3:1.7b) hallucinate extra kwargs, write
  comments instead of code, and use `pass` instead of calling sub-agents when
  asked to orchestrate via code generation.  Structured JSON output is orders of
  magnitude more reliable for these models.

The ONLY CodeAgent in the pipeline is iot_action_agent — code generation is
appropriate there because it generates Python that calls CoreIoT tool functions.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from collections.abc import Callable
from typing import List, Optional

from app.agent_system.agents.clarification_agent import clarification_agent
from app.agent_system.agents.iot_action_agent import iot_action_agent
from app.agent_system.model import model
from app.agent_system.schemas import DeviceAction, DeviceActionList, UserIntent
from app.agent_system.tools.buffer_window_tools import check_buffer_window_tool
from app.agent_system.tools.yaml_iterator import iterate_smart_home_yaml, list_available_rooms, get_device_keyword_mapping

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts for structured LLM calls
# ---------------------------------------------------------------------------

def _get_intent_system_prompt() -> str:
    from collections import defaultdict
    rooms = list_available_rooms()
    mapping = get_device_keyword_mapping()
    
    # Reverse mapping for grouping
    type_to_keywords = defaultdict(list)
    for kw, t in mapping.items():
        if t != "all" and kw != t:
            type_to_keywords[t].append(kw)
            
    device_mapping_lines = []
    for t, keys in type_to_keywords.items():
        device_mapping_lines.append(f"  {' / '.join(keys)} → \"{t}\"")
        
    return f"""\
You are an IoT intent extractor for a Vietnamese smart home system.
Given the user's message, extract room_name and type_device.
Output ONLY valid JSON: {{"room_name": "...", "type_device": "..."}}

Available Rooms in system: {', '.join(rooms)}
Room mappings examples (infer based on meaning):
  tất cả phòng / các phòng   → "all"
  2 phòng trở lên            → ["living_room", "kitchen", ...]
  not mentioned              → null

Device type mappings:
{chr(10).join(device_mapping_lines)}
  tất cả thiết bị / các thiết bị / mọi thứ / tất cả  → "all"
  not mentioned                                   → null

Examples:
  "bật đèn trần phòng khách" → {{"room_name": "living_room", "type_device": "smart_light"}}
  "tắt quạt bếp"             → {{"room_name": "kitchen",     "type_device": "smart_fan"}}
  "bật tất cả thiết bị bếp"  → {{"room_name": "kitchen",     "type_device": "all"}}
  "tắt hết mọi thiết bị"     → {{"room_name": "all",         "type_device": "all"}}
  "bật đèn"                  → {{"room_name": null,          "type_device": "smart_light"}}
  "bật tất cả đèn"           → {{"room_name": "all",         "type_device": "smart_light"}}
  "tất cả thiết bị ở phòng khách và bếp"         → {{"room_name": ["living_room", "kitchen"], "type_device": "all"}}
  "cả nhà"                   → {{"room_name": "all",         "type_device": "all"}}
  "phòng khách"              → {{"room_name": "living_room", "type_device": null}}
"""

_RETRIEVER_SYSTEM_PROMPT = """\
You are a device-selector for an IoT smart home.
Given the user request and a YAML device config, select the matching device(s).
Output ONLY valid JSON object: {"devices": [...]}

Output schema for each device:
{
  "name_device":  "<name from YAML>",
  "token":        "<device_token from YAML — NEVER invent>",
  "device_id":    "<device_id from YAML — NEVER invent, null if not present>",
  "room":         "<room name from YAML>",
  "type_device":  "<name_type from YAML e.g. smart_light, smart_fan>",
  "shared_attribute": {"<attr_key>": <value>}
}

Intent → shared_attribute rules:
  "bật" / "turn on"            → boolean attr: true   (find the key with value=boolean)
  "tắt" / "turn off"           → boolean attr: false
  "đang ... hay ...?" / "?" / "trạng thái" → all attrs: null  (read current state)
  "độ sáng X" / "brightness X" → {"brightness": X}   (integer 0-100)
  "tốc độ X" / "speed X"       → {"speed": X}        (integer 0-3)

Selection rules:
  - User names a specific device → include only that device
  - "all" / "tất cả"            → include every device from the YAML
  - Only one device in YAML     → include it automatically

CRITICAL: copy device_token and device_id EXACTLY from the YAML. Never invent values.
"""


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------


def _parse_json(text: str) -> any:
    """Extract and parse the first JSON object or array from a model response.

    Handles plain JSON, markdown code blocks (```json ... ``` or ``` ... ```),
    and any surrounding text the model may add.
    """
    if not text:
        raise ValueError("Empty response from model")
    # Strip markdown code fences
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fence:
        text = fence.group(1)
    # Find first { or [ and parse from there
    match = re.search(r"[{\[]", text)
    if match:
        text = text[match.start():]
        
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Nhờ ast (Abstract Syntax Tree) để parse json xài nháy đơn (dict của Python)
        return ast.literal_eval(text)


def _extract_intent(
    user_message: str,
    history: Optional[list] = None,
) -> UserIntent:
    """Call the LLM to extract room + device type from the user message.

    History (last few turns) is prepended so follow-up messages like
    "phòng khách" are correctly resolved in context.
    """
    def _text(text: str) -> list:
        return [{"type": "text", "text": text}]

    messages: list = [{"role": "system", "content": _text(_get_intent_system_prompt())}]

    if history:
        for msg in history[-4:]:  # last 2 turns (4 messages) for context
            messages.append({"role": msg["role"], "content": _text(msg["content"])})

    messages.append({"role": "user", "content": _text(user_message)})

    try:
        response = model(messages=messages)
        data = _parse_json(response.content)
        return UserIntent(
            room_name=data.get("room_name") or None,
            type_device=data.get("type_device") or None,
        )
    except Exception as e:
        logger.warning(f"Intent extraction failed: {str(e)} — returning empty intent")
        return UserIntent()


def _select_devices(user_message: str, yaml_subset: str) -> List[DeviceAction]:
    """Call the LLM to select device(s) from the YAML subset and determine
    what value to set (or None for read).

    Returns a validated list of DeviceAction objects.
    """
    def _text(text: str) -> list:
        return [{"type": "text", "text": text}]

    try:
        response = model(
            messages=[
                {"role": "system", "content": _text(_RETRIEVER_SYSTEM_PROMPT)},
                {
                    "role": "user",
                    "content": _text(
                        f"User request: {user_message}\n\nDevice YAML:\n{yaml_subset}"
                    ),
                },
            ],
        )
        data = _parse_json(response.content)

        # Normalise: model may return {"devices": [...]} or bare [...]
        if isinstance(data, list):
            raw_list = data
        elif isinstance(data, dict) and "devices" in data:
            raw_list = data["devices"]
        else:
            raw_list = [data]

        return DeviceActionList(devices=raw_list).devices

    except Exception as e:
        logger.warning(f"Device selection failed: {str(e)}")
        return []


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------


def run_iot_pipeline(
    user_message: str,
    history: Optional[list] = None,
    on_step: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Deterministic IoT orchestration pipeline.

    Args:
        user_message: The latest user message.
        history:      Prior turns as [{"role": "user"|"assistant", "content": str}].
                      Used to provide context for follow-up messages.
        on_step:      Optional callback for streaming step-level status updates.

    Returns:
        The final answer string to send back to the user.
    """

    def emit(agent_name: str, text: str) -> None:
        if on_step:
            on_step(f"{agent_name}: {text}\n\n")

    # ------------------------------------------------------------------
    # Step 1 — Extract intent (Pydantic structured output, no code gen)
    # ------------------------------------------------------------------
    emit("Orchestrator", "Đang phân tích yêu cầu...")
    intent = _extract_intent(user_message, history=history)
    logger.info("Extracted intent: room=%s type=%s", intent.room_name, intent.type_device)

    # ------------------------------------------------------------------
    # Step 2 — Check Buffer Window Memory
    # ------------------------------------------------------------------
    cached = check_buffer_window_tool.forward(user_message)
    if cached != "[]":
        try:
            hit = json.loads(cached)[0]
            if intent.room_name is None:
                intent.room_name = hit.get("room")
            if intent.type_device is None:
                intent.type_device = hit.get("type_device") or None
            emit(
                "Orchestrator",
                f"Bộ nhớ đệm: {hit.get('device_name')} "
                f"({hit.get('room')})"
            )
            logger.debug("Buffer hit: %s", hit)
        except (json.JSONDecodeError, IndexError, KeyError):
            pass

    # ------------------------------------------------------------------
    # Step 3 — Clarify if info still missing
    # ------------------------------------------------------------------
    if intent.room_name is None or intent.type_device is None:
        emit("Orchestrator", "Cần thêm thông tin...")
        question = clarification_agent.run(user_message, intent)
        return f"Clarification_Agent: {str(question)}"

    emit("Orchestrator", f"Phòng: {intent.room_name} | Thiết bị: {intent.type_device}")

    # ------------------------------------------------------------------
    # Step 4 — Iterate YAML → focused subset (deterministic, no LLM)
    # ------------------------------------------------------------------
    yaml_subset = iterate_smart_home_yaml(
        room_name=None if intent.room_name == "all" else intent.room_name,
        type_device=None if intent.type_device == "all" else intent.type_device,
    )
    if yaml_subset.strip() == "rooms: []":
        return (
            f"Orchestrator: Không tìm thấy thiết bị '{intent.type_device}' "
            f"trong '{intent.room_name}'."
        )

    # ------------------------------------------------------------------
    # Step 5 — Select target device(s) (Pydantic structured output)
    # ------------------------------------------------------------------
    emit("Orchestrator", "Đang chọn thiết bị...")
    devices = _select_devices(user_message, yaml_subset)

    if not devices:
        return "Orchestrator: Không thể xác định thiết bị cần điều khiển."

    devices_json = json.dumps(
        [d.model_dump() for d in devices], ensure_ascii=False
    )
    logger.info("Selected devices: %s", devices_json)

    # ------------------------------------------------------------------
    # Step 6 — Route & execute via iot_action_agent (CodeAgent)
    # Decides write vs read split, calls post/read tools, updates BufferWindow
    # ------------------------------------------------------------------
    emit("Orchestrator", "Đang thực thi lệnh...")
    result = iot_action_agent.run(devices_json)
    return f"IoT_Action_Agent: {str(result)}"
