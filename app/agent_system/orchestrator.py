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
from app.agent_system.memory.buffer_window import get_current_buffer
from app.vectore_store.conversation_memory import load_conversation_context
from app.agent_system.tools.yaml_iterator import (
    iterate_smart_home_yaml,
    list_available_rooms,
    get_room_and_device_types,
    get_device_summary,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ---------------------------------------------------------------------------
# System prompts for structured LLM calls
# ---------------------------------------------------------------------------

def _get_intent_system_prompt(session_id: str = None, user_message: str = "") -> str:
    device_summary = get_device_summary()
    buf = get_current_buffer()
    recent_context = buf.to_context_string(limit=2) if buf else ""
    conversation_history = load_conversation_context(user_message, session_id=session_id) if session_id else ""
    
    return f"""\
Extract intent as JSON: {{"room_name": "...", "type_device": "...", "device_name": "..."}}. Use `null` if missing. Maps to "all" if user says "tất cả".
You MUST output ONLY a valid JSON object. No explanations, no conversation.
Available devices:
{device_summary}

Recent Context: {recent_context}
History: {conversation_history}

Examples:
"bật đèn phòng khách" → {{"room_name": "living_room", "type_device": "smart_light_fan", "device_name": "Celling_fan_bedside_night_light"}}
"tắt quạt" → {{"room_name": null, "type_device": "smart_light_fan", "device_name": null}}
"""

_RETRIEVER_SYSTEM_PROMPT = """\
Select matching device(s) and ONLY the requested sensor(s) from YAML. Output JSON ONLY: {"devices": [{"name_device":"<name>","token":"<token>","device_id":"<id>","room":"<room>","type_device":"<type>","sensors":[{"sensor_name":"<sensor>","shared_attributes":{"<k>":<v>}}]}]}

Rules:
1. ONLY include the specific `sensor_name` and `name_key` that match the user's request. DO NOT include all sensors if the user only asked for one.
2. `shared_attributes` MUST be ONE dict.
3. If changing state ("bật"/"tắt"/etc): `<v>` must be the actual value (`true`, `false`, `1`, `0`, etc).
4. If checking/reading state ("read"/"kiểm tra"/etc): `<v>` MUST literally be `null`.
5. Copy token and device_id EXACTLY from YAML. NEVER invent.
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
    session_id: str,
    history: Optional[list] = None,
) -> UserIntent:
    """Call the LLM to extract room + device type from the user message.

    History (last few turns) is prepended so follow-up messages like
    "phòng khách" are correctly resolved in context.
    """
    def _text(text: str) -> list:
        return [{"type": "text", "text": text}]

    messages: list = [{"role": "system", "content": _text(_get_intent_system_prompt(session_id, user_message))}]

    # Combine history into the user message to prevent the model from roleplaying as "assistant"
    # and falling into conversational output instead of JSON.
    user_req = ""
    if history:
        user_req += "[Recent Conversation History]\n"
        for msg in history[-4:]:  # last 2 turns
            user_req += f"{msg['role'].upper()}: {msg['content']}\n"
        user_req += "\n"
        
    user_req += f"[Current Request]\nUSER: {user_message}\n\n=> Output JSON intent now:"

    messages.append({"role": "user", "content": _text(user_req)})

    try:
        logger.info(f"API CALL: _extract_intent")
        logger.info(f"PROMPT messages for _extract_intent:\n{json.dumps(messages, ensure_ascii=False, indent=2)}")
        response = model(messages=messages)
        logger.info(f"AI RESPONSE: _extract_intent: {response.content}")
        data = _parse_json(response.content)
        return UserIntent(
            room_name=data.get("room_name") or None,
            type_device=data.get("type_device") or None,
            device_name=data.get("device_name") or None,
        )
    except Exception as e:
        logger.warning(f"Intent extraction failed: {str(e)} — returning empty intent")
        return UserIntent()


def _select_devices(user_message: str, yaml_subset: str, intent: UserIntent = None, session_id: str = None) -> List[DeviceAction]:
    """Call the LLM to select device(s) from the YAML subset and determine
    what value to set (or None for read).

    Returns a validated list of DeviceAction objects.
    """
    conversation_history = load_conversation_context(user_message, session_id=session_id) if session_id else "(empty history)"
    
    def _text(text: str) -> list:
        return [{"type": "text", "text": text}]

    intent_yaml = ""
    if intent:
        intent_yaml = (
            "Extracted Intent (YAML):\n"
            "rooms:\n"
            f"  - name: {intent.room_name}\n"
            "    type_device:\n"
            f"      - name_type: {intent.type_device}\n"
            "        devices:\n"
            f"          - name: {intent.device_name}\n\n"
        )

    try:
        logger.info("API CALL: _select_devices")
        messages = [
            {"role": "system", "content": _text(_RETRIEVER_SYSTEM_PROMPT)},
            {
                "role": "user",
                "content": _text(
                    f"User request: {user_message}\n\n{intent_yaml}Device YAML:\n{yaml_subset}\n\nLong-term Conversation History:\n{conversation_history}"
                ),
            },
        ]
        logger.info(f"PROMPT messages for _select_devices:\n{json.dumps(messages, ensure_ascii=False, indent=2)}")
        response = model(messages=messages)
        logger.info(f"AI RESPONSE: _select_devices: {response.content}")
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


def _generate_final_response(user_message: str, result: str, yaml_subset: str = "") -> str:
    """Sử dụng LLM để chuyển đổi kết quả thực thi thành câu trả lời ngôn ngữ tự nhiên."""
    def _text(text: str) -> list:
        return [{"type": "text", "text": text}]
        
    system_prompt = (
        "Act as a smart home assistant. Write a short, natural response based on 'User Request' & 'System Result'. "
        "Explain errors sympathetically if any. Map 'System Result' values using 'Device Configuration'. Output final answer only."
    )
    
    try:
        logger.info("API CALL: _generate_final_response")
        messages = [
            {"role": "system", "content": _text(system_prompt)},
            {
                "role": "user",
                "content": _text(
                    f"User Request: {user_message}\n\nSystem Result:\n{result}\n\nDevice Configuration:\n{yaml_subset}"
                ),
            },
        ]
        logger.info(f"PROMPT messages for _generate_final_response:\n{json.dumps(messages, ensure_ascii=False, indent=2)}")
        response = model(messages=messages)
        logger.info(f"AI RESPONSE: _generate_final_response: {response.content}")
        return response.content
    except Exception as e:
        logger.warning(f"Final response generation failed: {str(e)}")
        return f"System reported result:\n{result}"


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------


def run_iot_pipeline(
    user_message: str,
    session_id: str,
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
    emit("Orchestrator", "Analyzing request...")
    intent = _extract_intent(user_message, session_id=session_id, history=history)
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
                f"Buffer Memory: {hit.get('device_name')} "
                f"({hit.get('room')})"
            )
            logger.debug("Buffer hit: %s", hit)
        except (json.JSONDecodeError, IndexError, KeyError):
            pass

    # ------------------------------------------------------------------
    # Step 3 — Clarify if info still missing
    # ------------------------------------------------------------------
    if intent.room_name is None or intent.type_device is None:
        emit("Orchestrator", "Need more information...")
        question = clarification_agent.run(user_message, intent)
        return f"Clarification_Agent: {str(question)}"

    emit("Orchestrator", f"Room: {intent.room_name} | Device: {intent.type_device}")

    # ------------------------------------------------------------------
    # Step 4 — Iterate YAML → focused subset (deterministic, no LLM)
    # ------------------------------------------------------------------
    yaml_subset = iterate_smart_home_yaml(
        room_name=None if intent.room_name == "all" else intent.room_name,
        type_device=None if intent.type_device == "all" else intent.type_device,
    )
    if yaml_subset.strip() == "rooms: []":
        return (
            f"Orchestrator: Could not find device '{intent.type_device}' "
            f"in '{intent.room_name}'."
        )

    # ------------------------------------------------------------------
    # Step 5 — Select target device(s) (Pydantic structured output)
    # ------------------------------------------------------------------
    emit("Orchestrator", "Selecting devices...")
    devices = _select_devices(user_message, yaml_subset, intent, session_id=session_id)

    if not devices:
        return "Orchestrator: Could not determine target device."

    devices_json = json.dumps(
        [d.model_dump() for d in devices], ensure_ascii=False
    )
    logger.info("Selected devices: %s", devices_json)

    # ------------------------------------------------------------------
    # Step 6 — Route & execute via iot_action_agent (CodeAgent)
    # Decides write vs read split, calls post/read tools, updates BufferWindow
    # ------------------------------------------------------------------
    emit("Orchestrator", "Executing command...")
    raw_result = iot_action_agent.run(devices_json)
    
    emit("Orchestrator", "Generating response...")
    final_response = _generate_final_response(user_message, raw_result, yaml_subset)
    
    # Optional prefixes like "IoT_Action_Agent: " may not be needed anymore
    return final_response
