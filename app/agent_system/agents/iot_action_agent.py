"""
IoT Action Agent — Device Router.

Receives a JSON list of DeviceAction objects from the orchestrator and
routes each device to the correct tool:
  - shared_attribute has ANY non-null value  → WRITE → post_shared_attribute_tool
  - shared_attribute has ALL null values     → READ  → read_shared_attribute_tool

No LLM or code generation — the routing rule is deterministic.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.agent_system.tools.iot_action_tools import (
    post_shared_attribute_tool,
    read_shared_attribute_tool,
)

logger = logging.getLogger(__name__)


class IoTActionAgent:
    """Routes devices to the correct CoreIoT tool based on shared_attribute values.

    Write (non-null values) → post_shared_attribute_tool → RPC setValue
    Read  (all-null values) → read_shared_attribute_tool → GET clientKeys
    """

    def run(self, devices_json: str) -> str:
        try:
            devices: list[dict[str, Any]] = json.loads(devices_json)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error("IoTActionAgent: invalid JSON input — %s", exc)
            return f"LỖI: dữ liệu devices không hợp lệ — {exc}"

        write_devices = [
            d for d in devices
            if any(v is not None for v in d.get("shared_attribute", {}).values())
        ]
        read_devices = [
            d for d in devices
            if all(v is None for v in d.get("shared_attribute", {}).values())
        ]

        lines: list[str] = []

        if write_devices:
            result_json = post_shared_attribute_tool.forward(json.dumps(write_devices, ensure_ascii=False))
            for r in json.loads(result_json):
                if r.get("error"):
                    lines.append(f"{r['name_device']} ({r['room']}): LỖI — {r['error']}")
                elif r.get("posted"):
                    lines.append(f"{r['name_device']} ({r['room']}): thành công → {r['after']}")
                else:
                    lines.append(f"{r['name_device']} ({r['room']}): đã ở trạng thái yêu cầu")

        if read_devices:
            result_json = read_shared_attribute_tool.forward(json.dumps(read_devices, ensure_ascii=False))
            for r in json.loads(result_json):
                if r.get("error"):
                    lines.append(f"{r['name_device']} ({r['room']}): LỖI — {r['error']}")
                else:
                    lines.append(f"{r['name_device']} ({r['room']}): {r['shared']}")

        return "\n".join(lines) if lines else "Không có kết quả."


iot_action_agent = IoTActionAgent()
