"""
IoT Action Agent — Device Router.

Receives a JSON list of DeviceAction objects from the orchestrator and
routes each device to the correct tool:
  - shared_attributes has ANY non-null value  → WRITE → post_shared_attributes_tool
  - shared_attributes has ALL null values     → READ  → read_shared_attributes_tool

No LLM or code generation — the routing rule is deterministic.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.agent_system.tools.iot_action_tools import (
    post_shared_attributes_tool,
    read_shared_attributes_tool,
)

logger = logging.getLogger(__name__)


class IoTActionAgent:
    """Routes devices to the correct CoreIoT tool based on shared_attributes values.

    Write (non-null values) → post_shared_attributes_tool → RPC setValue
    Read  (all-null values) → read_shared_attributes_tool → GET clientKeys
    """

    def run(self, devices_json: str) -> str:
        try:
            devices: list[dict[str, Any]] = json.loads(devices_json)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error("IoTActionAgent: invalid JSON input — %s", exc)
            return f"ERROR: invalid devices payload — {exc}"

        write_devices = []
        read_devices = []
        dict_by_token = {}

        for d in devices:
            token = d.get("token")
            devices_sensors = d.get("sensors", [])
            # fallback if LLM occasionally puts 'shared_attributes' instead of 'sensors' array
            if not devices_sensors and "shared_attributes" in d:
                devices_sensors = [{"sensor_name": "basic_sensor", "shared_attributes": d["shared_attributes"]}]
                
            dict_by_token[token] = devices_sensors

            write_attrs = {}
            read_attrs = {}

            for sensor in devices_sensors:
                attr = sensor.get("shared_attributes", {})
                for k, v in attr.items():
                    if v is not None:
                        write_attrs[k] = v
                    else:
                        read_attrs[k] = v

            if write_attrs:
                d_write = d.copy()
                d_write["shared_attributes"] = write_attrs
                write_devices.append(d_write)

            if read_attrs:
                d_read = d.copy()
                d_read["shared_attributes"] = read_attrs
                read_devices.append(d_read)

        lines: list[str] = []

        if write_devices:
            result_json = post_shared_attributes_tool.forward(json.dumps(write_devices, ensure_ascii=False))
            for r in json.loads(result_json):
                token = r.get("token")
                orig_sensors = dict_by_token.get(token, [])
                
                # Format sensor output
                s_strings = []
                for sensor in orig_sensors:
                    s_name = sensor.get("sensor_name", "unknown")
                    # Filter attributes to those actually present in the write response/intent
                    s_attr = {k: v for k, v in sensor.get("shared_attributes", {}).items() if v is not None}
                    if s_attr:
                        s_strings.append(f"{s_name}: {s_attr}")
                sensors_info = " | ".join(s_strings) if s_strings else str(r.get("after", {}))

                if r.get("error"):
                    lines.append(f"{r['name_device']} ({r['room']}): ERROR — {r['error']}")
                elif r.get("posted"):
                    lines.append(f"{r['name_device']} ({r['room']}): success → {sensors_info}")
                else:
                    lines.append(f"{r['name_device']} ({r['room']}): already in requested state → {sensors_info}")

        if read_devices:
            result_json = read_shared_attributes_tool.forward(json.dumps(read_devices, ensure_ascii=False))
            for r in json.loads(result_json):
                if r.get("error"):
                    lines.append(f"{r['name_device']} ({r['room']}): ERROR — {r['error']}")
                else:
                    # You could also format read_devices by sensor here if needed, 
                    # but simple string dict is fine to pass to the generating agent
                    lines.append(f"{r['name_device']} ({r['room']}): {r['shared']}")

        return "\n".join(lines) if lines else "No results."


iot_action_agent = IoTActionAgent()
