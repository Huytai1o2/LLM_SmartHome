"""
SensorLogsTool — Live sensor data via the smart home hub API.

Makes HTTP GET requests to the hub to fetch current/recent sensor readings
instead of reading from local CSV files.

API endpoints used:
    GET {SMART_HOME_API_BASE}/api/devices/{device_id}/sensors
        Returns the latest readings for a specific device.

    GET {SMART_HOME_API_BASE}/api/sensors
        Returns readings for all devices, with optional ?sensor_type= filter.

The base URL is configured via the SMART_HOME_API_BASE environment variable
(defaults to http://localhost:8123).
"""

from __future__ import annotations

import json
import os
from typing import Optional

import httpx
from smolagents import Tool


class SensorLogsTool(Tool):
    name = "sensor_logs_reader"
    description = (
        "Fetches live sensor readings from the smart home hub API. "
        "Use this to get the CURRENT state or reading of any device, including: "
        "temperature, humidity, CO2 (ppm), motion detection, power consumption (W/kWh), "
        "lock state (locked/unlocked), light state (on/off), brightness (%), "
        "color temperature (K), blind position (%), speaker volume (%), and more. "
        "Always use this tool when asked about CURRENT values — do not rely on the knowledge base for live readings. "
        "Provide the device_id when known (e.g. 'light_living_01', 'thermostat_01'); "
        "leave it empty to query all devices."
    )
    inputs = {
        "device_id": {
            "type": "string",
            "description": (
                "The device ID to query (e.g. 'temp_bedroom_01', 'motion_living_01'). "
                "Leave empty to retrieve readings for all devices."
            ),
            "nullable": True,
        },
        "sensor_type": {
            "type": "string",
            "description": (
                "Filter readings by sensor type (e.g. 'temperature', 'humidity', 'co2', "
                "'motion', 'power', 'lock_state'). Leave empty for all sensor types."
            ),
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        device_id: Optional[str] = None,
        sensor_type: Optional[str] = None,
    ) -> str:
        base_url = os.getenv("SMART_HOME_API_BASE", "http://localhost:8123").rstrip("/")

        try:
            if device_id:
                url = f"{base_url}/api/devices/{device_id}/sensors"
                params = {}
                if sensor_type:
                    params["sensor_type"] = sensor_type
            else:
                url = f"{base_url}/api/sensors"
                params = {}
                if sensor_type:
                    params["sensor_type"] = sensor_type

            response = httpx.get(url, params=params, timeout=10.0)
            response.raise_for_status()

            data = response.json()
            if not data:
                return "No sensor readings found for the given filters."

            # Format the response as readable text
            lines = []
            readings = data if isinstance(data, list) else [data]
            for reading in readings:
                parts = []
                for key in (
                    "device_id",
                    "sensor_type",
                    "value",
                    "unit",
                    "timestamp",
                    "status",
                ):
                    if key in reading:
                        parts.append(f"{key}: {reading[key]}")
                lines.append(" | ".join(parts))
            return "\n".join(lines)

        except httpx.HTTPStatusError as exc:
            return (
                f"Sensor API error for device '{device_id or 'all'}': "
                f"HTTP {exc.response.status_code} — {exc.response.text}"
            )
        except httpx.RequestError as exc:
            return (
                f"Could not reach the smart home hub at {base_url}. "
                f"Check that SMART_HOME_API_BASE is set correctly. Details: {exc}"
            )
        except (json.JSONDecodeError, ValueError) as exc:
            return f"Unexpected response format from hub: {exc}"


sensor_logs_tool = SensorLogsTool()
