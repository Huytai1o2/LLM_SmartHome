"""
SensorLogsTool — Dynamic sensor logs reader.

Reads live CSV sensor log files directly at query time without any embedding.
Maps to the "Dynamic — read direct" tier of the IoT RAG store architecture.
"""

from __future__ import annotations

from typing import Optional

from smolagents import Tool

from app.vectore_store.sensor_logs import read_sensor_logs


class SensorLogsTool(Tool):
    name = "sensor_logs_reader"
    description = (
        "Reads live sensor log data directly from CSV files without any embedding. "
        "Use this to answer questions about current or recent sensor readings, such as "
        "temperature, humidity, CO2, motion, power consumption, lock state, and more. "
        "Optionally filter by device ID or sensor type."
    )
    inputs = {
        "device_id": {
            "type": "string",
            "description": (
                "The device ID to filter by (e.g. 'temp_bedroom_01', 'motion_living_01'). "
                "Leave empty to retrieve logs for all devices."
            ),
            "nullable": True,
        },
        "sensor_type": {
            "type": "string",
            "description": (
                "The sensor type to filter by (e.g. 'temperature', 'humidity', 'co2', "
                "'motion', 'power', 'lock_state'). Leave empty for all sensor types."
            ),
            "nullable": True,
        },
        "last_n": {
            "type": "integer",
            "description": "How many of the most recent records to return. Defaults to 20.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        device_id: Optional[str] = None,
        sensor_type: Optional[str] = None,
        last_n: Optional[int] = 20,
    ) -> str:
        return read_sensor_logs(
            device_id=device_id or None,
            sensor_type=sensor_type or None,
            last_n=last_n if last_n is not None else 20,
        )


sensor_logs_tool = SensorLogsTool()
