"""Smart-home YAML iterator.

Deterministic helper that filters ``smart_home_configuration.yaml`` by
``room_name`` and ``type_device`` keywords and returns a small focused YAML
subtree (the architecture's ``dataset_tmp.yaml``).

The YAML is the source of truth for the device registry — it is **never**
embedded into FAISS. Instead, agents extract keywords (room + device type)
and call this helper to get only the relevant slice, which is then handed
to the next agent (Retriever Agent) as inline context.

Typical usage
-------------
    from app.agent_system.tools.yaml_iterator import (
        iterate_smart_home_yaml,
        iterate_smart_home_yaml_tool,
    )

    subtree = iterate_smart_home_yaml("living_room", "smart_light")
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

import yaml
from smolagents import Tool

logger = logging.getLogger(__name__)

DEFAULT_YAML_PATH: str = os.environ.get(
    "SMART_HOME_CONFIG_PATH",
    "knowledge_base/iot_knowledge/smart_home_configuration.yaml",
)


# ---------------------------------------------------------------------------
# YAML loading (cached)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _load_yaml(path: str) -> dict[str, Any]:
    """Parse the smart-home YAML once per path and cache the result."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict) or "rooms" not in data:
        raise ValueError(
            f"Smart home YAML at '{path}' is missing the top-level 'rooms' key."
        )
    return data


def reload_yaml_cache() -> None:
    """Drop the cached YAML so the next call re-reads from disk."""
    _load_yaml.cache_clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(value: str | list[str] | None) -> list[str]:
    """Convert ``None`` / ``str`` / ``list[str]`` into a lower-cased ``list[str]``."""
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    else:
        items = list(value)
    return [str(item).strip().lower() for item in items if str(item).strip()]


def _filter_rooms(
    data: dict[str, Any],
    rooms_filter: list[str],
    types_filter: list[str],
) -> dict[str, Any]:
    """Return a deep-copied subtree containing only matching rooms / type_device."""
    matched_rooms: list[dict[str, Any]] = []
    for room in data.get("rooms", []):
        room_name = str(room.get("name", "")).lower()
        if rooms_filter and room_name not in rooms_filter:
            continue

        type_devices = room.get("type_device", []) or []
        if types_filter:
            type_devices = [
                td
                for td in type_devices
                if str(td.get("name_type", "")).lower() in types_filter
            ]
            if not type_devices:
                continue

        matched_rooms.append({"name": room.get("name"), "type_device": type_devices})

    return {"rooms": matched_rooms}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def iterate_smart_home_yaml(
    room_name: str | list[str] | None = None,
    type_device: str | list[str] | None = None,
    yaml_path: str = DEFAULT_YAML_PATH,
) -> str:
    """Filter the smart-home YAML and return the matching subtree as a YAML string.

    Parameters
    ----------
    room_name:
        One or more room names to keep (e.g. ``"living_room"`` or
        ``["living_room", "kitchen"]``). ``None`` keeps every room.
    type_device:
        One or more ``name_type`` values to keep (e.g. ``"smart_light"``).
        ``None`` keeps every device type within each matched room.
    yaml_path:
        Path to ``smart_home_configuration.yaml``. Defaults to the project's
        canonical location.

    Returns
    -------
    str
        A small YAML document of the form::

            rooms:
              - name: living_room
                type_device:
                  - name_type: smart_light
                    devices:
                      - name: Đèn trần
                        device_token: xdF2nW4aR9SAdqqPiym0
                        ...

        If nothing matches the filter, returns ``"rooms: []\\n"``.
    """
    data = _load_yaml(yaml_path)

    rooms_filter = _normalise(room_name)
    types_filter = _normalise(type_device)

    subtree = _filter_rooms(data, rooms_filter, types_filter)
    return yaml.safe_dump(subtree, sort_keys=False, allow_unicode=True)


def list_available_rooms(yaml_path: str = DEFAULT_YAML_PATH) -> list[str]:
    """Return every ``rooms[].name`` defined in the YAML."""
    data = _load_yaml(yaml_path)
    return [str(room.get("name")) for room in data.get("rooms", []) if room.get("name")]


def list_available_type_devices(
    room_name: str | None = None,
    yaml_path: str = DEFAULT_YAML_PATH,
) -> list[str]:
    """Return every distinct ``name_type`` (optionally restricted to ``room_name``)."""
    data = _load_yaml(yaml_path)
    types: list[str] = []
    for room in data.get("rooms", []):
        if room_name and str(room.get("name", "")).lower() != room_name.lower():
            continue
        for td in room.get("type_device", []) or []:
            name_type = td.get("name_type")
            if name_type and name_type not in types:
                types.append(str(name_type))
    return types


# ---------------------------------------------------------------------------
# smolagents Tool wrapper
# ---------------------------------------------------------------------------


class IterateSmartHomeYamlTool(Tool):
    name = "iterate_smart_home_yaml"
    description = (
        "Filter the smart_home_configuration.yaml by room_name and type_device "
        "and return only the matching subtree as a YAML string. "
        "Use this when you have already extracted the room and device-type "
        "keywords from the user's request and want to hand a small focused "
        "device list to the next agent. The YAML output contains every device's "
        "name, device_token, description_location and shared_attributes schema."
    )
    inputs = {
        "room_name": {
            "type": "string",
            "description": (
                "Room name to filter by, e.g. 'living_room', 'kitchen'. "
                "Pass null or empty string to include every room."
            ),
            "nullable": True,
        },
        "type_device": {
            "type": "string",
            "description": (
                "Device type to filter by, e.g. 'smart_light', 'smart_fan'. "
                "Pass null or empty string to include every device type within "
                "the matched rooms."
            ),
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        room_name: str | None = None,
        type_device: str | None = None,
    ) -> str:
        try:
            return iterate_smart_home_yaml(
                room_name=room_name or None,
                type_device=type_device or None,
            )
        except FileNotFoundError as exc:
            return f"Error: smart_home_configuration.yaml not found ({exc})."
        except yaml.YAMLError as exc:
            return f"Error: invalid YAML in smart_home_configuration.yaml ({exc})."
        except Exception as exc:  # noqa: BLE001
            logger.exception("iterate_smart_home_yaml failed")
            return f"Error iterating smart-home YAML: {exc}"


iterate_smart_home_yaml_tool = IterateSmartHomeYamlTool()


__all__ = [
    "iterate_smart_home_yaml",
    "list_available_rooms",
    "list_available_type_devices",
    "reload_yaml_cache",
    "IterateSmartHomeYamlTool",
    "iterate_smart_home_yaml_tool",
    "DEFAULT_YAML_PATH",
]
