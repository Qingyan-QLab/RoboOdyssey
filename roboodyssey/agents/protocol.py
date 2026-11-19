"""Validation for the language-independent JSONL Agent Protocol v1."""
from __future__ import annotations

from typing import Any

PROTOCOL_VERSION = "1.1"
OBSERVATION_STREAM = {'fps': 2.0, 'mode': 'push'}
OBSERVATION_SEND_TIMEOUT = 0.5
REQUEST_TYPES = frozenset({"session_start", "step", "qa"})
ACTION_TYPES = frozenset({
    "no_op", "move_forward", "move_backward", "move_left", "move_right",
    "turn_left", "turn_right",
    "look_up", "look_down", "pick", "place", "open", "close", "reset",
})


def action_name(event, *, required=False):
    """Read an action label from string or structured action evidence."""
    action = event['action'] if required else event.get('action')
    if isinstance(action, str):
        return action
    return action['action'] if required else (action or {}).get('action')


def required_action_name(event):
    """Require complete action evidence for numerical scoring and diagnostics."""
    return action_name(event, required=True)


def validate_response(request_type: str, response: Any) -> dict[str, Any]:
    if request_type not in REQUEST_TYPES:
        raise ValueError(f"unknown request type: {request_type!r}")
    if not isinstance(response, dict):
        raise ValueError("agent response must be a JSON object")
    if response.get('agent_status', 'ok') not in {'ok', 'unsupported', 'integration_error'}:
        raise ValueError('unknown agent_status')
    if response.get('agent_status') in {'unsupported', 'integration_error'} and not response.get('reason_code'):
        raise ValueError('non-ok agent_status requires reason_code')
    if "memory_update" in response and not isinstance(response["memory_update"], dict):
        raise ValueError("memory_update must be an object")
    if "done" in response and not isinstance(response["done"], bool):
        raise ValueError("done must be a boolean")
    if request_type == "step" or (request_type == 'qa' and 'action' in response):
        action = response.get("action")
        name = action if isinstance(action, str) else action.get("action") if isinstance(action, dict) else None
        if name not in ACTION_TYPES:
            raise ValueError(f"unknown action: {name!r}")
        if request_type == 'qa' and name in {'pick', 'place', 'open', 'close'}:
            raise ValueError('inspection QA only allows navigation, look, reset and no_op')
        if isinstance(action, dict):
            if "distance" in action and not isinstance(action["distance"], (int, float)):
                raise ValueError("action distance must be numeric")
            if "angle_degrees" in action and not isinstance(action["angle_degrees"], (int, float)):
                raise ValueError("action angle_degrees must be numeric")
    elif request_type == "qa" and "answer" not in response and "prediction" not in response:
        raise ValueError("qa response requires 'answer' or 'prediction'")
    return response
