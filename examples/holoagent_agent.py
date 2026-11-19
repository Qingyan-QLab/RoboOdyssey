#!/usr/bin/env python3
"""HoloAgent -> Long-Term Benchmark Protocol v1 adapter.

HoloAgent is a framework, not a single checkpoint.  This process keeps the
benchmark's stdin/stdout JSONL contract and delegates planning/execution to a
small HTTP gateway exposed by AgentOS, an FSR-VLN service, or a test stub.

The gateway receives ``{"protocol_version": "1.0", "request": ...}`` and
returns either a protocol response directly or ``{"response": {...}}``.  A
response may contain a normal benchmark ``action`` or a gateway-normalized
``relative_nav`` object with ``forward``, ``left`` and ``rotation`` fields.
The latter is not the raw response of HoloAgent's robot_bridge: robot_bridge
returns command-accepted status and reports completion asynchronously.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roboodyssey.agents.stream import serve


ACTION_NAMES = {
    "no_op", "move_forward", "move_backward", "move_left", "move_right",
    "turn_left", "turn_right", "look_up", "look_down", "pick", "place",
    "open", "close", "reset",
}

_ACTION_ALIASES = {
    "forward": "move_forward", "backward": "move_backward",
    "left": "turn_left", "right": "turn_right", "rotate_left": "turn_left",
    "rotate_right": "turn_right", "pickup": "pick", "pick_up": "pick",
    "put_down": "place", "drop": "place", "stop": "no_op", "wait": "no_op",
}


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"HoloAgent gateway request failed: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("HoloAgent gateway must return a JSON object")
    return value.get("response", value)


def _relative_action(value: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one HoloAgent relative-nav result to one benchmark action."""
    try:
        forward = float(value.get("forward", 0.0))
        left = float(value.get("left", 0.0))
        rotation = float(value.get("rotation", 0.0))
    except (TypeError, ValueError):
        return None
    if abs(rotation) >= 1e-3:
        return {"action": "turn_left" if rotation > 0 else "turn_right",
                "angle_degrees": abs(rotation)}
    if abs(forward) >= 1e-3:
        return {"action": "move_forward" if forward > 0 else "move_backward",
                "distance": abs(forward)}
    if abs(left) >= 1e-3:
        return {"action": "move_left" if left > 0 else "move_right",
                "distance": abs(left)}
    return {"action": "no_op"}


def normalize_response(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize gateway output without hiding malformed actions."""
    response = dict(value)
    action = response.get("action")
    if action is None and isinstance(response.get("relative_nav"), dict):
        action = _relative_action(response["relative_nav"])
    if action is None and isinstance(response.get("skill_result"), dict):
        result = response["skill_result"]
        if isinstance(result.get("relative_nav"), dict):
            action = _relative_action(result["relative_nav"])
    if action is None and ("success" in response or "accepted" in response) \
            and not response.get("gateway_status"):
        raise RuntimeError(
            "received a raw HoloAgent robot_bridge status; configure a "
            "/benchmark/step gateway that returns a Protocol v1 action")
    if action is None:
        action = {"action": "no_op"}
    if isinstance(action, str):
        action = {"action": action}
    if isinstance(action, dict):
        action = dict(action)
        name = str(action.get("action", "")).strip().lower().replace(" ", "_")
        action["action"] = _ACTION_ALIASES.get(name, name)
        arguments = action.pop("arguments", None) or action.pop("parameters", None)
        if isinstance(arguments, dict):
            for key, item in arguments.items():
                action.setdefault(key, item)
        if action["action"] in {"pick", "open", "close"} and not action.get("object_id"):
            action["object_id"] = action.get("target") or action.get("object")
        if action["action"] == "place" and not action.get("room_id"):
            action["room_id"] = action.get("location") or action.get("target")
    if not isinstance(action, dict) or action.get("action") not in ACTION_NAMES:
        raise RuntimeError(f"HoloAgent returned unsupported action: {action!r}")
    response["action"] = action
    response.setdefault("done", False)
    return response


class HoloAgentAdapter:
    def __init__(self, endpoint: str | None, timeout: float, dry_run: bool = False):
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self.dry_run = dry_run
        self.memory: dict[str, Any] = {}
        self.scene_id: str | None = None

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.dry_run:
            return {"action": {"action": "no_op"}, "done": False,
                    "memory_update": {"adapter_dry_run": True},
                    **({'sequence_reset_ack': payload['sequence_context']['run_id']}
                       if payload.get('sequence_context', {}).get('reset_memory') else {})}
        if not self.endpoint:
            raise RuntimeError("set --endpoint or HOLOAGENT_ENDPOINT, or use --dry-run")
        response = _post_json(self.endpoint, {
            "protocol_version": "1.0", "source": "longterm_benchmark",
            "request": payload,
        }, self.timeout)
        return normalize_response(response) if payload.get("type") == "step" else response

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_type = payload.get("type")
        if request_type == "session_start":
            self.scene_id = payload.get("scene_id")
            self.memory = dict(payload.get("memory") or {})
        elif request_type == "step":
            self.memory.update(payload.get("memory") or {})
        response = self.request(payload)
        sequence = payload.get('sequence_context', {})
        if sequence.get('reset_memory') and response.get('sequence_reset_ack') != sequence['run_id']:
            raise RuntimeError('HoloAgent gateway must reset/isolate its sequence state and acknowledge run_id')
        update = response.get("memory_update")
        if isinstance(update, dict):
            self.memory.update(update)
        return response


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.getenv("HOLOAGENT_ENDPOINT"),
                        help="AgentOS/FSR gateway URL; or set HOLOAGENT_ENDPOINT")
    parser.add_argument("--timeout", type=float,
                        default=float(os.getenv("HOLOAGENT_TIMEOUT", "30")))
    parser.add_argument("--dry-run", action="store_true",
                        help="keep protocol alive without calling HoloAgent")
    args = parser.parse_args()
    adapter = HoloAgentAdapter(args.endpoint, args.timeout, args.dry_run)
    serve(adapter.handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
