#!/usr/bin/env python3
"""SimpleNav JSONL adapter for :mod:`benchmark.run`.

The adapter keeps the benchmark protocol independent from SimpleNav.  Navigation
requests are converted to SimpleNav waypoint queries; unsupported QA and
manipulation requests are answered as failures so the benchmark can continue.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
from typing import Any
from urllib.parse import urlparse

import numpy as np


def _decode(value: Any) -> Any:
    """Decode arrays produced by benchmark.runtime.session_runner._jsonable."""
    if isinstance(value, dict):
        if value.get("encoding") == "base64" and {"dtype", "shape", "data"} <= value.keys():
            raw = base64.b64decode(value["data"])
            return np.frombuffer(raw, dtype=np.dtype(value["dtype"])).reshape(value["shape"]).copy()
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _url_parts(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        raise ValueError(f"--url must be a websocket URL, got {url!r}")
    return parsed.hostname, int(parsed.port or 10093)


def _instruction_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # New scene packages carry compact room introductions.  The benchmark
        # sends the rendered images separately in session_views; this text is
        # the persistent initial scene memory used by the policy.
        rooms = value.get("rooms")
        if isinstance(rooms, list):
            return " ".join(str(room.get("text", "")) for room in rooms
                            if isinstance(room, dict) and room.get("text")).strip()
    return str(value or "")


def _simple_action(raw: np.ndarray) -> dict[str, Any]:
    """Project a SimpleNav local waypoint onto benchmark high-level actions."""
    waypoint = np.asarray(raw, dtype=np.float32).reshape(-1, 4)[0]
    dx, dy, _dz, dyaw = (float(item) for item in waypoint)
    if not all(math.isfinite(item) for item in (dx, dy, dyaw)):
        return {"action": "no_op"}
    if abs(dyaw) >= 0.05 and abs(dyaw) >= max(abs(dx), abs(dy)):
        return {"action": "turn_right" if dyaw > 0 else "turn_left",
                "angle_degrees": max(1.0, math.degrees(abs(dyaw)))}
    if abs(dx) >= 0.05:
        return {"action": "move_forward" if dx > 0 else "move_backward",
                "distance": max(0.05, abs(dx))}
    if abs(dy) >= 0.05:
        return {"action": "move_right" if dy > 0 else "move_left",
                "distance": max(0.05, abs(dy))}
    return {"action": "no_op"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:10093")
    parser.add_argument("--checkpoint", required=True,
                        help="SimpleNav checkpoint; its dataset_statistics.json is used for unnormalization")
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--history-frames", type=int, default=8)
    parser.add_argument("--collision-reset-threshold", type=int, default=5,
                        help="reset after this many consecutive blocked moves")
    args = parser.parse_args()

    # These imports intentionally happen after argument parsing: this script is
    # run by the SimpleNav environment, not by Isaac Sim's Python.
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
    from NavVLAeval.common.data.runtime_dataset import (
        OnlineNavVLARuntimeDatasetAdapter,
        VLNCE_PLATFORM_TEXT,
    )
    from NavVLAeval.common.model.action_codec import ActionCodec
    from NavVLAeval.common.types import EpisodeHistory

    host, port = _url_parts(args.url)
    policy = WebsocketClientPolicy(host=host, port=port)
    codec = ActionCodec.from_checkpoint(args.checkpoint, unnorm_key=args.unnorm_key)
    adapter = OnlineNavVLARuntimeDatasetAdapter(
        required_cameras=("front",),
        history_image_frames=args.history_frames,
        history_selection="recent",
        platform_text=VLNCE_PLATFORM_TEXT,
    )
    history = EpisodeHistory()
    instruction = ""
    session_instruction = ""
    step_index = 0
    scene_id = None
    task_id = None
    collision_streak = 0

    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            request = _decode(json.loads(line))
            request_type = request.get("type")
            if request_type == "session_start":
                next_scene_id = request.get("scene_id")
                if next_scene_id != scene_id:
                    history = EpisodeHistory()
                    step_index = 0
                    scene_id = next_scene_id
                task_id = None
                collision_streak = 0
                session_instruction = _instruction_text(request.get("instruction"))
                instruction = session_instruction
                # Seed SimpleNav's visual history with the authored room views
                # captured by the benchmark before the task starts.
                for view in request.get("session_views") or []:
                    view_observation = view.get("observation") or {}
                    image = (view_observation.get("images") or {}).get("front")
                    if image is not None:
                        history.images.append(image)
                        history.observations.append(view_observation)
                        history.instructions.append(str(view.get("text") or session_instruction))
                response = {"memory_update": {}}
            elif request_type == "qa":
                # SimpleNav is navigation-only.  Empty answer is deliberately
                # scored as a failed QA task by the benchmark.
                response = {"answer": "", "done": True}
            elif request_type == "step":
                task_type = str(request.get("task_type", "navigation"))
                if task_type != "navigation":
                    response = {"action": {"action": "no_op"}, "done": True}
                else:
                    current_task_id = request.get("task_id")
                    if current_task_id != task_id:
                        task_id = current_task_id
                        collision_streak = 0
                    instruction = str(request.get("instruction") or instruction)
                    observation_input = request.get("observation") or {}
                    feedback = observation_input.get("action_feedback") or request.get("previous_result") or {}
                    nearby_doors = observation_input.get("nearby_doors") or []
                    closed_door = next((door for door in nearby_doors
                                        if not door.get("open") and float(door.get("distance_m", 999.0)) <= 1.0), None)
                    if closed_door is not None and (
                            bool(feedback.get("collision")) or
                            float(closed_door.get("distance_m", 999.0)) <= 0.6):
                        action = {"action": "open", "object_id": closed_door["object_id"],
                                  "source": "adapter_auto", "agent_requested": False}
                        print(
                            f"step={step_index} task={request.get('task_id')} "
                            f"action=open source=adapter_auto object_id={closed_door['object_id']}",
                            file=sys.stderr, flush=True)
                        response = {"action": action, "done": False,
                                    "action_source": "adapter_auto", "agent_requested": False}
                        collision_streak = 0
                        step_index += 1
                        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                        sys.stdout.flush()
                        continue
                    if bool(feedback.get("collision")):
                        collision_streak += 1
                    else:
                        collision_streak = 0
                    if collision_streak >= max(1, args.collision_reset_threshold):
                        print(
                            f"step={step_index} task={request.get('task_id')} "
                            f"collision_streak={collision_streak} action=reset",
                            file=sys.stderr,
                            flush=True,
                        )
                        collision_streak = 0
                        response = {"action": {"action": "reset"}, "done": False}
                        step_index += 1
                        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                        sys.stdout.flush()
                        continue
                    started = time.monotonic()
                    print(
                        f"step={step_index} task={request.get('task_id')} build example",
                        file=sys.stderr,
                        flush=True,
                    )
                    observation = adapter.prepare_observation_for_step(
                        observation=observation_input, step=step_index)
                    model_instruction = instruction
                    if session_instruction and session_instruction != instruction:
                        model_instruction = (
                            "Scene memory (initial room descriptions): " + session_instruction
                            + "\nCurrent task: " + instruction)
                    if bool(feedback.get("collision")):
                        model_instruction += (
                            " Previous movement was blocked by a collision; the attempted "
                            "distance was only partially executed. Choose a different "
                            "walkable direction or distance."
                        )
                    example = adapter.build_example(
                        observation=observation, history=history, instruction=model_instruction)
                    try:
                        print(
                            f"step={step_index} task={request.get('task_id')} server inference",
                            file=sys.stderr,
                            flush=True,
                        )
                        server_response = policy.predict_action({"examples": [example]})
                        if server_response.get("ok") is False:
                            error = server_response.get("error") or {}
                            raise RuntimeError(error.get("message", error))
                        model_response = server_response.get("data", server_response)
                        prediction = codec.decode(model_response)
                        adapter.update_history(
                            history=history, observation=observation,
                            prediction=prediction, instruction=model_instruction)
                        action = _simple_action(prediction.raw_actions)
                        response = {"action": action, "done": False}
                        print(
                            f"step={step_index} task={request.get('task_id')} "
                            f"inference_seconds={time.monotonic() - started:.2f} action={action['action']}",
                            file=sys.stderr,
                            flush=True,
                        )
                    except Exception as exc:
                        # A model-side inability is a failed task, not a failed
                        # benchmark run.  Keep diagnostics out of stdout.
                        print(f"simplenav inference failed: {exc}", file=sys.stderr)
                        response = {"action": {"action": "no_op"}, "done": True}
                    step_index += 1
            else:
                raise ValueError(f"unknown request type: {request_type!r}")
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    finally:
        policy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
