#!/usr/bin/env python3
"""HTTP/ROS command bridge for running HoloAgent against Isaac Sim.

This is a transport bridge, not a replacement for HoloAgent's planner.  The
benchmark-side adapter calls ``POST /benchmark/step`` while HoloAgent (or its
robot-service skill) may send the documented robot commands to the same
process:

* ``POST /api/relative_nav`` with ``{"cmd":"forward,left,degrees"}``;
* ``POST /api/semantic_nav`` with ``{"cmd":"floor,room,object"}``;
* ``POST /api/navigation/{name}`` for a configured simulated waypoint.

The bridge converts a queued command into one Protocol v1 action.  Relative
commands are split into movement/rotation actions because the benchmark action
space has one primitive per step.  Semantic navigation waits for a
``/object_pose`` ROS message (``geometry_msgs/msg/PoseStamped``), then uses a
small closed-loop controller based on the pose included in the benchmark
observation.  No benchmark ground truth is read by this process.

The implementation uses only the Python standard library unless ``--ros`` is
specified.  ROS 2 support is optional so the HTTP-only command bridge can be
tested on a CPU-only machine.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse


def _finite_number(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def parse_relative_command(value: Any) -> tuple[float, float, float]:
    """Parse HoloAgent's canonical ``forward,left,degrees`` string."""
    if not isinstance(value, str):
        raise ValueError("relative navigation cmd must be a string")
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError('expected "forward,left,degrees"')
    return tuple(_finite_number(part, name) for part, name in zip(
        parts, ("forward", "left", "degrees")))  # type: ignore[return-value]


def relative_to_actions(value: Any, *, command_id: str | None = None) -> list[dict[str, Any]]:
    """Project one HoloAgent relative target onto benchmark primitives."""
    forward, left, degrees = parse_relative_command(value)
    metadata = {"command_id": command_id} if command_id else {}
    actions: list[dict[str, Any]] = []
    # Translation is emitted before rotation so it uses the original heading,
    # matching relative_goal_node's target computation as closely as possible.
    if abs(forward) > 1e-6:
        actions.append({"action": "move_forward" if forward > 0 else "move_backward",
                        "distance": abs(forward), **metadata})
    if abs(left) > 1e-6:
        actions.append({"action": "move_left" if left > 0 else "move_right",
                        "distance": abs(left), **metadata})
    if abs(degrees) > 1e-6:
        actions.append({"action": "turn_left" if degrees > 0 else "turn_right",
                        "angle_degrees": abs(degrees), **metadata})
    if not actions:
        actions.append({"action": "no_op", **metadata})
    return actions


def _yaw_from_quaternion(value: Any) -> float | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, z, w = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _pose_position(pose: Any) -> tuple[float, float] | None:
    if not isinstance(pose, dict):
        return None
    position = pose.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    try:
        return float(position[0]), float(position[1])
    except (TypeError, ValueError):
        return None


def _normalise_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class TargetPose:
    position: tuple[float, float]
    waypoint_name: str | None = None
    command_id: str | None = None
    robot_id: int | None = None


class SimGatewayState:
    """Thread-safe command queue shared by HTTP and optional ROS callbacks."""

    def __init__(self, *, command_wait_s: float = 0.15,
                 position_tolerance_m: float = 0.25,
                 max_segment_m: float = 0.5,
                 turn_threshold_degrees: float = 10.0,
                 waypoints: dict[str, Any] | None = None) -> None:
        self.command_wait_s = max(0.0, float(command_wait_s))
        self.position_tolerance_m = max(0.01, float(position_tolerance_m))
        self.max_segment_m = max(0.01, float(max_segment_m))
        self.turn_threshold = math.radians(max(1.0, float(turn_threshold_degrees)))
        self.waypoints = waypoints or {}
        self._condition = threading.Condition()
        self._actions: list[dict[str, Any]] = []
        self._target: TargetPose | None = None
        self._semantic_query: str | None = None
        self._last_command: dict[str, Any] | None = None
        self._completed: set[str] = set()
        self._reached_targets: dict[int, str] = {}
        self._on_reached: Callable[[str], None] | None = None
        self._on_semantic_query: Callable[[str], None] | None = None

    def set_on_reached(self, callback: Callable[[str], None] | None) -> None:
        self._on_reached = callback

    def set_on_semantic_query(self, callback: Callable[[str], None] | None) -> None:
        self._on_semantic_query = callback

    @staticmethod
    def new_command_id() -> str:
        return uuid.uuid4().hex[:12]

    def reset_session(self) -> None:
        with self._condition:
            self._actions.clear()
            self._target = None
            self._semantic_query = None
            self._last_command = None
            self._completed.clear()
            self._reached_targets.clear()

    def enqueue_relative(self, cmd: str) -> dict[str, Any]:
        command_id = self.new_command_id()
        actions = relative_to_actions(cmd, command_id=command_id)
        with self._condition:
            self._actions.extend(actions)
            self._last_command = {"command_id": command_id, "kind": "relative_nav",
                                  "cmd": cmd, "accepted_at": time.time()}
            self._condition.notify_all()
        return {"success": True, "accepted": True, "command_id": command_id,
                "execution": "benchmark_step"}

    def enqueue_semantic(self, cmd: str, *, forward_to_ros: bool = False) -> dict[str, Any]:
        parts = [part.strip() for part in cmd.split(",")]
        if len(parts) != 3 or not all(parts):
            raise ValueError('expected "floor,room,object"')
        command_id = self.new_command_id()
        with self._condition:
            self._semantic_query = cmd
            self._last_command = {"command_id": command_id, "kind": "semantic_nav",
                                  "cmd": cmd, "accepted_at": time.time()}
            self._condition.notify_all()
        if forward_to_ros and self._on_semantic_query is not None:
            self._on_semantic_query(cmd)
        return {"success": True, "accepted": True, "command_id": command_id,
                "execution": "waiting_for_object_pose"}

    def set_target_pose(self, position: Any, *, waypoint_name: str | None = None,
                        command_id: str | None = None, robot_id: int | None = None) -> None:
        if isinstance(position, dict):
            parsed = _pose_position(position)
        elif isinstance(position, (list, tuple)) and len(position) >= 2:
            parsed = (float(position[0]), float(position[1]))
        else:
            parsed = None
        if parsed is None or not all(math.isfinite(item) for item in parsed):
            raise ValueError("target pose requires finite x,y position")
        with self._condition:
            self._target = TargetPose(parsed, waypoint_name, command_id, robot_id)
            self._condition.notify_all()

    def set_waypoint(self, name: str, *, robot_id: int | None = None) -> dict[str, Any]:
        value = self.waypoints.get(name)
        if value is None:
            return {"success": False, "accepted": False,
                    "detail": f"simulated waypoint {name!r} is not configured"}
        command_id = self.new_command_id()
        self.set_target_pose(value, waypoint_name=name, command_id=command_id,
                             robot_id=robot_id)
        with self._condition:
            self._last_command = {"command_id": command_id, "kind": "navigation",
                                  "name": name, "robot_id": robot_id,
                                  "accepted_at": time.time()}
        return {"success": True, "accepted": True, "command_id": command_id,
                "execution": "benchmark_step"}

    def stop(self) -> dict[str, Any]:
        with self._condition:
            self._actions.clear()
            self._target = None
            self._semantic_query = None
            self._condition.notify_all()
        return {"success": True, "accepted": True, "execution": "stopped"}

    def set_ros_object_pose(self, pose: Any) -> None:
        # PoseStamped is converted by RosBridgeNode before reaching here.
        with self._condition:
            command_id = (self._last_command or {}).get("command_id")
        self.set_target_pose(pose, command_id=command_id)

    def _controller_action(self, request: dict[str, Any]) -> dict[str, Any] | None:
        with self._condition:
            target = self._target
        if target is None:
            return None
        current = _pose_position((request.get("observation") or {}).get("pose"))
        if current is None:
            return {"action": {"action": "no_op"}, "done": False,
                    "gateway_status": "waiting_for_agent_pose",
                    "command_id": target.command_id}
        dx, dy = target.position[0] - current[0], target.position[1] - current[1]
        distance = math.hypot(dx, dy)
        if distance <= self.position_tolerance_m:
            with self._condition:
                self._target = None
                self._semantic_query = None
                command_id = target.command_id
                waypoint_name = target.waypoint_name
                if command_id:
                    self._completed.add(command_id)
                if target.robot_id is not None and waypoint_name:
                    self._reached_targets[int(target.robot_id)] = waypoint_name
            if waypoint_name and self._on_reached is not None:
                self._on_reached(waypoint_name)
            return {"action": {"action": "no_op"}, "done": False,
                    "gateway_status": "target_reached", "command_id": command_id,
                    "waypoint_name": waypoint_name}
        pose = (request.get("observation") or {}).get("pose") or {}
        yaw = _yaw_from_quaternion(pose.get("orientation"))
        if yaw is None:
            return {"action": {"action": "no_op"}, "done": False,
                    "gateway_status": "waiting_for_agent_orientation",
                    "command_id": target.command_id}
        desired = math.atan2(dy, dx)
        error = _normalise_angle(desired - yaw)
        if abs(error) > self.turn_threshold:
            angle = min(45.0, max(5.0, math.degrees(abs(error))))
            action = {"action": "turn_left" if error > 0 else "turn_right",
                      "angle_degrees": angle}
        else:
            action = {"action": "move_forward", "distance": min(distance, self.max_segment_m)}
        if target.command_id:
            action["command_id"] = target.command_id
        return {"action": action, "done": False, "gateway_status": "target_tracking",
                "command_id": target.command_id}

    def next_response(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = request.get("type")
        if request_type == "session_start":
            self.reset_session()
            sequence = request.get('sequence_context', {})
            return {'memory_update': {'holoagent_sim_gateway': 'connected'},
                    'gateway_status': 'session_ready',
                    **({'sequence_reset_ack': sequence['run_id']} if sequence.get('reset_memory') else {})}
        if request_type != "step":
            return {"action": {"action": "no_op"}, "done": True,
                    "gateway_status": "unknown_request_type"}

        # Pose-based semantic navigation has priority over a stale command
        # queue, but newly queued relative actions are consumed first.
        with self._condition:
            if not self._actions and self._target is None:
                self._condition.wait(timeout=self.command_wait_s)
            if self._actions:
                action = self._actions.pop(0)
                return {"action": action, "done": False,
                        "gateway_status": "command_ready",
                        "command_id": action.get("command_id")}
        controlled = self._controller_action(request)
        if controlled is not None:
            return controlled
        with self._condition:
            status = "waiting_for_object_pose" if self._semantic_query else "waiting_for_holoagent_command"
            command_id = (self._last_command or {}).get("command_id")
        return {"action": {"action": "no_op"}, "done": False,
                "gateway_status": status, "command_id": command_id}

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "pending_actions": len(self._actions),
                "semantic_query": self._semantic_query,
                "target": None if self._target is None else {
                    "position": list(self._target.position),
                    "waypoint_name": self._target.waypoint_name,
                    "command_id": self._target.command_id,
                },
                "last_command": self._last_command,
                "completed_commands": sorted(self._completed),
                "reached_targets": {str(key): value
                                    for key, value in sorted(self._reached_targets.items())},
            }

    def reset_reached(self, robot_ids: list[int] | None = None) -> dict[str, Any]:
        with self._condition:
            if robot_ids:
                for robot_id in robot_ids:
                    self._reached_targets.pop(int(robot_id), None)
            else:
                self._reached_targets.clear()
            return {"reset_robot_ids": robot_ids or [],
                    "remaining_reached": sorted(self._reached_targets)}

    def reached(self, robot_id: int, target: str | None = None) -> dict[str, Any]:
        with self._condition:
            stored = self._reached_targets.get(int(robot_id))
            matched = target is None or stored == target
            return {"robot_id": int(robot_id), "reached": stored is not None and matched,
                    "target": stored, "target_matched": matched}


class GatewayHandler(BaseHTTPRequestHandler):
    state: SimGatewayState

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[holoagent-sim-gateway] {fmt % args}", flush=True)

    def _write(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/health":
            self._write(200, {"status": "ok", "node": "holoagent_sim_gateway"})
        elif path == "/benchmark/state":
            self._write(200, self.state.snapshot())
        else:
            match = re.fullmatch(r"/robot_reached/(\d+)", path)
            if match:
                from urllib.parse import parse_qs
                query = parse_qs(parsed.query)
                target = (query.get("target") or [None])[0]
                self._write(200, self.state.reached(int(match.group(1)), target))
                return
            self._write(404, {"success": False, "detail": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        try:
            body = self._json()
            if path == "/benchmark/step":
                # holoagent_agent.py sends the Protocol v1 request inside the
                # documented integration envelope.  Accept both the envelope
                # and a direct request so the gateway is easy to smoke-test.
                request = body.get("request") if isinstance(body.get("request"), dict) else body
                self._write(200, self.state.next_response(request))
                return
            if path == "/api/relative_nav":
                value = body.get("cmd")
                parse_relative_command(value)
                self._write(200, self.state.enqueue_relative(value))
                return
            if path == "/api/semantic_nav":
                value = body.get("cmd")
                self._write(200, self.state.enqueue_semantic(value, forward_to_ros=True))
                return
            if path == "/reset_reached":
                raw_ids = body.get("robot_ids", [])
                if not isinstance(raw_ids, list):
                    raise ValueError("robot_ids must be an array")
                self._write(200, self.state.reset_reached([int(item) for item in raw_ids]))
                return
            match = re.fullmatch(r"/api/navigation/([^/]+)", path)
            if match:
                name = match.group(1)
                from urllib.parse import parse_qs
                query = parse_qs(urlparse(self.path).query)
                raw_robot_id = (query.get("robot_id") or [None])[0]
                robot_id = None if raw_robot_id is None else int(raw_robot_id)
                result = self.state.stop() if name == "stop" else self.state.set_waypoint(
                    name, robot_id=robot_id)
                self._write(200 if result.get("accepted") else 422, result)
                return
            match = re.fullmatch(r"/trigger/([^/]+)", path)
            if match:
                from urllib.parse import parse_qs
                query = parse_qs(urlparse(self.path).query)
                raw_robot_id = (query.get("robot_id") or [None])[0]
                robot_id = None if raw_robot_id is None else int(raw_robot_id)
                result = self.state.set_waypoint(match.group(1), robot_id=robot_id)
                self._write(200 if result.get("accepted") else 422, result)
                return
            if re.fullmatch(r"/api/arm/[^/]+", path):
                self._write(501, {"success": False, "accepted": False,
                                  "detail": "arm actions are not simulated by IsaacVLNEnv"})
                return
            self._write(404, {"success": False, "detail": "not_found"})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._write(400, {"success": False, "accepted": False, "detail": str(exc)})
        except Exception as exc:  # keep the protocol endpoint JSON-shaped
            self._write(500, {"success": False, "accepted": False, "detail": str(exc)})


class RosBridgeNode:
    """Optional ROS 2 adapter for the documented HoloAgent topics."""

    def __init__(self, state: SimGatewayState) -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import String

        self._rclpy = rclpy
        rclpy.init()
        self.node = rclpy.create_node("holoagent_sim_gateway")
        self._reached_pub = self.node.create_publisher(String, "waypoint_reached", 10)
        self._semantic_pub = self.node.create_publisher(String, "chat_loc_pub", 10)
        self.node.create_subscription(String, "relative_nav",
                                       lambda msg: state.enqueue_relative(msg.data), 10)
        self.node.create_subscription(String, "chat_loc_pub",
                                       lambda msg: state.enqueue_semantic(msg.data), 10)
        self.node.create_subscription(String, "chat_signal_pub",
                                       lambda msg: self._signal(state, msg.data), 10)
        self.node.create_subscription(PoseStamped, "object_pose",
                                       lambda msg: state.set_ros_object_pose([
                                           msg.pose.position.x, msg.pose.position.y]), 10)
        state.set_on_reached(self._publish_reached)
        state.set_on_semantic_query(self._publish_semantic)

    def _signal(self, state: SimGatewayState, value: str) -> None:
        if value.strip() == "stop":
            state.stop()
        else:
            state.set_waypoint(value.strip())

    def _publish_reached(self, name: str) -> None:
        from std_msgs.msg import String
        message = String()
        message.data = str(name)
        self._reached_pub.publish(message)

    def _publish_semantic(self, query: str) -> None:
        from std_msgs.msg import String
        message = String()
        message.data = str(query)
        self._semantic_pub.publish(message)

    def spin(self) -> None:
        self._rclpy.spin(self.node)

    def close(self) -> None:
        self.node.destroy_node()
        self._rclpy.shutdown()


def _load_waypoints(value: str | None) -> dict[str, Any]:
    if not value:
        value = os.getenv("HOLOAGENT_SIM_WAYPOINTS", "")
    if not value:
        return {}
    path = value
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as stream:
                parsed = json.load(stream)
        else:
            parsed = json.loads(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid waypoint map: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("waypoint map must be a JSON object")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--ros", action="store_true",
                        help="subscribe to HoloAgent ROS 2 command/object-pose topics")
    parser.add_argument("--command-wait-ms", type=float, default=150.0)
    parser.add_argument("--position-tolerance-m", type=float, default=0.25)
    parser.add_argument("--max-segment-m", type=float, default=0.5)
    parser.add_argument("--waypoints", help="JSON file or inline map of waypoint names to [x,y]")
    args = parser.parse_args()
    state = SimGatewayState(
        command_wait_s=max(0.0, args.command_wait_ms / 1000.0),
        position_tolerance_m=args.position_tolerance_m,
        max_segment_m=args.max_segment_m,
        waypoints=_load_waypoints(args.waypoints),
    )
    server = ThreadingHTTPServer((args.host, args.port),
                                 type("ConfiguredGatewayHandler", (GatewayHandler,), {"state": state}))
    ros = None
    if args.ros:
        ros = RosBridgeNode(state)
        threading.Thread(target=ros.spin, name="holoagent-sim-ros", daemon=True).start()
    print(f"holoagent_sim_gateway listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        if ros is not None:
            ros.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
