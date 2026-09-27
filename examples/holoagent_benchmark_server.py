#!/usr/bin/env python3
"""Expose one HoloAgent-compatible benchmark service endpoint.

This is the process that replaces a model server in a two-terminal run.  It
forwards benchmark Protocol v1 requests to a HoloAgent planner endpoint when
``--planner-url`` is supplied.  The planner must return either a Protocol v1
action or ``{"relative_nav": {"forward", "left", "rotation"}}``.  The
included command gateway remains available as a fallback for HoloAgent ROS /
HTTP skills, so ``/api/relative_nav`` commands are converted into benchmark
primitive actions.

With ``--live-ovo`` it also starts HoloAgent's online RGB-D mapper in-process,
so the same service consumes Isaac Sim ROS topics and produces navigation
actions without a pre-generated HMSG graph.  The Isaac Sim process is still
started by ``run_benchmark.sh``; this script does not import Isaac Sim.
"""
from __future__ import annotations

import argparse
import json
from http.server import ThreadingHTTPServer
import math
from pathlib import Path
import sys
from typing import Any
from urllib import error, request

try:
    from examples.holoagent_agent import normalize_response
    from examples.holoagent_sim_gateway import (
        GatewayHandler,
        RosBridgeNode,
        SimGatewayState,
        _load_waypoints,
    )
except ModuleNotFoundError:  # direct ``python examples/<script>.py`` launch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples.holoagent_agent import normalize_response
    from examples.holoagent_sim_gateway import (
        GatewayHandler,
        RosBridgeNode,
        SimGatewayState,
        _load_waypoints,
    )


def _pose_xy(pose: Any) -> tuple[float, float] | None:
    if not isinstance(pose, dict):
        return None
    position = pose.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    try:
        return float(position[0]), float(position[1])
    except (TypeError, ValueError):
        return None


def _pose_yaw(pose: Any) -> float | None:
    if not isinstance(pose, dict):
        return None
    value = pose.get("orientation")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, z, w = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class FsrVlnPlanner:
    """Small action layer over HoloAgent's existing FSR-VLN query API.

    FSR-VLN returns a scene-graph target, not a primitive action.  This class
    supplies the missing benchmark-facing control layer: query the target at
    task start, then use only the observation pose to emit bounded turn/move
    primitives until the target is reached.  It does not read benchmark
    ground truth.
    """

    def __init__(self, config_path: str, *, fsr_root: str | None = None,
                 use_hmsg_graph: bool = False, use_gpt: bool = False,
                 tolerance_m: float = 0.35, max_segment_m: float = 0.5,
                 turn_threshold_degrees: float = 10.0) -> None:
        root = Path(fsr_root or "").expanduser() if fsr_root else None
        if root is not None:
            sys.path.insert(0, str(root))
        try:
            from omegaconf import OmegaConf
            from fsr_vln import FsrVlnClient
        except Exception as exc:
            raise RuntimeError(
                "FSR-VLN planner requires HoloAgent fsr_vln dependencies: "
                f"{exc}") from exc
        self._client = FsrVlnClient(
            OmegaConf.load(str(Path(config_path).expanduser())),
            use_hmsg_graph=use_hmsg_graph,
        )
        self._use_gpt = bool(use_gpt)
        self._tolerance = max(0.05, float(tolerance_m))
        self._max_segment = max(0.05, float(max_segment_m))
        self._turn_threshold = math.radians(max(1.0, float(turn_threshold_degrees)))
        self._session_instruction: Any = None
        self._task_instruction: str | None = None
        self._target: tuple[float, float] | None = None

    @staticmethod
    def _instruction_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            rooms = value.get("rooms")
            if isinstance(rooms, list):
                return " ".join(str(item.get("text", "")) for item in rooms
                                 if isinstance(item, dict)).strip()
        return ""

    def _query_target(self, instruction: str) -> None:
        if not instruction:
            return
        result = self._client.query(instruction, top_k=1, use_gpt=self._use_gpt)
        if not result.targets:
            self._target = None
            return
        center = result.targets[0].center_map
        if len(center) < 2:
            self._target = None
            return
        self._target = (float(center[0]), float(center[1]))

    def reset(self) -> None:
        self._task_instruction = None
        self._target = None

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_type = payload.get("type")
        if request_type == "session_start":
            self._session_instruction = payload.get("instruction")
            self.reset()
            return {"memory_update": {"holoagent_planner": "fsr_vln_ready"}}
        if request_type == "qa":
            return {"answer": ""}
        if request_type != "step":
            return {"action": {"action": "no_op"}, "done": True}
        if payload.get("task_type") != "navigation":
            return {"action": {"action": "no_op"}, "done": False,
                    "gateway_status": "fsr_vln_navigation_only"}

        instruction = self._instruction_text(payload.get("instruction"))
        # Mapping is asynchronous: the first step can arrive before the
        # mapper has produced its first OVO instance.  Retry while unresolved
        # instead of caching a permanent no-op for the whole task.
        if instruction and (instruction != self._task_instruction or self._target is None):
            self._task_instruction = instruction
            self._query_target(instruction)
        if self._target is None:
            return {"action": {"action": "no_op"}, "done": False,
                    "gateway_status": "fsr_vln_target_unresolved"}
        pose = (payload.get("observation") or {}).get("pose")
        position = _pose_xy(pose)
        yaw = _pose_yaw(pose)
        if position is None or yaw is None:
            return {"action": {"action": "no_op"}, "done": False,
                    "gateway_status": "waiting_for_agent_pose"}
        dx, dy = self._target[0] - position[0], self._target[1] - position[1]
        distance = math.hypot(dx, dy)
        if distance <= self._tolerance:
            return {"action": {"action": "no_op"}, "done": False,
                    "gateway_status": "target_reached", "target": list(self._target)}
        error_angle = (math.atan2(dy, dx) - yaw + math.pi) % (2.0 * math.pi) - math.pi
        if abs(error_angle) > self._turn_threshold:
            return {"action": {"action": "turn_left" if error_angle > 0 else "turn_right",
                                "angle_degrees": min(45.0, max(5.0, abs(math.degrees(error_angle))))},
                    "done": False, "gateway_status": "target_tracking",
                    "target": list(self._target)}
        return {"action": {"action": "move_forward",
                            "distance": min(distance, self._max_segment)},
                "done": False, "gateway_status": "target_tracking",
                "target": list(self._target)}


class LiveOvoPlanner:
    """Query the OVO representation being updated by the online mapper."""

    def __init__(self, mapper: Any, *, tolerance_m: float = 0.35,
                 max_segment_m: float = 0.5,
                 turn_threshold_degrees: float = 10.0) -> None:
        self.mapper = mapper
        self._tolerance = max(0.05, float(tolerance_m))
        self._max_segment = max(0.05, float(max_segment_m))
        self._turn_threshold = math.radians(max(1.0, float(turn_threshold_degrees)))
        self._task_instruction: str | None = None
        self._target: tuple[float, float] | None = None

    def _query_target(self, instruction: str) -> None:
        detector = self.mapper.obj_detect_track
        slam = self.mapper.slam_backbone
        if detector is None or slam is None:
            self._target = None
            return
        with detector.objects_lock:
            objects = list(detector.objects.values())
            if not objects:
                self._target = None
                return
            scores = detector.query([instruction])
            best_index = int(scores[0].argmax().item())
            if best_index >= len(objects):
                self._target = None
                return
            object_id = int(objects[best_index].id)
            points, _, point_object_ids = slam.get_map()
            points_np = (points.detach().cpu().numpy()
                         if hasattr(points, "detach") else points)
            ids_np = (point_object_ids.detach().cpu().numpy()
                       if hasattr(point_object_ids, "detach") else point_object_ids)
            if hasattr(ids_np, "reshape"):
                ids_np = ids_np.reshape(-1).tolist()
            else:
                ids_np = list(ids_np)
            selected = [point for point, value in zip(points_np, ids_np)
                        if int(value) == object_id]
            if not selected:
                self._target = None
                return
            self._target = (
                sum(float(point[0]) for point in selected) / len(selected),
                sum(float(point[1]) for point in selected) / len(selected),
            )

    def _status(self) -> dict[str, Any]:
        """Expose mapper readiness without exposing model internals."""
        detector = getattr(self.mapper, "obj_detect_track", None)
        objects = getattr(detector, "objects", {}) if detector is not None else {}
        return {
            "mapper_frame_id": int(getattr(self.mapper, "frame_id", -1)),
            "mapper_object_count": len(objects) if hasattr(objects, "__len__") else 0,
            "mapper_rgb_messages": int(getattr(self.mapper, "rgb_message_count", 0)),
            "mapper_depth_messages": int(getattr(self.mapper, "depth_message_count", 0)),
            "mapper_pose_messages": int(getattr(self.mapper, "pose_message_count", 0)),
            "mapper_processing_frame_id": int(getattr(self.mapper, "processing_frame_id", -1)),
            "mapper_initialization_error": getattr(self.mapper, "initialization_error", None),
            "mapper_processing_error": getattr(self.mapper, "processing_error", None),
        }

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_type = payload.get("type")
        if request_type == "session_start":
            self._task_instruction = None
            self._target = None
            return {"memory_update": {"holoagent_planner": "live_ovo_ready"}}
        if request_type == "qa":
            return {"answer": ""}
        if request_type != "step":
            return {"action": {"action": "no_op"}, "done": True}
        if payload.get("task_type") != "navigation":
            return {"action": {"action": "no_op"}, "done": False,
                    "gateway_status": "live_ovo_navigation_only"}
        status = self._status()
        instruction = FsrVlnPlanner._instruction_text(payload.get("instruction"))
        # Mapping is asynchronous: the first step can arrive before the
        # mapper has produced its first OVO instance. Retry while unresolved
        # instead of caching a permanent no-op for the whole task.
        if instruction and (instruction != self._task_instruction or self._target is None):
            self._task_instruction = instruction
            self._query_target(instruction)
        if self._target is None:
            if status["mapper_initialization_error"] or status["mapper_processing_error"]:
                unresolved_status = "live_ovo_mapper_error"
            elif status["mapper_frame_id"] < 0:
                unresolved_status = "live_ovo_waiting_for_sensor_frames"
            else:
                unresolved_status = "live_ovo_target_unresolved"
            return {"action": {"action": "no_op"}, "done": False,
                    "gateway_status": unresolved_status,
                    **status}
        pose = (payload.get("observation") or {}).get("pose")
        position = _pose_xy(pose)
        yaw = _pose_yaw(pose)
        if position is None or yaw is None:
            return {"action": {"action": "no_op"}, "done": False,
                    "gateway_status": "waiting_for_agent_pose"}
        dx, dy = self._target[0] - position[0], self._target[1] - position[1]
        distance = math.hypot(dx, dy)
        if distance <= self._tolerance:
            return {"action": {"action": "no_op"}, "done": False,
                    "gateway_status": "target_reached", "target": list(self._target)}
        error_angle = (math.atan2(dy, dx) - yaw + math.pi) % (2.0 * math.pi) - math.pi
        if abs(error_angle) > self._turn_threshold:
            return {"action": {"action": "turn_left" if error_angle > 0 else "turn_right",
                                "angle_degrees": min(45.0, max(5.0, abs(math.degrees(error_angle))))},
                    "done": False, "gateway_status": "target_tracking",
                    "target": list(self._target)}
        return {"action": {"action": "move_forward",
                            "distance": min(distance, self._max_segment)},
                "done": False, "gateway_status": "target_tracking",
                "target": list(self._target)}


def _start_live_ovo(args: argparse.Namespace) -> tuple[LiveOvoPlanner, Any, Any]:
    """Start HoloAgent's online mapper and return its live planner handle."""
    if not args.mapping_config or not args.mapping_root:
        raise RuntimeError("--live-ovo requires --mapping-root and --mapping-config")
    root = Path(args.mapping_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    try:
        import rclpy
        from ovo.entities.semantic_mapping_online import SemanticMapping
        from ovo.utils import io_utils
    except Exception as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name == "rclpy":
            raise RuntimeError(
                "live OVO requires ROS 2 rclpy. The current Python interpreter "
                f"({sys.executable}) cannot import rclpy. Create/activate the "
                "HoloAgent environment and source a ROS 2 setup.bash before "
                "starting this server; do not continue after a failed `conda "
                "activate`. On Ubuntu 24.04 use ROS 2 Jazzy."
            ) from exc
        raise RuntimeError(f"live OVO requires HoloAgent mapping dependencies: {exc}") from exc
    config_path = Path(args.mapping_config).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    config = io_utils.load_config(str(config_path))
    dataset_cfg = root / "configs" / args.mapping_dataset / f"{args.mapping_dataset.lower()}.yaml"
    if dataset_cfg.is_file():
        io_utils.update_recursive(config, io_utils.load_config(str(dataset_cfg)))
    if args.mapping_device:
        config["device"] = str(args.mapping_device)
    slam_module = config.get("slam", {}).get("slam_module")
    slam_cfg = root / config.get("slam", {}).get("config_path", "configs/slam") / str(slam_module) / f"{args.mapping_dataset.lower()}.yaml"
    if slam_cfg.is_file():
        io_utils.update_recursive(config, io_utils.load_config(str(slam_cfg)))
    config.setdefault("data", {})["scene_name"] = args.mapping_scene
    config["data"]["input_path"] = str(Path(args.mapping_data_dir).expanduser() / "input")
    # Isaac Sim ROS2 camera helpers use sensor-data QoS (BEST_EFFORT).  The
    # online mapper otherwise defaults to RELIABLE and the RGB/Depth/Pose
    # synchronizer receives zero tuples, leaving every navigation target
    # unresolved.  Keep the override local to the benchmark integration; the
    # original HoloAgent config remains unchanged for real ZED runs.
    config.setdefault("ros", {})["reliability"] = "BEST_EFFORT"
    # Isaac Sim camera helpers and the Python PoseStamped publisher may stamp
    # messages from different clocks.  Pair the latest received RGB/Depth/Pose
    # messages in the mapper instead of dropping every tuple in ATS.
    config.setdefault("topic", {})["sync_mode"] = "latest"
    config["use_wandb"] = False
    output = Path(args.mapping_output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    mapper = SemanticMapping(config, str(output), scene=args.mapping_scene)
    if getattr(mapper, "initialization_error", None):
        rclpy.shutdown()
        raise RuntimeError(
            "HoloAgent mapper initialization failed; refusing to serve no-op "
            f"actions: {mapper.initialization_error}")
    thread = __import__("threading").Thread(target=mapper.spin, name="holoagent-live-ovo", daemon=True)
    thread.start()
    return LiveOvoPlanner(mapper, tolerance_m=args.position_tolerance_m,
                          max_segment_m=args.max_segment_m), mapper, rclpy
def _post_json(url: str, value: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(value, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"planner request failed: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("planner response must be a JSON object")
    return result.get("response", result)


class PlannerGatewayHandler(GatewayHandler):
    """Reuse the command gateway while optionally delegating step decisions."""

    planner_url: str | None = None
    planner_timeout: float = 30.0
    fsr_planner: FsrVlnPlanner | None = None

    def _planner_response(self, body: dict[str, Any]) -> dict[str, Any] | None:
        value = body.get("request") if isinstance(body.get("request"), dict) else body
        if not isinstance(value, dict):
            raise ValueError("benchmark request must be an object")
        if self.fsr_planner is not None:
            return self.fsr_planner.plan(value)
        if not self.planner_url:
            return None
        response = _post_json(self.planner_url, {
            "protocol_version": "1.0",
            "source": "longterm_benchmark_server",
            "request": value,
        }, self.planner_timeout)
        if value.get("type") == "step":
            return normalize_response(response)
        return response

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        from urllib.parse import urlparse
        if (urlparse(self.path).path == "/benchmark/step"
                and (self.planner_url or self.fsr_planner is not None)):
            try:
                body = self._json()
                response = self._planner_response(body)
                self._write(200, response or {"action": {"action": "no_op"}})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._write(400, {"success": False, "detail": str(exc)})
            except Exception as exc:
                self._write(502, {"success": False, "detail": str(exc)})
            return
        super().do_POST()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--planner-url",
                        help="HoloAgent planner endpoint receiving Protocol v1 requests")
    parser.add_argument("--planner-timeout", type=float, default=30.0)
    parser.add_argument("--fsr-config",
                        help="HoloAgent FSR-VLN Hydra config; enables built-in target planner")
    parser.add_argument("--fsr-root",
                        help="HoloAgent agentic_robot/fsr_vln directory")
    parser.add_argument("--fsr-use-hmsg-graph", action="store_true")
    parser.add_argument("--fsr-use-gpt", action="store_true")
    parser.add_argument("--live-ovo", action="store_true",
                        help="run HoloAgent semantic_mapping_online.py in this process")
    parser.add_argument("--mapping-root",
                        help="HoloAgent agentic_robot/fsr_vln directory")
    parser.add_argument("--mapping-config", default="configs/ovo_zed.yaml")
    parser.add_argument("--mapping-dataset", default="Zed")
    parser.add_argument("--mapping-scene", required=False, default="scene000430")
    parser.add_argument("--mapping-data-dir", default="/tmp/holoagent_live_mapping")
    parser.add_argument("--mapping-output", default="/tmp/holoagent_live_mapping/output")
    parser.add_argument("--mapping-device", default=None,
                        help="override HoloAgent mapper torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--ros", action="store_true",
                        help="subscribe to HoloAgent ROS command/object-pose topics")
    parser.add_argument("--command-wait-ms", type=float, default=150.0)
    parser.add_argument("--position-tolerance-m", type=float, default=0.25)
    parser.add_argument("--max-segment-m", type=float, default=0.5)
    parser.add_argument("--waypoints", help="JSON file or inline waypoint map")
    args = parser.parse_args()
    if args.planner_timeout <= 0:
        parser.error("--planner-timeout must be positive")
    if sum(bool(value) for value in (args.planner_url, args.fsr_config, args.live_ovo)) > 1:
        parser.error("use only one of --planner-url, --fsr-config, or --live-ovo")
    if args.live_ovo and args.ros:
        parser.error("--live-ovo already owns the ROS context; omit --ros")
    fsr_planner = None
    if args.fsr_config:
        fsr_planner = FsrVlnPlanner(
            args.fsr_config,
            fsr_root=args.fsr_root,
            use_hmsg_graph=args.fsr_use_hmsg_graph,
            use_gpt=args.fsr_use_gpt,
            tolerance_m=args.position_tolerance_m,
            max_segment_m=args.max_segment_m,
        )
    live_mapper = None
    live_rclpy = None
    if args.live_ovo:
        fsr_planner, live_mapper, live_rclpy = _start_live_ovo(args)
    state = SimGatewayState(
        command_wait_s=max(0.0, args.command_wait_ms / 1000.0),
        position_tolerance_m=args.position_tolerance_m,
        max_segment_m=args.max_segment_m,
        waypoints=_load_waypoints(args.waypoints),
    )
    handler = type("ConfiguredPlannerGatewayHandler", (PlannerGatewayHandler,), {
        "state": state,
        "planner_url": args.planner_url,
        "planner_timeout": args.planner_timeout,
        "fsr_planner": fsr_planner,
    })
    server = ThreadingHTTPServer((args.host, args.port), handler)
    ros = None
    if args.ros:
        ros = RosBridgeNode(state)
        import threading
        threading.Thread(target=ros.spin, name="holoagent-benchmark-ros", daemon=True).start()
    print(f"holoagent_benchmark_server listening on http://{args.host}:{args.port}", flush=True)
    if args.planner_url:
        print(f"planner endpoint: {args.planner_url}", flush=True)
    elif fsr_planner is not None:
        print("planner: built-in live OVO planner" if args.live_ovo
              else "planner: built-in FSR-VLN target planner", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        if ros is not None:
            ros.close()
        if live_mapper is not None:
            try:
                live_mapper.shutdown_event.set()
                live_mapper.destroy_node()
            finally:
                if live_rclpy is not None and live_rclpy.ok():
                    live_rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
