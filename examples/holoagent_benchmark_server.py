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
import os
from pathlib import Path
import re
import sys
from typing import Any
from urllib import error, request

try:
    from examples.holoagent_agent import normalize_response
    from examples.openai_compatible_vlm_agent import (
        Adapter as VisionLanguageAdapter,
        LocalTransformersAdapter,
    )
    from examples.holoagent_sim_gateway import (
        GatewayHandler,
        RosBridgeNode,
        SimGatewayState,
        _load_waypoints,
    )
except ModuleNotFoundError:  # direct ``python examples/<script>.py`` launch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples.holoagent_agent import normalize_response
    from examples.openai_compatible_vlm_agent import (
        Adapter as VisionLanguageAdapter,
        LocalTransformersAdapter,
    )
    from examples.holoagent_sim_gateway import (
        GatewayHandler,
        RosBridgeNode,
        SimGatewayState,
        _load_waypoints,
    )


from benchmark.agents.protocol import PROTOCOL_VERSION


def _navigation_request(payload):
    return payload.get('type') == 'step' and set(payload.get('capabilities', [])) == {'navigation'}


def _unsupported(reason):
    return {'action': 'no_op', 'done': True, 'agent_status': 'unsupported', 'reason_code': reason}


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
        return ""

    @staticmethod
    def _point_target(instruction: str) -> tuple[float, float] | None:
        match = re.search(
            r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
            instruction,
        )
        return (float(match.group(1)), float(match.group(2))) if match else None

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
            sequence = payload.get('sequence_context', {})
            return {'memory_update': {'holoagent_planner': 'fsr_vln_ready'},
                    **({'sequence_reset_ack': sequence['run_id']} if sequence.get('reset_memory') else {})}
        if request_type != "step":
            return {"action": {"action": "no_op"}, "done": True}
        if not _navigation_request(payload):
            return _unsupported('fsr_vln_navigation_only')

        instruction = self._instruction_text(payload.get("instruction"))
        # Mapping is asynchronous: the first step can arrive before the
        # mapper has produced its first OVO instance.  Retry while unresolved
        # instead of caching a permanent no-op for the whole task.
        if instruction and (instruction != self._task_instruction or self._target is None):
            self._task_instruction = instruction
            self._target = self._point_target(instruction)
            if self._target is None:
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
        self._sequence_run_id = None
        self._tolerance = max(0.05, float(tolerance_m))
        self._max_segment = max(0.05, float(max_segment_m))
        self._turn_threshold = math.radians(max(1.0, float(turn_threshold_degrees)))
        self._task_instruction: str | None = None
        self._target: tuple[float, float] | None = None
        self._candidate_object_id: int | None = None
        self._attempted_object_ids: set[int] = set()
        self._awaiting_candidate_check = False
        self._exploration_steps = 0

    def _query_target(self, instruction: str) -> None:
        try:
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
                ranked = scores[0].argsort(descending=True).detach().cpu().tolist()
                best_index = next(
                    (int(index) for index in ranked
                     if int(index) < len(objects)
                     and int(objects[int(index)].id) not in self._attempted_object_ids),
                    None,
                )
                if best_index is None:
                    self._target = None
                    return
                object_id = int(objects[best_index].id)
                points, _, point_object_ids = slam.get_map()
                points_np = (points.detach().cpu().numpy()
                             if hasattr(points, "detach") else points)
                ids_np = (point_object_ids.detach().cpu().numpy()
                           if hasattr(point_object_ids, "detach") else point_object_ids)
                ids_np = ids_np.reshape(-1).tolist() if hasattr(ids_np, "reshape") else list(ids_np)
                selected = [point for point, value in zip(points_np, ids_np)
                            if int(value) == object_id]
                if not selected:
                    self._target = None
                    return
                self._target = (
                    sum(float(point[0]) for point in selected) / len(selected),
                    sum(float(point[1]) for point in selected) / len(selected),
                )
                self._candidate_object_id = object_id
                self._awaiting_candidate_check = False
        except Exception:
            # The online map is updated concurrently; an incomplete snapshot
            # means "not resolved yet", not a failed protocol request.
            self._target = None

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
            sequence = payload.get('sequence_context', {})
            if sequence.get('reset_memory'):
                if self._sequence_run_id is not None and self._sequence_run_id != sequence['run_id']:
                    raise RuntimeError('restart the live OVO service for a fresh sequence map')
                self._sequence_run_id = sequence['run_id']
            self._task_instruction = None
            self._target = None
            self._candidate_object_id = None
            self._attempted_object_ids.clear()
            self._awaiting_candidate_check = False
            self._exploration_steps = 0
            return {'memory_update': {'holoagent_planner': 'live_ovo_ready'},
                    **({'sequence_reset_ack': sequence['run_id']} if sequence.get('reset_memory') else {})}
        if request_type != "step":
            return {"action": {"action": "no_op"}, "done": True}
        if not _navigation_request(payload):
            return _unsupported('live_ovo_navigation_only')
        status = self._status()
        instruction = FsrVlnPlanner._instruction_text(payload.get("instruction"))
        if instruction != self._task_instruction:
            self._task_instruction = instruction
            self._target = None
            self._candidate_object_id = None
            self._attempted_object_ids.clear()
            self._awaiting_candidate_check = False
            self._exploration_steps = 0
        # Mapping is asynchronous: the first step can arrive before the
        # mapper has produced its first OVO instance. Retry while unresolved
        # instead of caching a permanent no-op for the whole task.
        if instruction and self._target is None:
            self._target = FsrVlnPlanner._point_target(instruction)
            if self._target is None:
                self._query_target(instruction)
        if self._target is None:
            if status["mapper_initialization_error"] or status["mapper_processing_error"]:
                unresolved_status = "live_ovo_mapper_error"
            elif status["mapper_frame_id"] < 0:
                unresolved_status = "live_ovo_waiting_for_sensor_frames"
            else:
                unresolved_status = "live_ovo_target_unresolved"
            self._exploration_steps += 1
            if self._exploration_steps % 12 == 0:
                action = {"action": "move_forward", "distance": self._max_segment}
            else:
                action = {"action": "turn_left", "angle_degrees": 30.0}
            return {"action": action, "done": False,
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
            # One no-op lets the benchmark evaluate arrival. If it asks for
            # another step, this was a false semantic match: exclude that OVO
            # instance and continue searching instead of idling indefinitely.
            if self._candidate_object_id is not None and self._awaiting_candidate_check:
                self._attempted_object_ids.add(self._candidate_object_id)
                self._candidate_object_id = None
                self._target = None
                self._awaiting_candidate_check = False
                self._exploration_steps += 1
                return {"action": {"action": "turn_left", "angle_degrees": 30.0},
                        "done": False, "gateway_status": "false_candidate_rejected",
                        **status}
            self._awaiting_candidate_check = True
            return {"action": {"action": "no_op"}, "done": False,
                    "gateway_status": "candidate_reached_pending_evaluation",
                    "target": list(self._target)}
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


class DirectOvoPlanner(LiveOvoPlanner):
    """Feed protocol RGB-D/pose into HoloAgent OVO without ROS bindings."""

    @staticmethod
    def _array(value: Any) -> Any | None:
        if not isinstance(value, dict) or value.get("encoding") != "base64":
            return None
        import base64
        import numpy as np
        array = np.frombuffer(base64.b64decode(value["data"]), dtype=np.dtype(value["dtype"]))
        return array.reshape(value["shape"])

    def _ingest(self, payload: dict[str, Any]) -> None:
        import numpy as np
        observation = payload.get("observation") or {}
        rgb = self._array((observation.get("images") or {}).get("front"))
        depth = self._array(observation.get("depth"))
        pose = observation.get("pose") or {}
        position, quaternion = pose.get("position"), pose.get("orientation")
        if rgb is None or depth is None or not position or not quaternion:
            return
        x, y, z, w = (float(item) for item in quaternion)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = [[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                       [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                       [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]]
        c2w[:3, 3] = np.asarray(position[:3], dtype=np.float32)
        self.mapper.submit_frame(rgb[..., :3], depth, c2w)

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("type") == "step":
            self._ingest(payload)
        return super().plan(payload)


_PRIVATE_KEYS = frozenset({
    "ground_truth", "private", "expected_outcome", "reference_trajectory",
    "reference_pose", "target_references", "canonical_state", "mutation",
})


def assert_public_agent_request(value: Any, path: str = "request") -> None:
    """Reject accidental benchmark-private data before it reaches a model."""
    if isinstance(value, dict):
        leaked = _PRIVATE_KEYS.intersection(value)
        if leaked:
            raise ValueError(f"private benchmark field at {path}: {sorted(leaked)[0]}")
        for key, item in value.items():
            assert_public_agent_request(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_public_agent_request(item, f"{path}[{index}]")


class ManipulationStateMachine:
    """Add closed-loop manipulation guards around a general VLM policy."""

    def __init__(self, vlm: Any) -> None:
        self.vlm = vlm
        self.task_id: str | None = None
        self.recovery_turns = 0

    def reset(self):
        self.task_id = None
        self.recovery_turns = 0

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_id = str(payload.get('task_id', ''))
        if task_id != self.task_id:
            self.task_id, self.recovery_turns = task_id, 0
        previous = (payload.get('observation') or {}).get('action_feedback') or {}
        # Recover using public feedback; object selection and release are
        # decisions of the visual policy, using introduced names only.
        if previous.get('collision') or previous.get('reason') == 'blocked':
            self.recovery_turns += 1
            return {'action': {'action': 'turn_left' if self.recovery_turns % 2 else 'turn_right',
                               'angle_degrees': 45.0},
                    'done': False, 'gateway_status': 'manipulation_collision_recovery'}
        response = normalize_response(self.vlm.request(payload))
        response.setdefault('gateway_status', 'manipulation_vlm')
        return response


class CompositeHoloAgentPlanner:
    """Use HoloAgent spatial planning for navigation and a VLM for other skills."""

    def __init__(self, navigation: Any | None, vlm: VisionLanguageAdapter | None) -> None:
        self.navigation = navigation
        self.vlm = vlm
        self.manipulation = ManipulationStateMachine(vlm) if vlm is not None else None

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert_public_agent_request(payload)
        request_type = payload.get("type")
        if request_type == "session_start":
            updates: dict[str, Any] = {}
            sequence = payload.get('sequence_context', {})
            if sequence.get('reset_memory') and self.manipulation is not None:
                self.manipulation.reset()
            if self.navigation is not None:
                nav = self.navigation.plan(payload)
                if sequence.get('reset_memory') and nav.get('sequence_reset_ack') != sequence['run_id']:
                    raise RuntimeError('navigation backend did not acknowledge a fresh sequence')
                if isinstance(nav.get("memory_update"), dict):
                    updates.update(nav["memory_update"])
            if self.vlm is not None:
                response = self.vlm.request(payload)
                if sequence.get('reset_memory') and response.get('sequence_reset_ack') != sequence['run_id']:
                    raise RuntimeError('VLM backend did not acknowledge a fresh sequence')
                if isinstance(response.get("memory_update"), dict):
                    updates.update(response["memory_update"])
            return {'ok': True, 'memory_update': updates,
                    **({'sequence_reset_ack': sequence['run_id']} if sequence.get('reset_memory') else {})}
        if _navigation_request(payload) and self.navigation is not None:
            return self.navigation.plan(payload)
        if request_type == 'step' and 'manipulation' in payload.get('capabilities', []) \
                and self.manipulation is not None:
            return self.manipulation.plan(payload)
        if self.vlm is not None:
            response = self.vlm.request(payload)
            return normalize_response(response) if request_type == "step" else response
        return _unsupported('skill_model_not_configured')


class RemoteProtocolAdapter:
    def __init__(self, endpoint: str, timeout: float = 300.0) -> None:
        self.endpoint, self.timeout = endpoint, timeout

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _post_json(self.endpoint, payload, self.timeout)


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


def _start_direct_ovo(args: argparse.Namespace) -> tuple[DirectOvoPlanner, Any]:
    root = Path(args.mapping_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    from ovo.entities.semantic_mapping_online import SemanticMapping
    from ovo.utils import io_utils
    config_path = Path(args.mapping_config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = io_utils.load_config(str(config_path))
    config["device"] = args.mapping_device or "cuda:0"
    config["use_wandb"] = False
    config.setdefault("semantic", {}).setdefault("sam", {})["sam_ckpt_path"] = str(
        root.parents[1] / "VLN" / "vln_tmp" / "mapdsg" / "checkpoints" / "sam2")
    config.setdefault("slam", {})["slam_module"] = "vanilla"
    config.setdefault("data", {}).update(
        scene_name=args.mapping_scene, dataset_name="Benchmark", H=256, W=256)
    focal = 128.0 / math.tan(math.radians(100.0) / 2.0)
    config.setdefault("cam", {}).update(
        H=256, W=256, fx=focal, fy=focal, cx=127.5, cy=127.5,
        depth_scale=1.0, depth_th=10.0)
    output = Path(args.mapping_output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    mapper = SemanticMapping(config, str(output), scene=args.mapping_scene, direct=True)
    if mapper.initialization_error:
        raise RuntimeError(f"HoloAgent direct OVO initialization failed: {mapper.initialization_error}")
    mapper.start_direct()
    return DirectOvoPlanner(mapper, tolerance_m=args.position_tolerance_m,
                            max_segment_m=args.max_segment_m), mapper
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
    fsr_planner: Any | None = None

    def _planner_response(self, body: dict[str, Any]) -> dict[str, Any] | None:
        value = body.get("request") if isinstance(body.get("request"), dict) else body
        if not isinstance(value, dict):
            raise ValueError("benchmark request must be an object")
        assert_public_agent_request(value)
        if self.fsr_planner is not None:
            return self.fsr_planner.plan(value)
        if not self.planner_url:
            return None
        response = _post_json(self.planner_url, {
            "protocol_version": PROTOCOL_VERSION,
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
                import traceback
                traceback.print_exc()
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
    parser.add_argument("--vlm-endpoint", default=os.getenv("VLM_BASE_URL"),
                        help="OpenAI-compatible endpoint for manipulation, QA and memory")
    parser.add_argument("--vlm-model", default=os.getenv("VLM_MODEL"))
    parser.add_argument("--vlm-api-key", default=os.getenv("VLM_API_KEY"))
    parser.add_argument("--vlm-timeout", type=float, default=120.0)
    parser.add_argument("--vlm-planner-url",
                        help="protocol endpoint used for QA/manipulation/memory")
    parser.add_argument("--local-vlm-model",
                        help="local Hugging Face multimodal checkpoint; no API required")
    parser.add_argument("--local-vlm-device", default="auto")
    parser.add_argument("--local-vlm-max-new-tokens", type=int, default=256)
    parser.add_argument("--live-ovo", action="store_true",
                        help="run HoloAgent semantic_mapping_online.py in this process")
    parser.add_argument("--direct-ovo", action="store_true",
                        help="run HoloAgent OVO directly from protocol RGB-D")
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
    if bool(args.vlm_endpoint) != bool(args.vlm_model):
        parser.error("--vlm-endpoint and --vlm-model must be supplied together")
    if args.local_vlm_model and args.vlm_endpoint:
        parser.error("use only one of --local-vlm-model or --vlm-endpoint")
    if args.vlm_planner_url and (args.local_vlm_model or args.vlm_endpoint):
        parser.error("use only one VLM backend")
    if sum(bool(value) for value in (args.planner_url, args.fsr_config,
                                     args.live_ovo, args.direct_ovo)) > 1:
        parser.error("use only one navigation backend")
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
    if args.direct_ovo:
        if not args.mapping_root:
            parser.error("--direct-ovo requires --mapping-root")
        fsr_planner, live_mapper = _start_direct_ovo(args)
    if args.vlm_planner_url:
        vlm = RemoteProtocolAdapter(args.vlm_planner_url, args.vlm_timeout)
    elif args.local_vlm_model:
        vlm = LocalTransformersAdapter(
            args.local_vlm_model, args.local_vlm_device,
            args.local_vlm_max_new_tokens,
        )
    else:
        vlm = (VisionLanguageAdapter(args.vlm_endpoint, args.vlm_model,
                                     args.vlm_api_key, args.vlm_timeout)
               if args.vlm_endpoint else None)
    if fsr_planner is not None or vlm is not None:
        fsr_planner = CompositeHoloAgentPlanner(fsr_planner, vlm)
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
        print("planner: HoloAgent composite navigation/manipulation/QA", flush=True)
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
                if hasattr(live_mapper, "destroy_node"):
                    live_mapper.destroy_node()
            finally:
                if live_rclpy is not None and live_rclpy.ok():
                    live_rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
