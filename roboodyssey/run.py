"""Run a protocol agent over an ordered multi-session benchmark dataset."""

from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import logging
import math
import re
import sys
from pathlib import Path
from .artifacts.layout import sha256_file as _digest
import traceback
from .agents.agent_api import AgentProcess
from .agents.protocol import PROTOCOL_VERSION
from .scenes.dataset_loader import BenchmarkDataset
from .evaluation.report import write_summary
from .runtime.session_runner import SessionRunner
from .runtime.observations import OBSERVATION_POLICY
from .evaluation.task_metrics import METRIC_POLICY
from .scenes.holoagent_sensor_bridge import HoloAgentSensorTopics
from .artifacts.layout import initialize_run, update_status, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOLOAGENT_SENSOR_TOPICS = HoloAgentSensorTopics()


def _agent_name(command: str) -> str:
    """Derive a filesystem-safe short name from an agent command."""
    lowered = command.lower()
    if "simplenav" in lowered:
        name = "simplenav"
    elif "random_agent" in lowered:
        name = "random"
    else:
        name = Path(command.split()[0]).stem or "agent"
    return re.sub("[^a-zA-Z0-9_.-]+", "_", name)[:48]


def _create_run_directory(base: Path, command: str) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{timestamp}_{_agent_name(command)}"
    base.mkdir(parents=True, exist_ok=True)
    candidate = base / prefix
    suffix = 1
    while candidate.exists():
        candidate = base / f"{prefix}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return (candidate, timestamp)


def _dataset_digest(root: Path) -> str:
    """Hash dataset paths and contents so task/GT edits change the run identity."""
    value = hashlib.sha256()
    for path in sorted((item for item in root.rglob("*") if item.is_file())):
        value.update(path.relative_to(root).as_posix().encode("utf-8") + b"\x00")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
    return value.hexdigest()


def _inspect(dataset: BenchmarkDataset, scene_ids: list[str]) -> None:
    for scene_id in scene_ids:
        sessions = dataset.sessions(scene_id)
        if not sessions:
            raise RuntimeError(f"scene {scene_id} has no sessions")
        print(f"{scene_id}: {len(sessions)} sessions")
        for session in sessions:
            kinds = ",".join((str(task["task_type"]) for task in session.tasks))
            print(f"  {session.session_id}: [{kinds}]")


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--sequence",
        action="append",
        help="sequence_id in a scene collection; repeat for independent runs",
    )
    selection.add_argument(
        "--all-sequences",
        action="store_true",
        help="run every indexed sequence in fresh processes",
    )
    parser.add_argument(
        "--agent-cmd",
        default=f"python3 {ROOT / 'examples' / 'random_agent.py'}",
        help="command implementing the JSONL agent protocol",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results",
        help="base directory; each run is written to a timestamped child",
    )
    parser.add_argument(
        "--run-directory",
        type=Path,
        help="exact NEW run directory (used by the experiment manager)",
    )
    parser.add_argument(
        "--scene", action="append", help="scene id; repeat to select multiple scenes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="inspect tasks without starting Isaac Sim",
    )
    parser.add_argument(
        "--max-steps", type=int, default=2000, help="maximum actions per embodied task"
    )
    parser.add_argument(
        "--agent-timeout",
        type=float,
        default=120.0,
        help="seconds to wait for each agent response",
    )
    parser.add_argument(
        "--viewer", action="store_true", help="show Isaac Sim and the task overlay"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="Isaac/Vulkan device index; disables multi-GPU rendering",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="v5 private per-session CPU MP4 recording",
    )
    parser.add_argument(
        "--video-layout", choices=("front", "front_overview"), default="front_overview"
    )
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--video-event-seconds",
        type=float,
        default=0.5,
        help="playback hold per event; wall time is preserved separately in the index",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="CPU ffmpeg executable")
    parser.add_argument(
        "--step-delay", type=float, default=0.0, help="viewer delay after each action"
    )
    parser.add_argument(
        "--horizontal-fov-degrees",
        type=float,
        default=None,
        help="front camera horizontal FOV; defaults to 100, or the HoloAgent interface FOV when sensors are enabled",
    )
    parser.add_argument(
        "--holoagent-sensors",
        action="store_true",
        help="publish Isaac Sim RGB/depth/camera-info/pose to HoloAgent ROS 2 topics",
    )
    parser.add_argument(
        "--holoagent-rgb-topic", default=DEFAULT_HOLOAGENT_SENSOR_TOPICS.rgb
    )
    parser.add_argument(
        "--holoagent-depth-topic", default=DEFAULT_HOLOAGENT_SENSOR_TOPICS.depth
    )
    parser.add_argument(
        "--holoagent-camera-info-topic",
        default=DEFAULT_HOLOAGENT_SENSOR_TOPICS.camera_info,
    )
    parser.add_argument(
        "--holoagent-pose-topic", default=DEFAULT_HOLOAGENT_SENSOR_TOPICS.pose
    )
    parser.add_argument(
        "--holoagent-frame-id", default=DEFAULT_HOLOAGENT_SENSOR_TOPICS.frame_id
    )
    parser.add_argument(
        "--holoagent-pose-frame-id",
        default=DEFAULT_HOLOAGENT_SENSOR_TOPICS.pose_frame_id,
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help="enforce the frozen, full-dataset evaluation profile",
    )
    args = parser.parse_args(argv)
    if args.agent_timeout <= 0:
        parser.error("--agent-timeout must be positive")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.gpu is not None and args.gpu < 0:
        parser.error("--gpu must be non-negative")
    video_options = None
    if args.record_video:
        from .runtime.recording import validate_video_options

        try:
            video_options = validate_video_options(
                {
                    "layout": args.video_layout,
                    "fps": args.video_fps,
                    "event_seconds": args.video_event_seconds,
                    "ffmpeg": args.ffmpeg,
                }
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.official:
        if args.record_video:
            parser.error(
                "--record-video is a diagnostic profile; omit --official for review runs"
            )
        if args.scene:
            parser.error(
                "--official requires the complete dataset without selection filters"
            )
        if args.viewer or args.step_delay:
            parser.error("--official runs headless without artificial step delay")
        if args.max_steps != 2000 or args.agent_timeout != 120.0:
            parser.error("--official requires --max-steps 2000 and --agent-timeout 120")
        if args.horizontal_fov_degrees not in (None, 100.0):
            parser.error("--official requires a 100 degree horizontal FOV")
        if args.holoagent_sensors:
            parser.error("--official does not enable adapter-specific ROS sensors")
    if args.all_sequences or (args.sequence and len(args.sequence) > 1):
        if args.run_directory:
            parser.error(
                "--run-directory applies to one run; use --output for a sequence collection"
            )
        from .launch.sequences import selected_paths, run_collection

        paths = selected_paths(
            args.dataset, None if args.all_sequences else args.sequence
        )
        return run_collection(paths, args.output, raw_argv, dry_run=args.dry_run)
    dataset = BenchmarkDataset(
        args.dataset, sequence_id=args.sequence[0] if args.sequence else None
    )
    if args.official:
        from .validation.dataset import main as validate_main

        validation_code = validate_main([str(dataset.root)])
        if validation_code:
            return validation_code
    scenes = args.scene or dataset.scene_ids()
    unknown = sorted(set(scenes) - set(dataset.scene_ids()))
    if unknown:
        raise ValueError(f"unknown scenes: {unknown}")
    _inspect(dataset, scenes)
    if args.dry_run:
        return 0
    if args.run_directory:
        output_root = args.run_directory.resolve()
        output_root.mkdir(parents=True, exist_ok=False)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        output_root, run_timestamp = _create_run_directory(args.output, args.agent_cmd)
    initialize_run(output_root, dataset)
    write_json(
        output_root / "launch_config.json",
        {
            "schema_version": 1,
            "argv": raw_argv,
            "dataset_root": str(dataset.root),
            "sequence_identity": dataset.sequence_identity(),
            "python": sys.executable,
            "agent_command": args.agent_cmd,
        },
    )
    from .launch.provenance import capture_provenance

    capture_provenance(output_root / "provenance.json", dataset_root=dataset.root)
    from .scenes.isaac_backend import IsaacBackend

    try:
        backend = IsaacBackend(viewer=args.viewer, gpu=args.gpu)
    except BaseException as exc:
        update_status(
            output_root,
            phase="failed",
            execution="startup_failed",
            error=str(exc),
            cleanup="simulator_startup_failed",
        )
        raise
    import signal

    def interrupt_run(signum, frame):
        raise KeyboardInterrupt(f"benchmark interrupted by signal {signum}")

    signal.signal(signal.SIGINT, interrupt_run)
    signal.signal(signal.SIGTERM, interrupt_run)
    summary_written = False
    run_exit_code = 1
    try:
        logger = logging.getLogger("longterm_benchmark")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False
        handler = logging.FileHandler(output_root / "logs/run.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter("[benchmark] %(message)s"))
        logger.addHandler(console)
        logger.info(
            "benchmark_start dataset=%s scenes=%s agent=%s",
            dataset.root,
            scenes,
            args.agent_cmd,
        )
        from .scenes.isaac_env import EnvConfig

        holoagent_sensor_topics = None
        if args.holoagent_sensors:
            holoagent_sensor_topics = HoloAgentSensorTopics(
                rgb=args.holoagent_rgb_topic,
                depth=args.holoagent_depth_topic,
                camera_info=args.holoagent_camera_info_topic,
                pose=args.holoagent_pose_topic,
                frame_id=args.holoagent_frame_id,
                pose_frame_id=args.holoagent_pose_frame_id,
            )
        sensor_resolution = (
            (holoagent_sensor_topics.width, holoagent_sensor_topics.height)
            if holoagent_sensor_topics is not None
            else (256, 256)
        )
        horizontal_fov = (
            args.horizontal_fov_degrees
            if args.horizontal_fov_degrees is not None
            else holoagent_sensor_topics.horizontal_fov_degrees
            if holoagent_sensor_topics is not None
            else 100.0
        )
        camera_config = EnvConfig(
            horizontal_fov_degrees=horizontal_fov, resolution=sensor_resolution
        )
        width, height = camera_config.resolution
        focal_pixels = width / (
            2.0 * math.tan(math.radians(camera_config.horizontal_fov_degrees) / 2.0)
        )
        official_profile = {
            "dataset_manifest_sha256": _digest(dataset.root / "dataset_manifest.json"),
            "dataset_tree_sha256": _dataset_digest(dataset.root),
            "scene_ids": scenes,
            "max_steps_per_task": args.max_steps,
            "agent_timeout_seconds": args.agent_timeout,
            "camera_resolution": list(camera_config.resolution),
            "observation_policy": OBSERVATION_POLICY,
            "metric_policy": METRIC_POLICY,
            "horizontal_fov_degrees": camera_config.horizontal_fov_degrees,
            "protocol_version": PROTOCOL_VERSION,
            "sequence_identity": dataset.sequence_identity(),
        }
        official_profile_sha256 = hashlib.sha256(
            json.dumps(official_profile, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        run_config = {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "dataset_id": dataset.manifest.get("dataset_id"),
            "dataset_root": str(dataset.root),
            "runtime_scene_cache": str(dataset._requests.path("runtime_scene_cache")),
            "runtime_contract": dataset.manifest.get("runtime_contract"),
            "sequence_identity": dataset.sequence_identity(),
            "dataset_manifest_sha256": _digest(dataset.root / "dataset_manifest.json"),
            "dataset_tree_sha256": official_profile["dataset_tree_sha256"],
            "scene_ids": scenes,
            "agent_command": args.agent_cmd,
            "agent_timeout_seconds": args.agent_timeout,
            "max_steps_per_task": args.max_steps,
            "step_delay_seconds": args.step_delay,
            "viewer": args.viewer,
            "simulation_options": backend.options,
            "recording": video_options,
            "overview_viewer": "Isaac Sim 3D top-down viewport"
            if args.viewer
            else None,
            "observation_views": ["front"],
            "observation_policy": OBSERVATION_POLICY,
            "holoagent_sensors": None
            if holoagent_sensor_topics is None
            else holoagent_sensor_topics.as_dict(),
            "run_timestamp": run_timestamp,
            "run_directory": str(output_root),
            "agent_name": _agent_name(args.agent_cmd),
            "official": bool(args.official),
            "evaluation_profile": official_profile,
            "metric_policy": official_profile["metric_policy"],
            "evaluation_profile_sha256": official_profile_sha256,
            "camera": {
                "resolution": list(camera_config.resolution),
                "horizontal_fov_degrees": camera_config.horizontal_fov_degrees,
                "focal_length_mm": camera_config.focal_length_mm,
                "intrinsics": {
                    "fx": focal_pixels,
                    "fy": focal_pixels,
                    "cx": (width - 1) / 2.0,
                    "cy": (height - 1) / 2.0,
                },
            },
            "motion": {
                "robot_radius_m": camera_config.robot_radius,
                "default_move_step_m": camera_config.move_step,
                "default_turn_step_degrees": camera_config.turn_step_degrees,
                "max_move_distance_m": camera_config.max_move_distance,
                "max_turn_degrees": camera_config.max_turn_degrees,
                "collision_backoff_m": camera_config.collision_backoff,
                "door_open_angle_degrees": camera_config.door_open_angle_degrees,
            },
            "results_schema": {
                "run_log": "logs/run.log",
                "agent_io": "traces/agent_io.jsonl",
            },
        }
        (output_root / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        update_status(output_root, phase="running", execution="running")
        backend.configure_viewer()
        callbacks = backend.run_callbacks()
        with AgentProcess(
            args.agent_cmd,
            timeout=args.agent_timeout,
            stderr_callback=lambda line: logger.info("agent %s", line.rstrip()),
        ) as agent:
            runner = SessionRunner(
                output_root,
                agent,
                max_steps=args.max_steps,
                step_delay=args.step_delay,
                log_callback=lambda message: logger.info(message),
                video_options=video_options,
                **callbacks,
            )
            factory = backend.scene_factory(
                dataset,
                camera_config=camera_config,
                holoagent_sensor_topics=holoagent_sensor_topics,
                video_options=video_options,
            )
            for scene_id in scenes:
                all_sessions = dataset.sessions(scene_id)
                initial_state = dataset.canonical_state(scene_id)
                runner.run_scene(scene_id, all_sessions, initial_state, factory)
            summary = write_summary(dataset, output_root)
            summary_written = True
            if summary.get("evaluation_valid") is False:
                raise RuntimeError(
                    "completed run has invalid evaluation evidence; see summary.json"
                )
            update_status(
                output_root,
                phase="finalizing",
                execution="completed",
                summary="completed",
            )
        logger.info("benchmark_complete output=%s", output_root)
        run_exit_code = 0
    except BaseException as exc:
        update_status(output_root, phase="failed", execution="failed", error=str(exc))
        if "logger" in locals():
            logger.exception("benchmark_failed")
        traceback.print_exc()
        return 1
    finally:
        if not summary_written and "output_root" in locals():
            try:
                write_summary(dataset, output_root)
                update_status(output_root, summary="completed_after_failure")
            except Exception as exc:
                update_status(output_root, summary="failed", summary_error=str(exc))
                if "logger" in locals():
                    logger.exception("summary_write_failed")
        update_status(
            output_root,
            phase="completed" if run_exit_code == 0 else "failed",
            cleanup="simulator_close_requested",
            exit_code=run_exit_code,
        )
        backend.close(run_exit_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
