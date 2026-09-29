"""Run a protocol agent over an ordered multi-session benchmark dataset."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
import math
import re
import subprocess
import sys
from pathlib import Path
import traceback

from .agents.agent_api import AgentProcess
from .agents.protocol import PROTOCOL_VERSION
from .scenes.dataset_loader import BenchmarkDataset
from .scenes.state_composer import compose_session_state
from .evaluation.report import write_summary
from .runtime.session_runner import SessionRunner
from .scenes.holoagent_sensor_bridge import HoloAgentSensorTopics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset" / "scene000430_after_changes"
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
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)[:48]


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
    return candidate, timestamp


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _dataset_digest(root: Path) -> str:
    """Hash dataset paths and contents so task/GT edits change the run identity."""
    value = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        value.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
    return value.hexdigest()


def _bind_viewport_camera(viewport, camera_path: str) -> None:
    """Bind a Kit viewport to a USD camera using the Isaac Sim 6 API."""
    if viewport is None:
        return
    # ``camera_path`` is the public API in Isaac Sim 6.  Older Kit builds
    # exposed ``set_active_camera`` on the same object, so retain that small
    # fallback for local installations.
    try:
        # ViewportManager updates both the viewport API and its render product.
        # This is important when the viewport was created after the camera
        # annotator: assigning ``camera_path`` alone can leave a stale product.
        from isaacsim.core.rendering_manager import ViewportManager
        ViewportManager.set_camera(camera_path, render_product_or_viewport=viewport)
        # Some Kit 6.x builds keep a separate render-product relationship.  Set
        # it explicitly as well, otherwise the visible viewport may continue to
        # render the camera that was selected when the product was created.
        try:
            from pxr import Sdf, Usd
            product = ViewportManager.get_render_product(viewport)
            if product is not None:
                stage = __import__("omni.usd", fromlist=["get_context"]).get_context().get_stage()
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    product.GetCameraRel().SetTargets([Sdf.Path(camera_path)])
        except Exception:
            pass
    except Exception:
        try:
            viewport.camera_path = camera_path
        except Exception:
            setter = getattr(viewport, "set_active_camera", None)
            if setter is None:
                raise
            setter(camera_path)
    try:
        viewport.updates_enabled = True
    except Exception:
        pass
    # ViewportWindow invokes this after a scene/frame rebuild.  Calling it
    # explicitly is required by some Kit versions when a camera prim is moved
    # programmatically rather than by the viewport manipulator.
    try:
        import omni.usd
        from pxr import Sdf
        changed = getattr(viewport, "viewport_changed", None)
        stage = getattr(viewport, "stage", None) or omni.usd.get_context().get_stage()
        if changed is not None and stage is not None:
            changed(Sdf.Path(camera_path), stage)
    except Exception:
        pass


def _place_viewport_window(window) -> None:
    """Place the overview window in the top-right without assuming 1920px UI."""
    if window is None:
        return
    try:
        import omni.ui as ui
        width = int(ui.Workspace.get_main_window_width())
        window.position_x = max(0, width - 440)
        window.position_y = 42
    except Exception:
        # A fixed fallback is still preferable to leaving the window off-screen.
        try:
            window.position_x = 840
            window.position_y = 42
        except Exception:
            pass
    window.visible = True
    try:
        window.viewport_api.updates_enabled = True
    except Exception:
        pass


def _sync_viewer(viewport, overview_window, front_camera: str, overview_camera: str | None,
                 app) -> None:
    """Rebind and advance viewer products after an environment action.

    Isaac Sim can recreate a viewport render product when a stage is opened or
    a window is moved.  Reasserting both camera relationships after each step
    keeps the displayed cameras tied to the same USD prims used by the agent.
    """
    try:
        _bind_viewport_camera(viewport, front_camera)
        if overview_camera and overview_window is not None:
            overview_viewport = getattr(overview_window, "viewport_api", overview_window)
            _bind_viewport_camera(overview_viewport, overview_camera)
            _place_viewport_window(overview_window)
        app.update()
    except Exception as exc:
        # Rendering must not abort an otherwise valid benchmark run.
        print(f"[viewer] warning: viewport sync failed: {exc}")


def _make_env_factory(
    dataset: BenchmarkDataset, app, viewer: bool = False, overview_holder: dict | None = None,
    camera_config=None, holoagent_sensor_topics=None,
):
    import numpy as np
    from .scenes.isaac_env import EnvConfig, IsaacVLNEnv

    def factory(scene_id: str, initial_state: dict, session_id: str):
        with np.load(dataset.navigation_path(scene_id), allow_pickle=False) as archive:
            navigation = {key: archive[key].copy() for key in archive.files}
        env = IsaacVLNEnv(
            dataset.scene_asset(scene_id),
            camera_config or EnvConfig(camera_prim_path="/World/LongTermBenchmark/Camera"),
            holoagent_sensor_topics=holoagent_sensor_topics,
        )
        env.configure_scene(dataset.scene_metadata(scene_id), navigation,
                            dataset.manipulation(scene_id), initial_state)
        if viewer:
            overview_path = env.setup_overview()
        else:
            overview_path = None
        first = next((item for item in dataset.sessions(scene_id) if item.session_id == session_id), None)
        instruction = first.metadata.get("instruction") if first else None
        start_pose = first.metadata.get("start_pose") if first else None
        if start_pose is None and isinstance(instruction, dict):
            start_pose = instruction.get("start_pose")
        if start_pose:
            env.set_agent_pose(start_pose)
        try:
            from pxr import Gf, UsdGeom, UsdLux
            dome = UsdLux.DomeLight.Define(env.stage, "/World/BenchmarkRuntime/DomeLight")
            dome.CreateIntensityAttr(3000.0)
            fill = UsdLux.DistantLight.Define(env.stage, "/World/BenchmarkRuntime/FillLight")
            fill.CreateIntensityAttr(800.0)
            UsdGeom.XformCommonAPI(fill).SetRotate(Gf.Vec3f(25.0, -20.0, -35.0))
        except Exception as exc:
            print(f"warning: runtime lighting unavailable: {exc}")
        if viewer:
            from omni.kit.viewport.utility import get_active_viewport, get_viewport_from_window_name
            # Do not use only the focused viewport: creating/focusing the
            # Overview window can make it the active viewport.  The benchmark
            # main view must remain bound to the default Isaac Sim window.
            viewport = get_viewport_from_window_name() or get_active_viewport()
            if viewport is None:
                raise RuntimeError("Isaac Sim main viewport is unavailable")
            if overview_holder is not None:
                overview_holder["front_viewport"] = viewport
                overview_holder["overview_camera_path"] = overview_path
            _bind_viewport_camera(viewport, env.config.camera_prim_path)
            if overview_path:
                try:
                    try:
                        from isaacsim.core.rendering_manager import ViewportManager
                        overview = ViewportManager.create_viewport_window(
                            camera=overview_path, title="Overview", resolution=(420, 300))
                    except (ImportError, AttributeError):
                        from omni.kit.viewport.window import ViewportWindow
                        overview = ViewportWindow(name="Overview", width=420, height=300)
                    _bind_viewport_camera(
                        getattr(overview, "viewport_api", overview), overview_path)
                    if overview_holder is not None:
                        previous = overview_holder.get("window")
                        if previous is not None:
                            try:
                                previous.destroy()
                            except Exception:
                                pass
                        overview_holder["window"] = overview
                    env.attach_overview_window(overview)
                    _place_viewport_window(overview)
                    print(
                        f"[viewer] main_viewport_camera={getattr(viewport, 'camera_path', None)} "
                        f"front_camera={env.config.camera_prim_path} overview={overview_path}"
                    )
                except Exception as exc:
                    print(f"[viewer] warning: overview viewport unavailable: {exc}")
        for _ in range(120):
            app.update()
            if env.try_observe() is not None:
                break
            # Some Kit versions do not advance a newly-created render product
            # from app.update() alone.  Force Replicator only during warm-up;
            # doing this on every action breaks viewer backbuffers.
            env.render_once()
        if env.try_observe() is None:
            env.close()
            raise RuntimeError(
                "Isaac Sim camera produced no RGB frame after initialization; "
                "check viewer/GPU rendering and camera annotator setup"
            )
        if viewer:
            _bind_viewport_camera(viewport, env.config.camera_prim_path)
        if viewer and overview_holder.get("window") is not None:
            _place_viewport_window(overview_holder["window"])
        return env

    return factory


def _inspect(dataset: BenchmarkDataset, scene_ids: list[str]) -> None:
    for scene_id in scene_ids:
        sessions = dataset.sessions(scene_id)
        if not sessions:
            raise RuntimeError(f"scene {scene_id} has no sessions")
        print(f"{scene_id}: {len(sessions)} sessions")
        for session in sessions:
            kinds = ",".join(str(task["task_type"]) for task in session.tasks)
            print(f"  {session.session_id}: [{kinds}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--agent-cmd", default=f"python3 {ROOT / 'examples' / 'random_agent.py'}",
                        help="command implementing the JSONL agent protocol")
    parser.add_argument("--output", type=Path, default=ROOT / "results",
                        help="base directory; each run is written to a timestamped child")
    parser.add_argument("--scene", action="append", help="scene id; repeat to select multiple scenes")
    parser.add_argument("--session", action="append",
                        help="diagnostic session id; repeat to select multiple sessions")
    parser.add_argument("--task", action="append",
                        help="diagnostic task id; repeat to select multiple tasks")
    parser.add_argument("--task-type", action="append", choices=("navigation", "manipulation", "qa"),
                        help="diagnostic task type; repeat to select multiple types")
    parser.add_argument("--dry-run", action="store_true", help="inspect tasks without starting Isaac Sim")
    parser.add_argument("--max-steps", type=int, default=2000, help="maximum actions per embodied task")
    parser.add_argument("--agent-timeout", type=float, default=120.0,
                        help="seconds to wait for each agent response")
    parser.add_argument("--qa-judge-model", help="optional local model for open-ended QA judging")
    parser.add_argument("--viewer", action="store_true", help="show Isaac Sim and the task overlay")
    parser.add_argument("--step-delay", type=float, default=0.0, help="viewer delay after each action")
    parser.add_argument("--session-start-pose-mode", choices=("inherit", "explicit"), default="inherit",
                        help="agent pose at session boundaries (default: inherit previous final pose)")
    parser.add_argument("--horizontal-fov-degrees", type=float, default=None,
                        help="front camera horizontal FOV; defaults to 100, or the HoloAgent interface FOV when sensors are enabled")
    parser.add_argument("--holoagent-sensors", action="store_true",
                        help="publish Isaac Sim RGB/depth/camera-info/pose to HoloAgent ROS 2 topics")
    parser.add_argument("--holoagent-rgb-topic", default=DEFAULT_HOLOAGENT_SENSOR_TOPICS.rgb)
    parser.add_argument("--holoagent-depth-topic", default=DEFAULT_HOLOAGENT_SENSOR_TOPICS.depth)
    parser.add_argument("--holoagent-camera-info-topic",
                        default=DEFAULT_HOLOAGENT_SENSOR_TOPICS.camera_info)
    parser.add_argument("--holoagent-pose-topic", default=DEFAULT_HOLOAGENT_SENSOR_TOPICS.pose)
    parser.add_argument("--holoagent-frame-id", default=DEFAULT_HOLOAGENT_SENSOR_TOPICS.frame_id)
    parser.add_argument("--holoagent-pose-frame-id", default=DEFAULT_HOLOAGENT_SENSOR_TOPICS.pose_frame_id)
    parser.add_argument("--official", action="store_true",
                        help="enforce the frozen, full-dataset evaluation profile")
    args = parser.parse_args(argv)
    if args.agent_timeout <= 0:
        parser.error("--agent-timeout must be positive")
    if args.official:
        if args.scene or args.session or args.task or args.task_type:
            parser.error("--official requires the complete dataset without selection filters")
        if args.viewer or args.step_delay:
            parser.error("--official runs headless without artificial step delay")
        if args.max_steps != 2000 or args.agent_timeout != 120.0:
            parser.error("--official requires --max-steps 2000 and --agent-timeout 120")
        if args.horizontal_fov_degrees not in (None, 100.0):
            parser.error("--official requires a 100 degree horizontal FOV")
        if args.holoagent_sensors:
            parser.error("--official does not enable adapter-specific ROS sensors")

    dataset = BenchmarkDataset(args.dataset)
    if args.official:
        subprocess.run([
            sys.executable, str(ROOT / "scripts" / "validate_dataset.py"), str(dataset.root)
        ], check=True)
    scenes = args.scene or dataset.scene_ids()
    unknown = sorted(set(scenes) - set(dataset.scene_ids()))
    if unknown:
        raise ValueError(f"unknown scenes: {unknown}")
    _inspect(dataset, scenes)
    if args.dry_run:
        return 0

    output_root, run_timestamp = _create_run_directory(args.output, args.agent_cmd)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not args.viewer})
    overlay = None
    summary_written = False
    overview_holder = {}
    try:
        logger = logging.getLogger("longterm_benchmark")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False
        handler = logging.FileHandler(output_root / "run.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter("[benchmark] %(message)s"))
        logger.addHandler(console)
        logger.info("benchmark_start dataset=%s scenes=%s agent=%s", dataset.root, scenes, args.agent_cmd)
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
        sensor_resolution = ((holoagent_sensor_topics.width, holoagent_sensor_topics.height)
                             if holoagent_sensor_topics is not None else (256, 256))
        horizontal_fov = (args.horizontal_fov_degrees if args.horizontal_fov_degrees is not None
                          else (holoagent_sensor_topics.horizontal_fov_degrees
                                if holoagent_sensor_topics is not None else 100.0))
        camera_config = EnvConfig(horizontal_fov_degrees=horizontal_fov,
                                  resolution=sensor_resolution)
        width, height = camera_config.resolution
        focal_pixels = width / (2.0 * math.tan(
            math.radians(camera_config.horizontal_fov_degrees) / 2.0))
        official_profile = {
            "dataset_manifest_sha256": _digest(dataset.root / "dataset_manifest.json"),
            "dataset_tree_sha256": _dataset_digest(dataset.root),
            "scene_ids": scenes,
            "session_filter": args.session,
            "task_filter": args.task,
            "task_type_filter": args.task_type,
            "max_steps_per_task": args.max_steps,
            "agent_timeout_seconds": args.agent_timeout,
            "session_start_pose_mode": args.session_start_pose_mode,
            "camera_resolution": list(camera_config.resolution),
            "horizontal_fov_degrees": camera_config.horizontal_fov_degrees,
            "protocol_version": PROTOCOL_VERSION,
        }
        official_profile_sha256 = hashlib.sha256(json.dumps(
            official_profile, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        run_config = {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "dataset_id": dataset.manifest.get("dataset_id"),
            "dataset_root": str(dataset.root),
            "dataset_manifest_sha256": _digest(dataset.root / "dataset_manifest.json"),
            "dataset_tree_sha256": official_profile["dataset_tree_sha256"],
            "scene_ids": scenes,
            "agent_command": args.agent_cmd,
            "agent_timeout_seconds": args.agent_timeout,
            "max_steps_per_task": args.max_steps,
            "step_delay_seconds": args.step_delay,
            "session_start_pose_mode": args.session_start_pose_mode,
            "viewer": args.viewer,
            "overview_viewer": "Isaac Sim 3D top-down viewport" if args.viewer else None,
            "observation_views": ["front"],
            "holoagent_sensors": (None if holoagent_sensor_topics is None
                                   else holoagent_sensor_topics.as_dict()),
            "run_timestamp": run_timestamp,
            "run_directory": str(output_root),
            "agent_name": _agent_name(args.agent_cmd),
            "official": bool(args.official),
            "evaluation_profile": official_profile,
            "evaluation_profile_sha256": official_profile_sha256,
            "camera": {"resolution": list(camera_config.resolution),
                        "horizontal_fov_degrees": camera_config.horizontal_fov_degrees,
                        "focal_length_mm": camera_config.focal_length_mm,
                        "intrinsics": {"fx": focal_pixels, "fy": focal_pixels,
                                        "cx": (width - 1) / 2.0, "cy": (height - 1) / 2.0}},
            "motion": {"robot_radius_m": camera_config.robot_radius,
                        "default_move_step_m": camera_config.move_step,
                        "default_turn_step_degrees": camera_config.turn_step_degrees,
                        "max_move_distance_m": camera_config.max_move_distance,
                        "max_turn_degrees": camera_config.max_turn_degrees,
                        "collision_backoff_m": camera_config.collision_backoff,
                        "door_open_angle_degrees": camera_config.door_open_angle_degrees},
            "results_schema": {"run_log": "run.log", "agent_io": "agent_io.jsonl",
                               "session_meta": "session_meta.json"},
        }
        (output_root / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.viewer:
            from .viewer.viewer_overlay import ViewerOverlay
            overlay = ViewerOverlay()
        def update_overlay(**values):
            if overlay is None:
                return
            overlay.update(**values)
            # A single Kit update after each state/UI event keeps the commander
            # synchronized with the rendered room-memory frame without the
            # repeated-update loop that can invalidate viewer backbuffers.
            app.update()
        judge = None
        if args.qa_judge_model:
            from .evaluation.judges.local_llm import LocalLLMJudge
            judge = LocalLLMJudge(args.qa_judge_model)
        import omni.kit.app
        update = omni.kit.app.get_app().update
        with AgentProcess(args.agent_cmd, timeout=args.agent_timeout,
                          stderr_callback=lambda line: logger.info("agent %s", line.rstrip())) as agent:
            runner = SessionRunner(
                output_root, agent, max_steps=args.max_steps, qa_judge=judge,
                step_delay=args.step_delay, wait_callback=update,
                viewer_callback=update_overlay if overlay else None,
                transition_callback=(lambda: overlay.transition(1.0, omni.kit.app.get_app())) if overlay else None,
                reset_requested_callback=(lambda: overlay.consume_reset()) if overlay else None,
                viewer_sync_callback=(
                    lambda _env: _sync_viewer(
                        overview_holder.get("front_viewport"),
                        overview_holder.get("window"),
                        _env.config.camera_prim_path,
                        overview_holder.get("overview_camera_path"),
                        app,
                    )
                ) if args.viewer else None,
                session_start_pose_mode=args.session_start_pose_mode,
                task_ids=set(args.task) if args.task else None,
                task_types=set(args.task_type) if args.task_type else None,
                log_callback=lambda message: logger.info(message),
            )
            factory = _make_env_factory(dataset, app, viewer=args.viewer,
                                        overview_holder=overview_holder,
                                        camera_config=camera_config,
                                        holoagent_sensor_topics=holoagent_sensor_topics)
            for scene_id in scenes:
                all_sessions = dataset.sessions(scene_id)
                selected = ([item for item in all_sessions if item.session_id in set(args.session)]
                            if args.session else all_sessions)
                if args.session and not selected:
                    raise ValueError(f"none of the requested sessions exist in {scene_id}: {args.session}")
                initial_state = dataset.canonical_state(scene_id)
                if selected:
                    first_index = all_sessions.index(selected[0])
                    for prior in all_sessions[:first_index]:
                        initial_state = compose_session_state(initial_state, prior.mutation())
                runner.run_scene(scene_id, selected, initial_state, factory)
            write_summary(dataset, output_root)
            summary_written = True
        logger.info("benchmark_complete output=%s", output_root)
    except BaseException:
        if 'logger' in locals():
            logger.exception("benchmark_failed")
        traceback.print_exc()
        return 1
    finally:
        if not summary_written and 'output_root' in locals():
            try:
                write_summary(dataset, output_root)
            except Exception:
                if 'logger' in locals():
                    logger.exception("summary_write_failed")
        if overlay is not None:
            overlay.close()
        overview = overview_holder.get("window")
        if overview is not None:
            try:
                overview.destroy()
            except Exception:
                pass
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
