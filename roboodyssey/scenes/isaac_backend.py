"""Isaac Sim application, scene setup and viewer integration for benchmark runs."""
from __future__ import annotations

from contextlib import suppress
from copy import deepcopy
from functools import partial
from inspect import signature

from .dataset_loader import BenchmarkDataset
from ..viewer.viewport import _bind_viewport_camera, _place_viewport_window, _sync_viewer


def make_env_factory(
    dataset: BenchmarkDataset, app, viewer: bool = False, overview_holder: dict | None = None,
    camera_config=None, holoagent_sensor_topics=None, video_options=None,
):
    import numpy as np
    from .isaac_env import EnvConfig, IsaacVLNEnv

    holder = {} if overview_holder is None else overview_holder

    def factory(scene_id: str, initial_state: dict, session_id: str):
        with np.load(dataset.navigation_path(scene_id), allow_pickle=False) as archive:
            navigation = {key: archive[key].copy() for key in archive.files}
        env = IsaacVLNEnv(
            dataset.scene_asset(scene_id),
            camera_config or EnvConfig(camera_prim_path="/World/LongTermBenchmark/Camera"),
            holoagent_sensor_topics=holoagent_sensor_topics,
        )
        try:
            env.configure_scene(dataset.scene_metadata(scene_id), navigation,
                                dataset.manipulation(scene_id), initial_state)
            if viewer or (video_options and video_options['layout'] == 'front_overview'):
                overview_path = env.setup_overview()
            else:
                overview_path = None
            if video_options:
                env.setup_review_capture(video_options['layout'])
            start_pose = deepcopy(initial_state["robot_base_pose"])
            start_pose["position"][2] += env.config.camera_height
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
                holder["front_viewport"] = viewport
                holder["overview_camera_path"] = overview_path
                _bind_viewport_camera(viewport, env.config.camera_prim_path)
                # Recording already owns an overview render product. Keep one GUI
                # viewport to avoid duplicate overview products during warm-up.
                if overview_path and not video_options:
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
                        previous = holder.get("window")
                        if previous is not None:
                            with suppress(Exception):
                                previous.destroy()
                        holder["window"] = overview
                        env.attach_overview_window(overview)
                        _place_viewport_window(overview)
                        print(
                            f"[viewer] main_viewport_camera={getattr(viewport, 'camera_path', None)} "
                            f"front_camera={env.config.camera_prim_path} overview={overview_path}"
                        )
                    except Exception as exc:
                        print(f"[viewer] warning: overview viewport unavailable: {exc}")
            ready = False
            capture_error = None
            for _ in range(120):
                app.update()
                ready = env.try_observe() is not None
                if ready and video_options:
                    try:
                        env.capture_review_frames(video_options['layout'])
                    except RuntimeError as exc:
                        ready, capture_error = False, exc
                if ready:
                    break
                # Some Kit versions do not advance a newly-created render product
                # from app.update() alone.  Force Replicator only during warm-up;
                # doing this on every action breaks viewer backbuffers.
                env.render_once()
            if not ready:
                raise RuntimeError(
                    "Isaac Sim requested cameras produced no RGB frame after initialization; "
                    "check viewer/GPU rendering and camera annotator setup"
                ) from capture_error
            if viewer:
                _bind_viewport_camera(viewport, env.config.camera_prim_path)
            if viewer and holder.get("window") is not None:
                _place_viewport_window(holder["window"])
        except BaseException:
            with suppress(Exception):
                env.close()
            raise
        return env

    return factory


class IsaacBackend:
    """Own Kit and its optional viewer for one evaluation run."""

    def __init__(self, *, viewer: bool, gpu: int | None):
        from isaacsim import SimulationApp

        self.options = {"headless": not viewer}
        if gpu is not None:
            self.options.update(active_gpu=gpu, physics_gpu=gpu, multi_gpu=False)
        self.app = SimulationApp(self.options)
        self.viewer = viewer
        self.overview = {}
        self.overlay = None

    def configure_viewer(self) -> None:
        if self.viewer:
            from ..viewer.viewer_overlay import ViewerOverlay

            self.overlay = ViewerOverlay()

    def run_callbacks(self) -> dict:
        import omni.kit.app

        kit_app = omni.kit.app.get_app()
        callbacks = {"wait_callback": kit_app.update}
        if self.overlay is not None:
            callbacks.update(
                viewer_callback=self.update_overlay,
                transition_callback=partial(self.overlay.transition, 1.0, kit_app),
                reset_requested_callback=self.overlay.consume_reset,
                viewer_sync_callback=self.sync_viewer,
            )
        return callbacks

    def update_overlay(self, **values) -> None:
        self.overlay.update(**values)
        # Keep the UI in sync without repeatedly updating Kit's backbuffers.
        self.app.update()

    def sync_viewer(self, env) -> None:
        _sync_viewer(
            self.overview.get("front_viewport"),
            self.overview.get("window"),
            env.config.camera_prim_path,
            self.overview.get("overview_camera_path"),
            self.app,
        )

    def scene_factory(self, dataset, *, camera_config, holoagent_sensor_topics, video_options):
        return make_env_factory(
            dataset, self.app, viewer=self.viewer, overview_holder=self.overview,
            camera_config=camera_config, holoagent_sensor_topics=holoagent_sensor_topics,
            video_options=video_options,
        )

    def close(self, exit_code: int) -> None:
        try:
            if self.overlay is not None:
                self.overlay.close()
        finally:
            overview = self.overview.get("window")
            if overview is not None:
                with suppress(Exception):
                    overview.destroy()
            # Some Kit builds exit the interpreter during close.
            if "exit_code" in signature(self.app.close).parameters:
                self.app.close(exit_code=exit_code)
            else:
                self.app.close()
