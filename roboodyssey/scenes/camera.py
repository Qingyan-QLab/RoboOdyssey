"""Lazy Isaac scene/camera primitives shared by preparation and execution."""
import math
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import time

from ..artifacts.layout import json_digest, sha256_file
from .lighting import apply_scene_lighting


def open_scene_stage(scene_asset, *, settle_frames=12, max_updates=6000):
    """Open a stage in an already running SimulationApp, with bounded loading."""
    path = Path(scene_asset).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    import omni.usd
    import omni.kit.app
    app = omni.kit.app.get_app()
    context = omni.usd.get_context()
    context.open_stage(str(path))
    stable = 0
    for _ in range(max_updates):
        app.update()
        stable = stable+1 if context.get_stage_loading_status()[2] <= 0 else 0
        if stable >= settle_frames:
            break
    else:
        raise TimeoutError(f'timed out loading Isaac Sim stage: {path}')
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError(f'Isaac Sim could not open {path}')
    return stage, app


def create_camera(prim_path, resolution):
    from isaacsim.sensors.camera import Camera
    camera = Camera(prim_path=prim_path, resolution=tuple(resolution), frequency=-1)
    camera.initialize(attach_rgb_annotator=True)
    return camera


class SceneCamera:
    """Persistent preparation camera; view selection remains the caller's job.

    Construct only after SimulationApp. close() releases this camera's render
    product; it does not close the caller-owned application or shared stage.
    """
    def __init__(self, scene_asset, environment_setup=None, *, resolution=(640, 480),
                 horizontal_fov_degrees=100, warmup_frames=8,
                 camera_prim_path='/World/RoboOdyssey/PreparationCamera', lighting=None):
        if len(resolution) != 2 or any(type(v) is not int or v <= 0 for v in resolution):
            raise ValueError('resolution must contain positive integer width and height')
        if not 1 < horizontal_fov_degrees < 179 or type(warmup_frames) is not int or warmup_frames < 1:
            raise ValueError('invalid field of view or warmup frame count')
        self._asset_sha256 = sha256_file(Path(scene_asset).resolve())
        self._setup_sha256 = json_digest(environment_setup, allow_nan=False)
        self._frame_id = 0
        self._last_capture_metadata = None
        self.stage, self._app = open_scene_stage(scene_asset)
        from ..contracts.doors import apply_open_doors, verify_open_doors
        self.environment_state = apply_open_doors(self.stage, environment_setup)
        if environment_setup:
            self.environment_state['geometry_check'] = verify_open_doors(self.stage, environment_setup)
        self.lighting_state = apply_scene_lighting(self.stage, lighting)
        self.resolution, self.warmup_frames = tuple(resolution), warmup_frames
        self.camera = create_camera(camera_prim_path, resolution)
        self.closed = False
        focal_length = 24.0
        aperture = 2*focal_length*math.tan(math.radians(horizontal_fov_degrees)/2)
        try:
            self.camera.set_focal_length(focal_length)
            self.camera.set_horizontal_aperture(aperture, maintain_square_pixels=True)
            if hasattr(self.camera, 'set_vertical_aperture'):
                self.camera.set_vertical_aperture(aperture*resolution[1]/resolution[0])
            self.camera.set_clipping_range(near_distance=.05, far_distance=1000.0)
        except (AttributeError, TypeError):
            from pxr import UsdGeom
            prim = UsdGeom.Camera.Get(self.stage, camera_prim_path)
            prim.GetFocalLengthAttr().Set(focal_length)
            prim.GetHorizontalApertureAttr().Set(aperture)
            prim.GetVerticalApertureAttr().Set(aperture*resolution[1]/resolution[0])
            prim.GetClippingRangeAttr().Set((.05, 1000.0))

    @property
    def last_capture_metadata(self):
        """Prepared-input identity and sampled pose, never execution history.

        The state stamp covers the asset and verified preparation configuration;
        it does not attest to arbitrary external USD edits after construction.
        """
        return deepcopy(self._last_capture_metadata)

    def capture(self, position, orientation_xyzw):
        if self.closed:
            raise RuntimeError('camera is closed')
        if len(position) != 3 or len(orientation_xyzw) != 4 or not all(
                math.isfinite(float(v)) for v in (*position, *orientation_xyzw)):
            raise ValueError('camera pose requires finite XYZ and XYZW')
        if not math.isclose(sum(float(v)**2 for v in orientation_xyzw), 1., rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError('camera orientation must be a unit XYZW quaternion')
        started = time.monotonic_ns()
        x, y, z, w = map(float, orientation_xyzw)
        self.camera.set_world_pose(position=list(map(float, position)), orientation=[w, x, y, z], camera_axes='world')
        try:
            import omni.replicator.core as rep
            rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=True, wait_for_render=True)
        except Exception:
            pass  # SDK versions without this orchestrator use the update path.
        for _ in range(self.warmup_frames):
            self._app.update()
        frame = self.camera.get_rgb()
        if frame is None:
            raise RuntimeError('Isaac Sim camera produced no RGB frame')
        frame = frame.copy()
        if frame.shape != (self.resolution[1], self.resolution[0], 3):
            raise RuntimeError('Isaac Sim camera returned an unexpected RGB shape')
        if not frame.any():
            raise RuntimeError('preparation camera produced an all-black frame; check lighting, view and render readiness')
        self._frame_id += 1
        state_identity = {'kind': 'prepared_asset_and_environment',
            'scene_asset_sha256': self._asset_sha256, 'environment_setup_sha256': self._setup_sha256,
            'environment_state_sha256': json_digest(self.environment_state, allow_nan=False),
            'lighting_sha256': json_digest(self.lighting_state, allow_nan=False)}
        self._last_capture_metadata = {'schema_version': 1, 'state_source': 'preparation',
            'frame_id': self._frame_id, 'captured_at_utc': datetime.now(timezone.utc).isoformat(),
            'sample_started_monotonic_ns': started, 'captured_monotonic_ns': time.monotonic_ns(),
            'camera_pose': {'coordinate_frame': 'world', 'position': list(map(float, position)),
                            'orientation_xyzw': list(map(float, orientation_xyzw))},
            'resolution': list(self.resolution), 'image_sha256': hashlib.sha256(frame.tobytes()).hexdigest(),
            'image_dtype': str(frame.dtype), 'image_shape': list(frame.shape),
            'lighting': deepcopy(self.lighting_state),
            'image_statistics': {'min': float(frame.min()), 'max': float(frame.max()),
                                 'mean': float(frame.mean()), 'std': float(frame.std())},
            'state_identity': state_identity,
            'state_version': json_digest({'source': 'preparation', 'identity': state_identity}, allow_nan=False)}
        return frame

    def close(self):
        if self.closed:
            return
        self.closed = True
        cleanup = getattr(self.camera, 'destroy', None)
        if callable(cleanup):
            cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
