"""Self-contained Isaac Sim environment for the scene-package benchmark.

The benchmark intentionally controls a high-level avatar rather than a robot
arm.  Pick/place are logical interactions backed by USD transforms: a picked
object is attached to the camera's lower-right view space until it is placed.
All Isaac/pxr imports are delayed until the environment is constructed so
dataset validation and dry-runs remain usable on CPU-only machines.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnvConfig:
    camera_prim_path: str = "/World/LongTermBenchmark/Camera"
    resolution: tuple[int, int] = (256, 256)
    horizontal_fov_degrees: float = 90.0
    focal_length_mm: float = 24.0
    camera_height: float = 1.5
    step_seconds: float = 1.0 / 30.0
    move_step: float = 0.25
    turn_step_degrees: float = 15.0
    robot_radius: float = 0.1
    max_move_distance: float = 10.0
    max_turn_degrees: float = 720.0
    collision_backoff: float = 0.1
    collision_sample_step: float = 0.05
    door_open_angle_degrees: float = 90.0


def _quat_to_yaw(quaternion: list[float]) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_quaternion(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def _camera_quaternion(yaw: float, pitch: float) -> list[float]:
    """Return scalar-first WXYZ for yaw around Z then pitch around local Y."""
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    return [cy * cp, -sy * sp, cy * sp, sy * cp]


class IsaacVLNEnv:
    """A deterministic high-level avatar environment backed by Isaac Sim."""

    def __init__(self, scene_asset: str | Path, config: EnvConfig | None = None,
                 holoagent_sensor_topics: Any | None = None):
        import numpy as np
        import omni.usd
        from isaacsim.sensors.camera import Camera

        self._np = np
        self.config = config or EnvConfig()
        self.scene_asset = Path(scene_asset).resolve()
        if not self.scene_asset.is_file():
            raise FileNotFoundError(self.scene_asset)
        context = omni.usd.get_context()
        context.open_stage(str(self.scene_asset))
        # open_stage schedules USD/RTX loading asynchronously. Mutating prims
        # while the loader thread is still populating the stage can crash in
        # librtx.scenedb rather than raising a Python exception.
        import omni.kit.app
        app = omni.kit.app.get_app()
        for _ in range(6000):
            if context.get_stage_loading_status()[2] <= 0:
                break
            app.update()
        else:
            raise TimeoutError(f"timed out loading Isaac Sim stage: {self.scene_asset}")
        self.stage = context.get_stage()
        if self.stage is None:
            raise RuntimeError(f"Isaac Sim could not open {self.scene_asset}")

        self._metadata: dict[str, Any] = {}
        self._objects: dict[str, dict[str, Any]] = {}
        self._prim_paths: dict[str, str] = {}
        self._pickable: set[str] = set()
        self._navigation: dict[str, Any] | None = None
        self._room_ids: list[str] = []
        self._room_labels = None
        self._static = None
        self._static_base = None
        self._dynamic = None
        self._dynamic_base = None
        self._door_cells: dict[str, set[tuple[int, int]]] = {}
        self._door_open: set[str] = set()
        self._door_passable: dict[str, bool] = {}
        # Per-leaf visual state.  Each entry stores the original local
        # transform and a world-space hinge pivot so open/close is reversible.
        self._door_visual_rest: dict[str, dict[str, Any]] = {}
        self._origin = None
        self._resolution = 0.25
        self._floor_height = 0.0
        self._held_object: str | None = None
        self._active_task: dict[str, Any] = {}
        self._agent_position = np.array([0.0, 0.0, self.config.camera_height], dtype=float)
        self._initial_state: dict[str, Any] = {}
        self._safe_position = self._agent_position.copy()
        self._safe_yaw = 0.0
        self._safe_pitch = 0.0
        self._session_start_pose: dict[str, Any] | None = None
        self._reset_applied = False
        self._yaw = 0.0
        self._pitch = 0.0
        self._last_rgb = None
        self._last_depth = None
        self._last_action_feedback: dict[str, Any] | None = None
        self._holoagent_sensor_bridge = None
        self._overview_camera_path = "/World/LongTermBenchmark/OverviewCamera"
        self._overview_marker_path = "/World/LongTermBenchmark/OverviewAgent"
        self._overview_camera = None
        self._overview_marker = None
        self._overview_window = None
        self._metadata_bounds = None
        self.camera = Camera(
            prim_path=self.config.camera_prim_path,
            resolution=self.config.resolution,
        )
        # Isaac Sim's Camera initializes an RGB annotator by default.  The
        # optional HoloAgent bridge creates its own RGB-D render product, so
        # the benchmark observation protocol does not need to carry depth.
        self.camera.initialize(attach_rgb_annotator=True)
        try:
            self.camera.add_distance_to_image_plane_to_frame()
        except (AttributeError, RuntimeError):
            pass
        self._configure_camera_intrinsics()
        self._configure_scene_rendering()
        if holoagent_sensor_topics is not None:
            from .holoagent_sensor_bridge import IsaacSimHoloAgentSensorBridge
            self._holoagent_sensor_bridge = IsaacSimHoloAgentSensorBridge(
                self.camera, self.config.resolution, holoagent_sensor_topics)

    def attach_overview_window(self, window: Any) -> None:
        """Keep a strong reference so Kit does not garbage-collect the viewport."""
        self._overview_window = window

    def _configure_camera_intrinsics(self) -> None:
        """Set an explicit horizontal FOV instead of relying on Kit defaults."""
        fov = float(self.config.horizontal_fov_degrees)
        if not 1.0 < fov < 179.0:
            raise ValueError("horizontal_fov_degrees must be between 1 and 179")
        focal = float(self.config.focal_length_mm)
        aperture = 2.0 * focal * math.tan(math.radians(fov) / 2.0)
        try:
            self.camera.set_focal_length(focal)
            self.camera.set_horizontal_aperture(aperture, maintain_square_pixels=True)
            if hasattr(self.camera, "set_vertical_aperture"):
                self.camera.set_vertical_aperture(aperture)
        except (AttributeError, TypeError):
            # Keep compatibility with Camera wrappers that expose only the USD
            # prim.  This is still one explicit configuration path, not a
            # fallback to the implicit default FOV.
            from pxr import UsdGeom
            prim = UsdGeom.Camera.Get(self.stage, self.config.camera_prim_path)
            prim.GetFocalLengthAttr().Set(focal)
            prim.GetHorizontalApertureAttr().Set(aperture)
            prim.GetVerticalApertureAttr().Set(aperture * self.config.resolution[1] / self.config.resolution[0])

        # Keep the near plane in front of the agent camera and the far plane
        # large enough for the complete indoor scene.  This does not affect
        # navigation collision checks; it only controls camera rendering.
        try:
            self.camera.set_clipping_range(near_distance=0.05, far_distance=1000.0)
        except (AttributeError, TypeError):
            from pxr import UsdGeom
            prim = UsdGeom.Camera.Get(self.stage, self.config.camera_prim_path)
            prim.GetClippingRangeAttr().Set((0.05, 1000.0))

    def _configure_scene_rendering(self) -> None:
        """Make imported architectural meshes visible from both sides.

        Some USDZ exports store room walls as single-sided meshes with normals
        facing the exterior.  RTX then culls the interior-facing wall surface,
        which makes the first-person view appear to see through walls.  The
        scene remains unchanged geometrically; this only authors a render-time
        double-sided override on the session layer.
        """
        from pxr import UsdGeom
        for prim in self.stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            attribute = mesh.GetDoubleSidedAttr()
            if not attribute:
                attribute = mesh.CreateDoubleSidedAttr()
            attribute.Set(True)

    @property
    def camera_parameters(self) -> dict[str, Any]:
        return {
            "resolution": list(self.config.resolution),
            "horizontal_fov_degrees": float(self.config.horizontal_fov_degrees),
            "focal_length_mm": float(self.config.focal_length_mm),
        }

    def render_once(self) -> None:
        """Force one Replicator render for cameras on Kit versions where
        ``app.update()`` alone does not advance the render product."""
        try:
            import omni.replicator.core as rep
            rep.orchestrator.step(rt_subframes=1)
        except Exception:
            # The normal Kit update path remains valid on versions that do not
            # expose the orchestrator or when a viewer drives rendering.
            return

    def configure_scene(
        self, metadata: dict[str, Any], navigation: dict[str, Any],
        manipulation: dict[str, Any], state: dict[str, Any],
    ) -> None:
        self._metadata = metadata
        self._objects = {str(key): dict(value) for key, value in state.get("objects", {}).items()}
        self._prim_paths = {str(raw["object_id"]): str(raw["prim_path"])
                            for raw in metadata.get("objects", []) if raw.get("object_id") and raw.get("prim_path")}
        self._pickable = {str(item["object_id"]) for item in manipulation.get("pick_place_candidates", [])}
        self._navigation = navigation
        self._room_ids = [str(value) for value in navigation["room_ids"].tolist()]
        self._room_labels = navigation["room_labels"]
        self._static = navigation["static_occupancy"]
        self._static_base = navigation.get("static_base_occupancy", self._static)
        self._dynamic = navigation["dynamic_occupancy"]
        self._dynamic_base = self._dynamic.copy()
        self._origin = navigation["origin_xy"]
        self._resolution = float(navigation["resolution"])
        self._floor_height = float(navigation["floor_height"])
        self._door_cells = self._build_door_cells(metadata)
        # Some cached navigation files rasterize the wall behind a door as
        # static occupancy.  Keep the opening reserved for the dynamic panel;
        # otherwise opening the door clears the panel but leaves a three-cell
        # wall plug and ``passable`` remains false.  The door-cell margin
        # already includes the avatar radius, so this does not widen unrelated
        # wall segments.
        self._static_base = self._static_base.copy()
        for cells in self._door_cells.values():
            for cell_x, cell_y in cells:
                self._static_base[cell_y, cell_x] = 0
        self._door_open = set()
        self._door_passable = {}
        self._door_visual_rest = {}
        self._metadata_bounds = self._bounds_from_metadata(metadata)
        self._initial_state = {"objects": {key: dict(value) for key, value in self._objects.items()}}
        free = self._np.argwhere((self._static == 0) & (self._dynamic == 0))
        if len(free):
            y, x = free[len(free) // 2]
            self._safe_position = self._np.array([
                self._origin[0] + (int(x) + .5) * self._resolution,
                self._origin[1] + (int(y) + .5) * self._resolution,
                self._floor_height + self.config.camera_height,
            ], dtype=float)
            self._agent_position = self._safe_position.copy()
        self._set_all_object_positions()
        self._author_camera_pose()
        self._last_action_feedback = None

    def _build_door_cells(self, metadata: dict[str, Any]) -> dict[str, set[tuple[int, int]]]:
        """Map door AABBs to dynamic-grid cells for open/close updates."""
        result: dict[str, set[tuple[int, int]]] = {}
        for entity in metadata.get("objects", []):
            object_id = str(entity.get("object_id", ""))
            if str(entity.get("category", "")).lower() != "door" or not object_id:
                continue
            bounds = entity.get("geometry", {}).get("world_aabb", {})
            minimum, maximum = bounds.get("min"), bounds.get("max")
            if not minimum or not maximum:
                continue
            margin = max(float(self.config.robot_radius),
                         float(self._navigation.get("robot_radius", 0.0)))
            x0 = int(math.floor((float(minimum[0]) - margin - float(self._origin[0])) / self._resolution))
            x1 = int(math.ceil((float(maximum[0]) + margin - float(self._origin[0])) / self._resolution))
            y0 = int(math.floor((float(minimum[1]) - margin - float(self._origin[1])) / self._resolution))
            y1 = int(math.ceil((float(maximum[1]) + margin - float(self._origin[1])) / self._resolution))
            result[object_id] = {(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)
                                 if 0 <= y < self._dynamic.shape[0] and 0 <= x < self._dynamic.shape[1]}
        return result

    def setup_overview(self) -> str:
        """Create a full-scene top-down camera and an agent direction marker."""
        from pxr import Gf, UsdGeom
        camera = UsdGeom.Camera.Define(self.stage, self._overview_camera_path)
        min_x, min_y, min_z, max_x, max_y, max_z = self._scene_xy_bounds()
        extent_x = max(1.0, max_x - min_x)
        extent_y = max(1.0, max_y - min_y)
        aspect = 420.0 / 300.0
        vertical_size = max(extent_y, extent_x / aspect) * 1.15
        camera.GetProjectionAttr().Set("orthographic")
        # In USD's Camera schema, orthographic framing uses the aperture
        # attributes (there is no GetOrthographicVerticalSizeAttr in Isaac
        # Sim 6.0's generated Python wrapper).
        # USD camera apertures are expressed in tenths of a scene unit.
        camera.GetVerticalApertureAttr().Set(vertical_size * 10.0)
        camera.GetHorizontalApertureAttr().Set(vertical_size * aspect * 10.0)
        camera.GetClippingRangeAttr().Set((0.1, max(1000.0, max_z - min_z + 100.0)))
        center = Gf.Vec3d(
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            max_z + max(8.0, vertical_size),
        )
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        translate = xform.AddTranslateOp()
        orient = xform.AddOrientOp()
        translate.Set(center)
        # USD cameras look along local -Z, so identity orientation is a true
        # top-down view when the camera is above the XY floor plane.
        orient.Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        self._overview_camera = camera

        root = UsdGeom.Xform.Define(self.stage, self._overview_marker_path)
        arrow = UsdGeom.Mesh.Define(self.stage, self._overview_marker_path + "/Arrow")
        # A flat plan-view arrow: +X is the agent forward direction.
        arrow.CreatePointsAttr().Set([Gf.Vec3f(0.9, 0.0, 0.0), Gf.Vec3f(0.15, 0.38, 0.0),
                                      Gf.Vec3f(0.15, 0.14, 0.0), Gf.Vec3f(-0.7, 0.14, 0.0),
                                      Gf.Vec3f(-0.7, -0.14, 0.0), Gf.Vec3f(0.15, -0.14, 0.0),
                                      Gf.Vec3f(0.15, -0.38, 0.0)])
        arrow.CreateFaceVertexCountsAttr().Set([7])
        arrow.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 3, 4, 5, 6])
        arrow.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.75, 0.05)])
        arrow.CreateDoubleSidedAttr().Set(True)
        arrow_scale = max(0.5, min(5.0, vertical_size * 0.035))
        UsdGeom.Xformable(arrow.GetPrim()).AddScaleOp().Set(
            Gf.Vec3f(arrow_scale, arrow_scale, arrow_scale))
        self._overview_marker = root
        self._update_overview_marker()
        return self._overview_camera_path

    def _scene_xy_bounds(self) -> tuple[float, float, float, float, float, float]:
        """Return complete scene bounds from metadata, then USD, then grid."""
        from pxr import Usd, UsdGeom

        if self._metadata_bounds is not None:
            return self._metadata_bounds

        try:
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_],
                                      useExtentsHint=True)
            bounds = cache.ComputeWorldBound(self.stage.GetPseudoRoot()).GetRange()
            minimum, maximum = bounds.GetMin(), bounds.GetMax()
            values = tuple(float(v) for v in (*minimum, *maximum))
            if all(math.isfinite(v) for v in values) and values[3] > values[0] and values[4] > values[1]:
                return values
        except Exception:
            pass
        extent_x = float(self._static.shape[1] * self._resolution) if self._static is not None else 10.0
        extent_y = float(self._static.shape[0] * self._resolution) if self._static is not None else 10.0
        origin_x, origin_y = (float(v) for v in (self._origin or [0.0, 0.0]))
        return (origin_x, origin_y, float(self._floor_height), origin_x + extent_x,
                origin_y + extent_y, float(self._floor_height + 3.0))

    @staticmethod
    def _bounds_from_metadata(metadata: dict[str, Any]):
        """Union every authored object AABB and room boundary polygon."""
        points: list[list[float]] = []
        for entity in metadata.get("objects", []):
            aabb = entity.get("geometry", {}).get("world_aabb", {})
            if aabb.get("min") and aabb.get("max"):
                points.extend([aabb["min"], aabb["max"]])
        for room in metadata.get("rooms", []):
            points.extend([[point[0], point[1], room.get("floor_height", 0.0)]
                           for point in room.get("boundary_polygon", [])])
        if not points:
            return None
        values = [[float(value) for value in point[:3]] for point in points]
        return (min(point[0] for point in values), min(point[1] for point in values),
                min(point[2] for point in values), max(point[0] for point in values),
                max(point[1] for point in values), max(point[2] for point in values))

    def _update_overview_marker(self) -> None:
        if self._overview_marker is None:
            return
        from pxr import Gf
        position = Gf.Vec3d(float(self._agent_position[0]), float(self._agent_position[1]),
                            float(self._floor_height + 0.12))
        api = __import__("pxr.UsdGeom", fromlist=["XformCommonAPI"]).XformCommonAPI(
            self._overview_marker.GetPrim())
        api.SetTranslate(position)
        api.SetRotate(Gf.Vec3f(0.0, 0.0, math.degrees(self._yaw)))

    def apply_state(self, state: dict[str, Any]) -> None:
        """Apply a composed session state without recreating the stage."""
        self._objects = deepcopy(state.get("objects", {}))
        self._initial_state = deepcopy(state)
        self._set_all_object_positions()
        self._restore_door_state()

    def runtime_snapshot(self) -> dict[str, Any]:
        """Return mutable runtime state that must survive session boundaries."""
        return {
            "objects": deepcopy(self._objects),
            "agent_pose": self.get_agent_pose(),
            "held_object": self._held_object,
            "door_open": sorted(self._door_open),
            "yaw": float(self._yaw),
            "pitch": float(self._pitch),
        }

    def apply_world_state(self, state: dict[str, Any], *, preserve_runtime: bool = True) -> None:
        """Apply logical state while preserving continuous runtime details."""
        snapshot = self.runtime_snapshot() if preserve_runtime else {}
        self._objects = deepcopy(state.get("objects", {}))
        if preserve_runtime:
            for object_id in snapshot.get("door_open", []):
                if object_id in self._objects:
                    self._objects[object_id]["door_open"] = True
        self._initial_state = deepcopy(state)
        self._set_all_object_positions()
        self._restore_door_state()
        held = snapshot.get("held_object") if preserve_runtime else None
        if held in self._objects:
            self._held_object = str(held)
            self._objects[self._held_object]["held"] = True
            self._set_kinematic(self._held_object, True)
            self._follow_held_object()
        else:
            self._held_object = None

    def _restore_door_state(self) -> None:
        desired = {str(object_id) for object_id, entity in self._objects.items()
                   if str(entity.get("category", "")).lower() == "door"
                   and bool(entity.get("door_open", False))}
        for object_id in list(self._door_open - desired):
            self._set_door_open(object_id, False)
        for object_id in sorted(desired - self._door_open):
            self._set_door_open(object_id, True)

    def begin_session(self, start_pose: dict[str, Any] | None, *, apply_pose: bool = True) -> None:
        """Set the session reset anchor without changing world state."""
        if start_pose is not None and apply_pose:
            self.set_agent_pose(start_pose, update_reset_pose=True)
        if not self._walkable(self._agent_position):
            self._agent_position = self._nearest_walkable_position(self._agent_position)
            self._safe_position = self._agent_position.copy()
            self._author_camera_pose()
            self._update_overview_marker()
        self._session_start_pose = deepcopy(self.get_agent_pose())
        self._reset_applied = False

    def runtime_state(self) -> dict[str, Any]:
        return {"schema_version": 1, "objects": deepcopy(self._objects),
                "held_object": self._held_object,
                "door_open": sorted(self._door_open),
                "agent_pose": self.get_agent_pose()}

    def set_task(self, task: dict[str, Any]) -> None:
        self._active_task = dict(task)

    def set_agent_pose(self, pose: dict[str, Any], *, update_reset_pose: bool = True) -> None:
        position = pose.get("position")
        if position is not None:
            self._agent_position = self._np.array(position, dtype=float)
            if update_reset_pose:
                self._safe_position = self._agent_position.copy()
        orientation = pose.get("orientation")
        if orientation is not None:
            self._yaw = _quat_to_yaw(orientation)
            if update_reset_pose:
                self._safe_yaw = self._yaw
                self._safe_pitch = self._pitch
        self._author_camera_pose()
        self._update_overview_marker()

    def _set_all_object_positions(self) -> None:
        for object_id, entity in self._objects.items():
            position = entity.get("pose", {}).get("position")
            if position is not None:
                self._set_object_world_position(object_id, position)

    def _prim(self, object_id: str):
        prim_path = self._prim_paths.get(object_id)
        if not prim_path:
            raise KeyError(f"object has no prim_path: {object_id}")
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim:
            raise KeyError(f"object prim not found: {object_id} ({prim_path})")
        return prim

    def _set_object_world_position(self, object_id: str, position: Any) -> None:
        from pxr import Gf, UsdGeom
        prim = self._prim(object_id)
        xform = UsdGeom.Xformable(prim)
        matrix = xform.ComputeLocalToWorldTransform(0)
        matrix.SetTranslateOnly(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        parent = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(0) if prim.GetParent() else Gf.Matrix4d(1.0)
        xform.MakeMatrixXform().Set(matrix * parent.GetInverse())
        if object_id in self._objects:
            self._objects[object_id].setdefault("pose", {})["position"] = [float(v) for v in position]

    def _get_object_world_position(self, object_id: str):
        from pxr import UsdGeom
        matrix = UsdGeom.Xformable(self._prim(object_id)).ComputeLocalToWorldTransform(0)
        value = matrix.ExtractTranslation()
        return self._np.array([float(value[0]), float(value[1]), float(value[2])], dtype=float)

    def _author_camera_pose(self) -> None:
        self.camera.set_world_pose(
            position=self._agent_position,
            orientation=_camera_quaternion(self._yaw, self._pitch),
            camera_axes="world",
        )

    def get_agent_pose(self) -> dict[str, Any]:
        return {"position": self._agent_position.tolist(), "orientation": _yaw_quaternion(self._yaw),
                "yaw_degrees": math.degrees(self._yaw), "pitch_degrees": math.degrees(self._pitch)}

    def get_camera_pose(self) -> dict[str, Any]:
        """Return the rendered camera pose in ROS quaternion order (x,y,z,w).

        The benchmark's public ``pose`` is a navigation/base pose and keeps a
        yaw-only quaternion for compatibility.  RGB-D consumers need the
        orientation of the actual render camera, including look pitch.
        """
        w, x, y, z = _camera_quaternion(self._yaw, self._pitch)
        return {
            "position": self._agent_position.tolist(),
            "orientation": [x, y, z, w],
            "yaw_degrees": math.degrees(self._yaw),
            "pitch_degrees": math.degrees(self._pitch),
        }

    def _cell(self, position: Any) -> tuple[int, int]:
        return (int(math.floor((float(position[0]) - float(self._origin[0])) / self._resolution)),
                int(math.floor((float(position[1]) - float(self._origin[1])) / self._resolution)))

    def _collision_radius(self) -> float:
        """Return the footprint radius for the avatar and any carried object.

        Pick/place is represented by a kinematic object that follows the
        avatar. Treating only the avatar as the swept body lets a large item
        (for example a computer) clip through a closed doorway even though the
        avatar itself is collision checked. A conservative circular radius
        from the carried object's world AABB keeps the high-level sweep
        physically consistent without depending on PhysX contact callbacks.
        """
        radius = max(0.0, float(self.config.robot_radius))
        if self._held_object:
            bounds = self._objects.get(self._held_object, {}).get("geometry", {}).get("world_aabb", {})
            minimum, maximum = bounds.get("min"), bounds.get("max")
            if minimum and maximum and len(minimum) >= 2 and len(maximum) >= 2:
                extent_x = max(0.0, float(maximum[0]) - float(minimum[0]))
                extent_y = max(0.0, float(maximum[1]) - float(minimum[1]))
                radius = max(radius, 0.5 * math.hypot(extent_x, extent_y))
        return radius

    def _walkable(self, position: Any) -> bool:
        """Return whether the agent's 2-D circular footprint is collision-free."""
        static = self._static_base if self._static_base is not None else self._static
        if static is None or self._dynamic is None:
            return False
        radius = max(0.0, float(self.config.robot_radius))
        center_x, center_y = float(position[0]), float(position[1])
        min_x = int(math.floor((center_x - radius - float(self._origin[0])) / self._resolution))
        max_x = int(math.floor((center_x + radius - float(self._origin[0])) / self._resolution))
        min_y = int(math.floor((center_y - radius - float(self._origin[1])) / self._resolution))
        max_y = int(math.floor((center_y + radius - float(self._origin[1])) / self._resolution))
        height, width = static.shape
        for cell_y in range(min_y, max_y + 1):
            for cell_x in range(min_x, max_x + 1):
                if cell_x < 0 or cell_y < 0 or cell_x >= width or cell_y >= height:
                    return False
                door_cell_open = any(
                    object_id in self._door_open and (cell_x, cell_y) in cells
                    for object_id, cells in self._door_cells.items())
                if not bool(static[cell_y, cell_x] or
                            (self._dynamic[cell_y, cell_x] and not door_cell_open)):
                    continue
                left = float(self._origin[0]) + cell_x * self._resolution
                right = left + self._resolution
                bottom = float(self._origin[1]) + cell_y * self._resolution
                top = bottom + self._resolution
                dx = max(left - center_x, 0.0, center_x - right)
                dy = max(bottom - center_y, 0.0, center_y - top)
                if dx * dx + dy * dy <= radius * radius:
                    return False
        return True

    def _carried_door_clear(self, position: Any) -> bool:
        """Check the carried item's footprint against closed door panels.

        The avatar's normal sweep remains responsible for static furniture and
        walls.  The carried item is checked separately against door cells so a
        pickup next to a desk does not become an impossible starting overlap,
        while a computer still cannot be swept through a closed doorway.
        """
        if not self._held_object:
            return True
        radius = self._collision_radius()
        center_x, center_y = float(position[0]), float(position[1])
        for object_id, cells in self._door_cells.items():
            if object_id in self._door_open:
                continue
            for cell_x, cell_y in cells:
                left = float(self._origin[0]) + cell_x * self._resolution
                right = left + self._resolution
                bottom = float(self._origin[1]) + cell_y * self._resolution
                top = bottom + self._resolution
                dx = max(left - center_x, 0.0, center_x - right)
                dy = max(bottom - center_y, 0.0, center_y - top)
                if dx * dx + dy * dy <= radius * radius:
                    return False
        return True

    def _nearest_walkable_position(self, position: Any, max_distance: float = 2.0) -> Any:
        """Snap an invalid authored start to the nearest collision-free cell center."""
        center_x, center_y = self._cell(position)
        limit = max(1, int(math.ceil(max_distance / self._resolution)))
        candidates = []
        for dy in range(-limit, limit + 1):
            for dx in range(-limit, limit + 1):
                cell_x, cell_y = center_x + dx, center_y + dy
                candidate = self._np.array([
                    float(self._origin[0]) + (cell_x + 0.5) * self._resolution,
                    float(self._origin[1]) + (cell_y + 0.5) * self._resolution,
                    float(position[2]),
                ], dtype=float)
                distance = float(self._np.linalg.norm(candidate[:2] - self._np.asarray(position)[:2]))
                if distance <= max_distance and self._walkable(candidate):
                    candidates.append((distance, cell_y, cell_x, candidate))
        if not candidates:
            raise RuntimeError(
                f"no collision-free agent start within {max_distance:.2f} m of "
                f"{self._np.asarray(position).tolist()}"
            )
        return min(candidates, key=lambda item: item[:3])[3]

    def _nearest_door(self, max_distance: float = 1.25) -> tuple[str, float] | None:
        candidates = []
        for object_id in self._door_cells:
            entity = self._objects.get(object_id, {})
            bounds = entity.get("geometry", {}).get("world_aabb", {})
            minimum, maximum = bounds.get("min"), bounds.get("max")
            if not minimum or not maximum:
                continue
            x, y = self._agent_position[:2]
            dx = max(float(minimum[0]) - x, 0.0, x - float(maximum[0]))
            dy = max(float(minimum[1]) - y, 0.0, y - float(maximum[1]))
            distance = math.hypot(dx, dy)
            if distance <= max_distance:
                candidates.append((distance, object_id))
        if not candidates:
            return None
        distance, object_id = min(candidates)
        return object_id, distance

    def _nearby_doors(self, max_distance: float = 3.0) -> list[dict[str, Any]]:
        values = []
        for object_id in self._door_cells:
            entity = self._objects.get(object_id, {})
            bounds = entity.get("geometry", {}).get("world_aabb", {})
            minimum, maximum = bounds.get("min"), bounds.get("max")
            if not minimum or not maximum:
                continue
            x, y = self._agent_position[:2]
            dx = max(float(minimum[0]) - x, 0.0, x - float(maximum[0]))
            dy = max(float(minimum[1]) - y, 0.0, y - float(maximum[1]))
            distance = math.hypot(dx, dy)
            if distance <= max_distance:
                values.append({"object_id": object_id, "distance_m": distance,
                               "open": object_id in self._door_open,
                               "passable": self._door_passable.get(object_id)})
        return sorted(values, key=lambda item: item["distance_m"])

    def _nearby_manipulable_objects(self, max_distance: float = 3.0) -> list[dict[str, Any]]:
        """Expose local action affordances, never task targets or ground truth."""
        values = []
        for object_id in self._pickable:
            entity = self._objects.get(object_id, {})
            distance = float(self._np.linalg.norm(
                self._get_object_world_position(object_id)[:2] - self._agent_position[:2]))
            if distance <= max_distance:
                values.append({"object_id": object_id,
                               "category": entity.get("category"),
                               "room_id": entity.get("room_id"),
                               "distance_m": distance,
                               "reachable": distance <= 1.5})
        return sorted(values, key=lambda item: item["distance_m"])

    def _door_passage_connected(self, object_id: str) -> bool:
        """Check that a circular footprint can cross an opened doorway locally."""
        entity = self._objects.get(object_id, {})
        bounds = entity.get("geometry", {}).get("world_aabb", {})
        minimum, maximum = bounds.get("min"), bounds.get("max")
        if not minimum or not maximum or self._resolution <= 0:
            return False
        # The thin AABB axis is the direction through the doorway.  Search a
        # padded local window and connect free cell centres with 4-neighbour
        # motion; _walkable applies the exact circular footprint test.
        extents = [float(maximum[i]) - float(minimum[i]) for i in (0, 1)]
        axis = 0 if extents[0] <= extents[1] else 1
        padding = max(1.0, 8.0 * self._resolution)
        x0 = max(0, int(math.floor((float(minimum[0]) - padding - self._origin[0]) / self._resolution)))
        x1 = min(self._static_base.shape[1] - 1,
                 int(math.ceil((float(maximum[0]) + padding - self._origin[0]) / self._resolution)))
        y0 = max(0, int(math.floor((float(minimum[1]) - padding - self._origin[1]) / self._resolution)))
        y1 = min(self._static_base.shape[0] - 1,
                 int(math.ceil((float(maximum[1]) + padding - self._origin[1]) / self._resolution)))
        if x0 > x1 or y0 > y1:
            return False
        free: set[tuple[int, int]] = set()
        for cell_y in range(y0, y1 + 1):
            for cell_x in range(x0, x1 + 1):
                position = [self._origin[0] + (cell_x + .5) * self._resolution,
                            self._origin[1] + (cell_y + .5) * self._resolution,
                            self._agent_position[2]]
                if self._walkable(position):
                    free.add((cell_x, cell_y))
        if not free:
            return False
        low = min(cell[axis] for cell in free)
        high = max(cell[axis] for cell in free)
        starts = {cell for cell in free if cell[axis] == low}
        goals = {cell for cell in free if cell[axis] == high}
        frontier = list(starts)
        visited = set(starts)
        while frontier:
            cell = frontier.pop()
            if cell in goals:
                return True
            for delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = (cell[0] + delta[0], cell[1] + delta[1])
                if neighbour in free and neighbour not in visited:
                    visited.add(neighbour)
                    frontier.append(neighbour)
        return False

    def _set_door_open(self, object_id: str, opened: bool) -> dict[str, Any]:
        if object_id not in self._door_cells:
            return {"success": False, "action": "open" if opened else "close",
                    "reason": "unknown_door", "object_id": object_id, "collision": False}
        if opened and object_id in self._door_open:
            return {"success": True, "action": "open", "reason": "already_open",
                    "object_id": object_id, "collision": False, "door_open": True,
                    "passable": self._door_passable.get(object_id, False)}
        if not opened and object_id not in self._door_open:
            return {"success": True, "action": "close", "reason": "already_closed",
                    "object_id": object_id, "collision": False, "door_open": False}
        self._door_open = (self._door_open | {object_id}) if opened else (self._door_open - {object_id})
        self._dynamic = self._dynamic_base.copy()
        for door_id in self._door_open:
            for x, y in self._door_cells.get(door_id, set()):
                self._dynamic[y, x] = 0
        self._door_passable[object_id] = self._door_passage_connected(object_id) if opened else False
        entity = self._objects.get(object_id)
        if entity is not None:
            entity["door_open"] = opened
        # Rotate each leaf around its own vertical hinge.  The leaf prim is
        # rotated as a whole, so all attached panels/handles follow it; the
        # frame/body remains fixed.  Existing revolute-joint metadata is used
        # when available.  Some assets incorrectly encode a double hinged door
        # as two prismatic joints, so those leaves receive a geometric hinge
        # inferred from their outer edge.
        try:
            from pxr import Gf, Usd, UsdGeom
            root = self._prim(object_id)
            leaves = [prim for prim in root.GetChildren()
                      if str(prim.GetName()).lower() in {"door_1", "door_2"}]
            if not leaves:
                leaves = [prim for prim in root.GetChildren()
                          if str(prim.GetName()).lower().startswith("door_")]
            root_world = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0)
            root_range = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                            [UsdGeom.Tokens.default_],
                                            useExtentsHint=True).ComputeWorldBound(root).GetRange()
            root_center = root_range.GetMidpoint()
            root_extent = root_range.GetSize()
            major_axis = 0 if root_extent[0] >= root_extent[1] else 1
            leaf_states: dict[str, dict[str, Any]] = {}
            for leaf in leaves:
                path = str(leaf.GetPath())
                xform = UsdGeom.Xformable(leaf)
                if path not in self._door_visual_rest:
                    local = xform.GetLocalTransformation()
                    local_matrix = local[0] if isinstance(local, tuple) else local
                    leaf_world = xform.ComputeLocalToWorldTransform(0)
                    leaf_range = UsdGeom.BBoxCache(
                        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_],
                        useExtentsHint=True).ComputeWorldBound(leaf).GetRange()
                    leaf_center = leaf_range.GetMidpoint()
                    hinge_world = None
                    for joint in leaf.GetChildren():
                        if "revolute" not in str(joint.GetName()).lower():
                            continue
                        value = joint.GetAttribute("physics:localPos0").Get()
                        if value is not None:
                            hinge_world = root_world.Transform(
                                Gf.Vec3d(float(value[0]), float(value[1]), float(value[2])))
                            break
                    if hinge_world is None:
                        # Infer the outer hinge edge for assets whose joints
                        # are authored as prismatic constraints.
                        hinge_coords = [float(leaf_center[0]), float(leaf_center[1]),
                                        float(leaf_center[2])]
                        if major_axis == 0:
                            hinge_coords[0] = (float(leaf_range.GetMin()[0])
                                               if leaf_center[0] < root_center[0]
                                               else float(leaf_range.GetMax()[0]))
                        else:
                            hinge_coords[1] = (float(leaf_range.GetMin()[1])
                                               if leaf_center[1] < root_center[1]
                                               else float(leaf_range.GetMax()[1]))
                        hinge_world = Gf.Vec3d(*hinge_coords)
                    side = -1.0 if (leaf_center[major_axis] >= root_center[major_axis]) else 1.0
                    self._door_visual_rest[path] = {
                        "local": local_matrix,
                        "world": leaf_world,
                        "hinge_world": hinge_world,
                        "open_degrees": float(self.config.door_open_angle_degrees) * side,
                    }
                state = self._door_visual_rest[path]
                leaf_states[path] = state
                if opened:
                    pivot = Gf.Matrix4d(1.0)
                    pivot.SetTranslate(state["hinge_world"])
                    inverse = Gf.Matrix4d(1.0)
                    hinge = state["hinge_world"]
                    inverse.SetTranslate(Gf.Vec3d(-float(hinge[0]), -float(hinge[1]),
                                                   -float(hinge[2])))
                    rotation = Gf.Matrix4d(1.0)
                    rotation.SetRotate(Gf.Rotation(
                        Gf.Vec3d(0.0, 0.0, 1.0), state["open_degrees"]))
                    parent = leaf.GetParent()
                    parent_world = (UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(0)
                                    if parent and parent.IsValid() else Gf.Matrix4d(1.0))
                    opened_world = pivot * rotation * inverse * state["world"]
                    xform.MakeMatrixXform().Set(opened_world * parent_world.GetInverse())
                else:
                    xform.MakeMatrixXform().Set(state["local"])

            # Handles are often authored as sibling prims rather than children
            # of the moving door leaf.  Follow their physics fixed/revolute
            # joints (body0 -> door_N) and apply the same hinge transform to
            # the top-level attachment, preserving its own local geometry.
            leaf_paths = set(leaf_states)
            for candidate in root.GetChildren():
                candidate_path = str(candidate.GetPath())
                if candidate_path in leaf_paths or str(candidate.GetName()).lower() in {
                    "body", "looks", "meshs"
                }:
                    continue
                owner = None
                for descendant in Usd.PrimRange(candidate):
                    relationship = descendant.GetRelationship("physics:body0")
                    if not relationship:
                        continue
                    targets = {str(target) for target in relationship.GetTargets()}
                    owner = next((path for path in leaf_paths if path in targets), None)
                    if owner:
                        break
                if owner is None:
                    continue
                state = leaf_states[owner]
                xform = UsdGeom.Xformable(candidate)
                if candidate_path not in self._door_visual_rest:
                    local = xform.GetLocalTransformation()
                    local_matrix = local[0] if isinstance(local, tuple) else local
                    self._door_visual_rest[candidate_path] = {
                        "local": local_matrix,
                        "world": xform.ComputeLocalToWorldTransform(0),
                        "hinge_world": state["hinge_world"],
                        "open_degrees": state["open_degrees"],
                    }
                attachment = self._door_visual_rest[candidate_path]
                if opened:
                    pivot = Gf.Matrix4d(1.0)
                    pivot.SetTranslate(attachment["hinge_world"])
                    hinge = attachment["hinge_world"]
                    inverse = Gf.Matrix4d(1.0)
                    inverse.SetTranslate(Gf.Vec3d(-float(hinge[0]), -float(hinge[1]),
                                                   -float(hinge[2])))
                    rotation = Gf.Matrix4d(1.0)
                    rotation.SetRotate(Gf.Rotation(
                        Gf.Vec3d(0.0, 0.0, 1.0), attachment["open_degrees"]))
                    parent = candidate.GetParent()
                    parent_world = (UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(0)
                                    if parent and parent.IsValid() else Gf.Matrix4d(1.0))
                    opened_world = pivot * rotation * inverse * attachment["world"]
                    xform.MakeMatrixXform().Set(opened_world * parent_world.GetInverse())
                else:
                    xform.MakeMatrixXform().Set(attachment["local"])
        except Exception:
            pass
        return {"success": True, "action": "open" if opened else "close", "reason": "opened" if opened else "closed",
                "object_id": object_id, "collision": False, "door_open": opened,
                "passable": self._door_passable[object_id]}

    def _free_distance(self, start: Any, direction: Any, distance: float) -> float:
        """Find the first collision along a 2-D swept circular footprint."""
        distance = max(0.0, float(distance))
        if distance == 0.0:
            return 0.0
        sample = max(1e-3, min(float(self.config.collision_sample_step), self._resolution / 2.0))
        steps = max(1, int(math.ceil(distance / sample)))
        for index in range(1, steps + 1):
            travelled = min(distance, index * sample)
            candidate = self._np.asarray(start, dtype=float).copy()
            candidate[:2] += self._np.asarray(direction, dtype=float)[:2] * travelled
            if not self._walkable(candidate) or not self._carried_door_clear(candidate):
                return max(0.0, (index - 1) * sample)
        return distance

    def _room_at(self, position: Any) -> str | None:
        x, y = self._cell(position)
        if self._room_labels is None or y < 0 or x < 0 or y >= self._room_labels.shape[0] or x >= self._room_labels.shape[1]:
            return None
        label = int(self._room_labels[y, x])
        return self._room_ids[label - 1] if 1 <= label <= len(self._room_ids) else None

    def _follow_held_object(self) -> None:
        if not self._held_object:
            return
        forward = self._np.array([math.cos(self._yaw), math.sin(self._yaw), 0.0])
        right = self._np.array([-math.sin(self._yaw), math.cos(self._yaw), 0.0])
        position = self._agent_position + forward * 0.75 + right * 0.35 + self._np.array([0.0, 0.0, -0.22])
        self._set_object_world_position(self._held_object, position)

    def _pick(self, action: dict[str, Any]) -> dict[str, Any]:
        object_id = str(action.get("object_id") or action.get("target") or "")
        if object_id not in self._objects and object_id:
            object_id = self._resolve_nearby_object(object_id, self._pickable) or object_id
        if not object_id:
            return {"success": False, "reason": "object_required", "object_id": ""}
        if object_id not in self._objects:
            return {"success": False, "reason": "unknown_object", "object_id": object_id}
        if object_id not in self._pickable:
            return {"success": False, "reason": "object_not_pickable", "object_id": object_id}
        if self._held_object is not None:
            return {"success": False, "reason": "already_holding", "object_id": self._held_object}
        distance = float(self._np.linalg.norm(self._get_object_world_position(object_id)[:2] - self._agent_position[:2]))
        if distance > 1.5:
            return {"success": False, "reason": "object_out_of_reach", "object_id": object_id, "distance": distance}
        self._held_object = object_id
        self._objects[object_id]["held"] = True
        self._set_kinematic(object_id, True)
        self._follow_held_object()
        return {"success": True, "reason": "picked", "object_id": object_id, "held_object": object_id}

    def _resolve_nearby_object(self, query: str, allowed: set[str],
                               max_distance: float = 1.5) -> str | None:
        """Resolve an agent-supplied semantic label among physically reachable objects.

        This is an action affordance, not task supervision: it never consults
        the active task target or ground truth.
        """
        words = {part for part in __import__("re").split(r"[^a-z0-9]+", query.lower()) if part}
        if not words:
            return None
        candidates = []
        for object_id in allowed:
            entity = self._objects.get(object_id, {})
            category = str(entity.get("category", ""))
            label = f"{object_id} {category}".lower()
            score = sum(word in label for word in words)
            if score <= 0:
                continue
            distance = float(self._np.linalg.norm(
                self._get_object_world_position(object_id)[:2] - self._agent_position[:2]))
            if distance <= max_distance:
                candidates.append((-score, distance, object_id))
        return min(candidates)[2] if candidates else None

    def _place(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._held_object:
            return {"success": False, "reason": "not_holding"}
        object_id = self._held_object
        requested_room = str(action.get("room_id") or action.get("location") or "")
        room_id = self._room_at(self._agent_position)
        if room_id is None:
            return {"success": False, "reason": "agent_outside_room", "object_id": self._held_object}
        if requested_room and requested_room != room_id:
            return {"success": False, "reason": "agent_not_in_requested_room",
                    "object_id": self._held_object, "room_id": room_id}
        position = self._agent_position + self._np.array([math.cos(self._yaw), math.sin(self._yaw), 0.0]) * 0.8
        bounds = self._objects[object_id].get("geometry", {}).get("world_aabb", {})
        height = float(bounds.get("max", [0, 0, .1])[2]) - float(bounds.get("min", [0, 0, 0])[2])
        position[2] = self._floor_height + max(0.025, height / 2.0)
        self._set_object_world_position(object_id, position)
        self._objects[object_id]["room_id"] = room_id
        self._objects[object_id]["held"] = False
        self._set_kinematic(object_id, False)
        self._held_object = None
        return {"success": True, "reason": "placed", "object_id": object_id,
                "room_id": self._objects[object_id].get("room_id")}

    def step(self, action: dict[str, Any] | str):
        action = {"action": action} if isinstance(action, str) else dict(action)
        name = str(action.get("action", "no_op"))
        info: dict[str, Any] = {"success": True, "action": name, "collision": False}
        before = self._agent_position.copy()
        if name in {"move_forward", "move_backward", "move_left", "move_right"}:
            requested_distance = max(0.0, float(action.get("distance", self.config.move_step)))
            distance = min(requested_distance, float(self.config.max_move_distance))
            forward = self._np.array([math.cos(self._yaw), math.sin(self._yaw), 0.0])
            right = self._np.array([-math.sin(self._yaw), math.cos(self._yaw), 0.0])
            if name == "move_forward":
                direction = forward
            elif name == "move_backward":
                direction = -forward
            elif name == "move_right":
                direction = right
            else:
                direction = -right
            free_distance = self._free_distance(self._agent_position, direction, distance)
            collision = free_distance + 1e-6 < distance
            executed_distance = distance
            if collision:
                # Never subtract more than the free travel before impact.  A
                # small collision distance must still leave the agent at the
                # last valid sample rather than collapsing to zero movement.
                backoff = min(float(self.config.collision_backoff),
                              max(0.0, free_distance - float(self.config.collision_sample_step)))
                executed_distance = max(0.0, free_distance - backoff)
                while executed_distance > 0.0:
                    candidate = self._agent_position + direction * executed_distance
                    if self._walkable(candidate):
                        break
                    executed_distance = max(0.0, executed_distance - float(self.config.collision_sample_step))
                if executed_distance > 0.0:
                    self._agent_position = self._agent_position + direction * executed_distance
                info.update(success=False, collision=True, reason="blocked",
                            partial=True, collision_distance=float(free_distance),
                            requested_distance=float(requested_distance))
            elif requested_distance > distance:
                self._agent_position = self._agent_position + direction * distance
                info.update(limited=True, reason="max_distance")
            else:
                self._agent_position = self._agent_position + direction * distance
            info["executed_distance"] = float(self._np.linalg.norm(self._agent_position[:2] - before[:2]))
        elif name in {"turn_left", "turn_right"}:
            requested_angle = float(action.get("angle_degrees", self.config.turn_step_degrees))
            angle = max(-float(self.config.max_turn_degrees),
                        min(float(self.config.max_turn_degrees), requested_angle))
            self._yaw += math.radians(angle) * (1.0 if name == "turn_left" else -1.0)
            info["angle_degrees"] = angle
        elif name in {"look_up", "look_down"}:
            angle = float(action.get("angle_degrees", self.config.turn_step_degrees))
            self._pitch += math.radians(angle) * (1.0 if name == "look_up" else -1.0)
            self._pitch = max(math.radians(-80), min(math.radians(80), self._pitch))
        elif name == "pick":
            info = self._pick(action)
        elif name == "place":
            info = self._place(action)
        elif name in {"open", "close"}:
            requested = str(action.get("object_id") or action.get("target") or "")
            if requested not in self._door_cells and requested:
                requested = self._resolve_nearby_object(
                    requested, set(self._door_cells), max_distance=3.0) or requested
            if not requested:
                nearest = self._nearest_door()
                requested = nearest[0] if nearest else ""
            info = self._set_door_open(requested, name == "open")
        elif name == "reset":
            self.reset()
            info.update(reason="reset", reset_applied=True, reset_pose=self.get_agent_pose())
        elif name != "no_op":
            info.update(success=False, reason=f"unsupported_action:{name}")
        self._author_camera_pose()
        self._update_overview_marker()
        self._follow_held_object()
        if name != "reset":
            self._reset_applied = False
        self._last_action_feedback = deepcopy(info)
        import omni.kit.app
        omni.kit.app.get_app().update()
        return self.observe(), 0.0, False, False, info

    def _set_kinematic(self, object_id: str, enabled: bool) -> None:
        from pxr import UsdPhysics
        body = UsdPhysics.RigidBodyAPI(self._prim(object_id))
        attribute = body.GetKinematicEnabledAttr()
        if not attribute:
            attribute = body.CreateKinematicEnabledAttr()
        attribute.Set(bool(enabled))

    def reset(self) -> None:
        pose = self._session_start_pose
        if pose is not None:
            self._agent_position = self._np.array(pose["position"], dtype=float)
            self._yaw = _quat_to_yaw(pose.get("orientation", [0.0, 0.0, 0.0, 1.0]))
            self._pitch = float(pose.get("pitch", 0.0))
        else:
            self._agent_position = self._safe_position.copy()
            self._yaw = self._safe_yaw
            self._pitch = self._safe_pitch
        self._reset_applied = True
        self._author_camera_pose()
        self._update_overview_marker()

    def observe(self) -> dict[str, Any]:
        self._follow_held_object()
        images = {}
        rgb = self.camera.get_rgb()
        try:
            frame = self.camera.get_current_frame()
            depth = frame.get("distance_to_image_plane") if isinstance(frame, dict) else None
            if depth is not None:
                self._last_depth = depth
        except (AttributeError, RuntimeError):
            pass
        if rgb is not None:
            self._last_rgb = rgb
            images["front"] = rgb
        elif self._last_rgb is not None:
            images["front"] = self._last_rgb
        result: dict[str, Any] = {
            "images": images, "depth": self._last_depth, "pose": self.get_agent_pose(),
            "held_object": self._held_object, "room_id": self._room_at(self._agent_position),
            "nearby_doors": self._nearby_doors(),
            "nearby_manipulable_objects": self._nearby_manipulable_objects(),
            "action_feedback": deepcopy(self._last_action_feedback),
            "reset_applied": bool(self._reset_applied),
        }
        if self._holoagent_sensor_bridge is not None:
            self._holoagent_sensor_bridge.publish_pose(self.get_camera_pose())
        return result

    def try_observe(self) -> dict[str, Any] | None:
        try:
            value = self.observe()
        except Exception:
            return None
        return value if value.get("images") else None

    def state_changes(self, initial_state: dict[str, Any]) -> list[dict[str, Any]]:
        changes = []
        for object_id, before in initial_state.get("objects", {}).items():
            if object_id not in self._objects:
                continue
            after = self._objects[object_id]
            before_pos = before.get("pose", {}).get("position")
            after_pos = after.get("pose", {}).get("position")
            room_before = before.get("room_id")
            room_after = after.get("room_id")
            support_before = before.get("support_object_id")
            support_after = after.get("support_object_id")
            door_state_changed = before.get("door_open") != after.get("door_open")
            moved = before_pos and after_pos and any(abs(float(a) - float(b)) > 1e-3 for a, b in zip(before_pos, after_pos))
            if moved or room_before != room_after or support_before != support_after or door_state_changed:
                changes.append({"object_id": object_id,
                                "after": {"pose": dict(after.get("pose", {})), "room_id": room_after,
                                           "support_object_id": support_after,
                                           "door_open": after.get("door_open")},
                                "source": "high_level_action"})
        return changes

    def close(self) -> None:
        self._overview_window = None
        if self._holoagent_sensor_bridge is not None:
            self._holoagent_sensor_bridge.close()
            self._holoagent_sensor_bridge = None
        try:
            self.camera.close()
        except Exception:
            pass
