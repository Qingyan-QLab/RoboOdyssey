"""CPU-validated robot profiles and optional USD bodies for the high-level avatar."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

from ..artifacts.layout import sha256_file

DEFAULT_ROBOT = "stretch3"
OFFICIAL_EMBODIMENT_VERSION = "stretch3_kinematic_circle_v1"
OFFICIAL_PROFILE_SHA256 = "77817109dbf2382f623889e181a8092184be74ce16c865716b1285a43dbf0be1"
OFFICIAL_ASSETS_SHA256 = "314f3bf749630681f50bd836b6837fb4b79fd7d61131372e322dac68345d45b3"

ROBOT_NAMES = ("cylinder", "stretch3", "tiago", "turtlebot3_burger", "turtlebot4", "go2", "g1")


def robot_assets_root(path: str | Path | None = None) -> Path:
    """Explicit directory, environment setting, then bundled public assets."""
    value = path if path is not None else os.environ.get("BENCHMARK_ROBOT_ASSETS")
    return Path(value).expanduser().resolve() if value else Path(__file__).resolve().parent / "robot_assets"


def validate_robot_selection(robot: str | None, profile: str | Path | None,
                             assets: str | Path | None = None) -> None:
    if robot is not None and robot not in ROBOT_NAMES:
        raise ValueError(f"unknown robot {robot!r}; choose from {', '.join(ROBOT_NAMES)}")
    if robot is not None and profile is not None:
        raise ValueError("robot and robot_profile are mutually exclusive")
    if profile is not None and assets is not None:
        raise ValueError("robot_assets applies to named robots, not robot_profile")


@dataclass(frozen=True)
class RobotSelection:
    name: str
    body: RobotBodyConfig | None
    profile_path: Path | None

    def identity(self) -> dict:
        parameters = self.body.identity() if self.body else {
            "model_id": "cylinder", "mode": "high_level_avatar",
            "robot_radius_m": .1, "camera_height_m": 1.5,
        }
        return {**parameters, "selection": self.name,
                "motion_model": "high_level_xy_yaw", "collision_model": "2d_circle",
                "manipulation_model": "logical_pick_place"}


def resolve_robot(robot: str | None = None, *, profile: str | Path | None = None,
                  assets: str | Path | None = None) -> RobotSelection:
    """Resolve a named body or custom profile; the cylinder needs no asset files."""
    validate_robot_selection(robot, profile, assets)
    name = "custom" if profile is not None else robot or DEFAULT_ROBOT
    if name == "cylinder":
        return RobotSelection(name, None, None)
    path = Path(profile).expanduser().resolve() if profile is not None else robot_assets_root(assets) / "profiles" / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"robot profile not found: {path}; prepare the robot asset package "
                                "and set --robot-assets or BENCHMARK_ROBOT_ASSETS")
    body = (default_robot_body() if name == DEFAULT_ROBOT and profile is None and assets is None
            and not os.environ.get("BENCHMARK_ROBOT_ASSETS") else load_robot_profile(path))
    if name != "custom" and body.model_id != name:
        raise ValueError(f"robot profile model_id {body.model_id!r} does not match selection {name!r}")
    return RobotSelection(name, body, path)


def main(argv=None) -> int:
    """List local body choices; optional full integrity verification is CPU-only."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-assets", type=Path)
    parser.add_argument("--check", action="store_true", help="verify all six USD packages and fail on missing/corrupt files")
    args = parser.parse_args(argv)
    rows = []
    for name in ROBOT_NAMES:
        try:
            if name == "cylinder" or args.check:
                selection = resolve_robot(name, assets=args.robot_assets)
                rows.append({"robot": name, "status": "builtin" if name == "cylinder" else "verified",
                             **selection.identity()})
            else:
                path = robot_assets_root(args.robot_assets) / "profiles" / f"{name}.json"
                rows.append({"robot": name, "status": "profile_found_unverified" if path.is_file() else "missing",
                             "profile": str(path)})
        except (ValueError, OSError, TypeError) as exc:
            rows.append({"robot": name, "status": "error", "error": str(exc)})
    print(json.dumps({"robot_assets": str(robot_assets_root(args.robot_assets)), "robots": rows},
                     ensure_ascii=False, indent=2))
    return 1 if args.check and any(row["status"] == "error" for row in rows) else 0


@dataclass(frozen=True)
class RobotBodyConfig:
    model_id: str
    asset: Path
    robot_radius_m: float
    camera_height_m: float
    base_offset_z_m: float
    profile_sha256: str
    assets_sha256: str
    profile_version: str = "unversioned"

    def identity(self) -> dict:
        """Portable evaluation identity; host absolute paths are not hashed."""
        return {"model_id": self.model_id, "mode": "kinematic_avatar",
                "profile_sha256": self.profile_sha256, "assets_sha256": self.assets_sha256,
                "profile_version": self.profile_version,
                "robot_radius_m": self.robot_radius_m, "camera_height_m": self.camera_height_m,
                "base_offset_z_m": self.base_offset_z_m}

    def body_position(self, camera_position) -> tuple[float, float, float]:
        return (float(camera_position[0]), float(camera_position[1]),
                float(camera_position[2]) - self.camera_height_m + self.base_offset_z_m)


def load_robot_profile(path: str | Path) -> RobotBodyConfig:
    """Validate the complete declared asset package without loading Isaac/USD."""
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("mode") != "kinematic_avatar":
        raise ValueError("unsupported robot profile; expected schema 1 / kinematic_avatar")
    if not isinstance(data.get("model_id"), str) or not data["model_id"].strip():
        raise ValueError("robot profile requires model_id")
    values = {}
    for key in ("robot_radius_m", "camera_height_m", "base_offset_z_m"):
        value = data.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if key != "base_offset_z_m" and value <= 0:
            raise ValueError(f"{key} must be positive")
        values[key] = float(value)
    if not isinstance(data.get("asset"), str):
        raise ValueError("robot profile requires an asset path")
    asset = (path.parent / data["asset"]).resolve()
    if asset.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("robot body must be a USD asset")
    files = data.get("asset_files")
    if not isinstance(files, list) or not files:
        raise ValueError("robot profile requires asset_files with SHA-256 checksums")
    seen = set()
    records = []
    for record in files:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("invalid robot asset file record")
        filename = (path.parent / record["path"]).resolve()
        # A profile may live beside an asset directory, but dependencies must
        # belong to the asset package rather than arbitrary files on the host.
        filename.relative_to(asset.parent)
        if filename in seen:
            raise ValueError("duplicate robot asset file")
        seen.add(filename)
        actual = sha256_file(filename)
        if actual != record.get("sha256"):
            raise ValueError(f"robot asset checksum mismatch: {filename}")
        records.append({"path": filename.relative_to(asset.parent).as_posix(), "sha256": actual})
    if asset not in seen:
        raise ValueError("robot root asset is missing from asset_files")
    fingerprint = hashlib.sha256(json.dumps(sorted(records, key=lambda r: r["path"]), sort_keys=True).encode()).hexdigest()
    return RobotBodyConfig(data["model_id"], asset, **values,
                           profile_sha256=hashlib.sha256(raw).hexdigest(), assets_sha256=fingerprint,
                           profile_version=data.get("profile_version", "unversioned"))


def default_robot_body() -> RobotBodyConfig:
    """Pinned public Stretch3 body; independent of checkout and environment overrides."""
    body = load_robot_profile(Path(__file__).resolve().parent / "robot_assets/profiles/stretch3.json")
    if (body.profile_sha256 != OFFICIAL_PROFILE_SHA256 or body.assets_sha256 != OFFICIAL_ASSETS_SHA256
            or body.profile_version != OFFICIAL_EMBODIMENT_VERSION):
        raise ValueError("bundled Stretch3 identity differs from the pinned official profile")
    return body


def default_robot_identity() -> dict:
    """Portable canonical identity shared by preparation, execution and evaluation."""
    return RobotSelection(DEFAULT_ROBOT, default_robot_body(), None).identity()


def robot_identity_for_config(config) -> dict:
    body = config.robot_body
    selected = RobotSelection(body.model_id if body else "cylinder", body, None).identity()
    selected["robot_radius_m"] = float(config.robot_radius)
    selected["camera_height_m"] = float(config.camera_height)
    return selected


def _scene_version(scene):
    if hasattr(scene, "_requests"):
        scene = scene._requests
    # Standalone Scene/Session bundles expose resources through their SceneSpec.
    if hasattr(scene, "scene"):
        scene = scene.scene
    if hasattr(scene, "data"):
        scene = scene.data
    if hasattr(scene, "version"):
        scene = scene.version
    return scene


def scene_robot_identity(scene) -> dict | None:
    """Return a declared scene body; absent declarations remain explicitly legacy."""
    scene = _scene_version(scene)
    if isinstance(scene, dict):
        manifest = scene
    elif hasattr(scene, "manifest_path"):
        manifest = json.loads(scene.manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = scene.manifest
    identity = manifest.get("robot_profile")
    if identity is None:
        return None
    required = {"model_id", "mode", "robot_radius_m", "camera_height_m",
                "motion_model", "collision_model", "manipulation_model"}
    if not isinstance(identity, dict) or required - identity.keys():
        raise ValueError("scene robot_profile requires a complete robot identity")
    for key in ("robot_radius_m", "camera_height_m"):
        value = identity[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("invalid scene robot_profile " + key)
    if identity["mode"] == "kinematic_avatar":
        for key in ("profile_sha256", "assets_sha256", "profile_version", "base_offset_z_m"):
            if key not in identity:
                raise ValueError("scene robot_profile missing " + key)
    return deepcopy(identity)


def validate_scene_robot(scene, runtime_identity: dict, *, require_declared=False) -> None:
    """Reject body mismatches and prepared-navigation radius drift before execution."""
    source = _scene_version(scene)
    declared = scene_robot_identity(source)
    if declared is None:
        if require_declared:
            raise ValueError("official Stretch3 evaluation requires a declared scene robot_profile; prepare a new scene revision")
        return
    # Selection is a CLI label, not geometry: a relocated custom profile can
    # carry exactly the same physical identity. Official CLI separately pins it.
    semantic = lambda value: {key: val for key, val in value.items() if key != "selection"}
    if semantic(declared) != semantic(runtime_identity):
        raise ValueError("scene robot_profile does not match runtime robot identity")
    if not isinstance(source, dict):
        import numpy as np
        navigation_path = (source.resources["navigation"] if hasattr(source, "resources")
                           else source.path("navigation"))
        with np.load(navigation_path, allow_pickle=False) as navigation:
            radius = float(np.asarray(navigation["robot_radius"]).item())
            if "robot_profile_json" in navigation:
                encoded = json.loads(str(navigation["robot_profile_json"].item()))
                if encoded != declared:
                    raise ValueError("scene navigation robot_profile differs from scene manifest")
        if not math.isclose(radius, declared["robot_radius_m"], rel_tol=0, abs_tol=1e-9):
            raise ValueError("scene navigation robot_radius differs from robot_profile")


def validate_official_robot(selection: RobotSelection) -> None:
    if selection.name != DEFAULT_ROBOT or selection.identity() != default_robot_identity():
        raise ValueError("--official requires the bundled, unmodified Stretch3 profile " + OFFICIAL_EMBODIMENT_VERSION)


class KinematicRobotBody:
    """Follow avatar XY/yaw with a fixed-pose visual model and no rigid bodies."""

    def __init__(self, stage, config: RobotBodyConfig):
        from pxr import Usd, UsdGeom, UsdPhysics

        source = Usd.Stage.Open(str(config.asset))
        if not source or not source.GetDefaultPrim():
            raise ValueError("robot USD must have a default prim")
        if UsdGeom.GetStageUpAxis(source) != "Z" or not math.isclose(UsdGeom.GetStageMetersPerUnit(source), 1.0):
            raise ValueError("robot USD must use metres and Z up")
        self.config = config
        self.root = UsdGeom.Xform.Define(stage, "/World/LongTermBenchmark/Robot")
        model = stage.DefinePrim(str(self.root.GetPath()) + "/Model", "Xform")
        model.GetReferences().AddReference(str(config.asset))
        for prim in Usd.PrimRange(model):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.ArticulationRootAPI) or prim.IsA(UsdPhysics.Joint):
                raise ValueError("kinematic_avatar requires a visual USD without physics; use robot/scripts/convert_models.py")
            if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                raise ValueError("avatar USD must not add physics colliders")
            xform = UsdGeom.Xformable(prim)
            if xform and xform.GetResetXformStack():
                raise ValueError("avatar USD must inherit its parent transform")
        self.transform = UsdGeom.XformCommonAPI(self.root)

    def sync(self, camera_position, yaw: float) -> None:
        from pxr import Gf

        self.transform.SetTranslate(Gf.Vec3d(*self.config.body_position(camera_position)))
        self.transform.SetRotate(Gf.Vec3f(0, 0, math.degrees(yaw)))
