"""Read published frozen scene collections and task sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_inside(root: Path, reference: str) -> Path:
    path = (root / reference).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"dataset reference escapes its root: {reference}") from exc
    return path


def _quaternion_from_matrix(matrix: list[list[float]]) -> list[float]:
    """Convert the metadata's row-vector rotation matrix to XYZW."""
    if len(matrix) < 3 or any(len(row) < 3 for row in matrix[:3]):
        return [0.0, 0.0, 0.0, 1.0]
    m00, m01, m02 = float(matrix[0][0]), float(matrix[1][0]), float(matrix[2][0])
    m10, m11, m12 = float(matrix[0][1]), float(matrix[1][1]), float(matrix[2][1])
    m20, m21, m22 = float(matrix[0][2]), float(matrix[1][2]), float(matrix[2][2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = (trace + 1.0) ** 0.5 * 2.0
        return [
            (m21 - m12) / scale,
            (m02 - m20) / scale,
            (m10 - m01) / scale,
            0.25 * scale,
        ]
    if m00 > m11 and m00 > m22:
        scale = (1.0 + m00 - m11 - m22) ** 0.5 * 2.0
        return [
            0.25 * scale,
            (m01 + m10) / scale,
            (m02 + m20) / scale,
            (m21 - m12) / scale,
        ]
    if m11 > m22:
        scale = (1.0 + m11 - m00 - m22) ** 0.5 * 2.0
        return [
            (m01 + m10) / scale,
            0.25 * scale,
            (m12 + m21) / scale,
            (m02 - m20) / scale,
        ]
    scale = (1.0 + m22 - m00 - m11) ** 0.5 * 2.0
    return [(m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale, (m10 - m01) / scale]


def canonical_state_from_metadata(
    metadata: dict[str, Any], scene_id: str
) -> dict[str, Any]:
    """Build the runtime state without writing a duplicate state file."""
    objects: dict[str, dict[str, Any]] = {}
    for raw in metadata.get("objects", []):
        object_id = str(raw.get("object_id", ""))
        if not object_id:
            continue
        entity = dict(raw)
        matrix = raw.get("geometry", {}).get("world_matrix4x4")
        if isinstance(matrix, list) and len(matrix) >= 4 and len(matrix[3]) >= 3:
            entity["pose"] = {
                "coordinate_frame": "world",
                "position": [float(value) for value in matrix[3][:3]],
                "orientation": _quaternion_from_matrix(matrix),
            }
        objects[object_id] = entity
    return {"schema_version": 1, "scene_id": scene_id, "objects": objects}


@dataclass(frozen=True)
class SessionRecord:
    scene_id: str
    path: Path
    metadata: dict[str, Any]
    private_paths: dict[str, Path]

    @property
    def session_id(self) -> str:
        return str(self.metadata["session_id"])

    @property
    def tasks(self) -> list[dict[str, Any]]:
        return list(self.metadata["tasks"])

    def ground_truth(self) -> list[dict[str, Any]]:
        return _read_json(self.private_paths["ground_truth"])


class BenchmarkDataset:
    """Read-only access to published scene assets and frozen requests."""

    def __init__(self, root: str | Path, *, sequence_id: str | None = None, scene=None):
        from ..data import load_sequence

        self._requests = load_sequence(root, sequence_id=sequence_id, scene=scene)
        self.root, self.manifest = self._requests.root, self._requests.manifest
        self._scenes = {self.manifest["scene_id"]: self.manifest}

    def scene_ids(self) -> list[str]:
        return list(self._scenes)

    def scene_entry(self, scene_id: str) -> dict[str, Any]:
        try:
            return self._scenes[scene_id]
        except KeyError as exc:
            raise KeyError(f"unknown scene_id: {scene_id}") from exc

    def scene_path(self, scene_id: str) -> Path:
        return self.scene_asset(scene_id).parent

    def scene_asset(self, scene_id: str) -> Path:
        self.scene_entry(scene_id)
        asset = self._requests.asset()
        if asset is None:
            raise ValueError("synthetic scene has no simulation asset")
        path = Path(asset["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def scene_metadata(self, scene_id: str) -> dict[str, Any]:
        self.scene_entry(scene_id)
        return _read_json(self._requests.path("scene_metainfo"))

    def scene_build(self, scene_id: str) -> dict[str, Any]:
        self.scene_entry(scene_id)
        return {
            "runtime_contract": self.manifest["runtime_contract"],
            "environment_setup": _read_json(self._requests.path("environment_setup")),
        }

    def navigation_path(self, scene_id: str) -> Path:
        self.scene_entry(scene_id)
        return self._requests.path("navigation")

    def manipulation(self, scene_id: str) -> dict[str, Any]:
        self.scene_entry(scene_id)
        return _read_json(self._requests.path("manipulation"))

    def sessions(self, scene_id: str) -> list[SessionRecord]:
        self.scene_entry(scene_id)
        records = []
        for entry in self.manifest["sessions"]:
            path = _resolve_inside(self.root, entry["session"])
            paths = {
                k: _resolve_inside(self.root, entry[k])
                for k in ("ground_truth", "templates", "mutations")
            }
            paths.update(
                runtime_scene_cache=self._requests.path("runtime_scene_cache"),
                dataset_root=self.root,
                plan=self._requests.path("plan"),
                scene_version_root=self._requests.version.directory,
            )
            metadata = _read_json(path)
            metadata["runtime_contract"] = self.manifest["runtime_contract"]
            metadata["sequence_id"] = self.manifest["sequence_id"]
            records.append(SessionRecord(scene_id, path.parent, metadata, paths))
        return records

    def canonical_state(self, scene_id: str) -> dict[str, Any]:
        state = canonical_state_from_metadata(self.scene_metadata(scene_id), scene_id)
        state["robot_base_pose"] = _read_json(
            self._requests.version.path("initial_state")
        )["robot_base_pose"]
        return state

    def sequence_identity(self):
        return {
            k: self.manifest[k] for k in ("scene_id", "sequence_id", "scene_reference")
        }
