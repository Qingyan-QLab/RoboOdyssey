"""Read the immutable ``scene_package_v2`` benchmark dataset."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_instruction(value: Any, path: Path) -> None:
    """Validate the compact room-only instruction contract."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"instruction must be an object or null in {path}")
    if "views" in value or "camera_trajectory" in value:
        raise ValueError(f"instruction must not contain frozen camera views in {path}")
    rooms = value.get("rooms")
    if not isinstance(rooms, list):
        raise ValueError(f"instruction.rooms must be a list in {path}")
    for index, room in enumerate(rooms):
        if not isinstance(room, dict) or not room.get("room_id"):
            raise ValueError(f"instruction.rooms[{index}] requires room_id in {path}")
        if not isinstance(room.get("text"), str) or not room["text"].strip():
            raise ValueError(f"instruction.rooms[{index}] requires text in {path}")
        region = room.get("render_region")
        if not isinstance(region, dict) or region.get("type") != "room":
            raise ValueError(
                f"instruction.rooms[{index}].render_region.type must be 'room' in {path}"
            )


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
        return [(m21 - m12) / scale, (m02 - m20) / scale,
                (m10 - m01) / scale, 0.25 * scale]
    if m00 > m11 and m00 > m22:
        scale = (1.0 + m00 - m11 - m22) ** 0.5 * 2.0
        return [0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale,
                (m21 - m12) / scale]
    if m11 > m22:
        scale = (1.0 + m11 - m00 - m22) ** 0.5 * 2.0
        return [(m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale,
                (m02 - m20) / scale]
    scale = (1.0 + m22 - m00 - m11) ** 0.5 * 2.0
    return [(m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale,
            (m10 - m01) / scale]


def canonical_state_from_metadata(metadata: dict[str, Any], scene_id: str) -> dict[str, Any]:
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

    @property
    def session_id(self) -> str:
        return str(self.metadata["session_id"])

    @property
    def tasks(self) -> list[dict[str, Any]]:
        return list(self.metadata["tasks"])

    def read_json(self, name: str) -> dict[str, Any]:
        return _read_json(self.path / name)

    def ground_truth(self) -> dict[str, Any]:
        return self.read_json(str(self.metadata.get("ground_truth_file", "ground_truth.json")))

    def mutation(self) -> dict[str, Any]:
        name = self.metadata.get("mutation_file")
        return self.read_json(str(name)) if name else {"changes": []}


class BenchmarkDataset:
    """Dataset access with one supported, explicit package contract."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "dataset_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"dataset_manifest.json not found under {self.root}")
        self.manifest = _read_json(manifest_path)
        if int(self.manifest.get("schema_version", 0)) != 1:
            raise ValueError("dataset_manifest.json must use schema_version 1")
        scenes = self.manifest.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError("dataset manifest must contain at least one scene")
        self._scenes = {str(item["scene_id"]): item for item in scenes}
        if len(self._scenes) != len(scenes):
            raise ValueError("dataset manifest contains duplicate scene_id values")

    def scene_ids(self) -> list[str]:
        return list(self._scenes)

    def scene_entry(self, scene_id: str) -> dict[str, Any]:
        try:
            return self._scenes[scene_id]
        except KeyError as exc:
            raise KeyError(f"unknown scene_id: {scene_id}") from exc

    def scene_path(self, scene_id: str) -> Path:
        entry = self.scene_entry(scene_id)
        reference = entry.get("scene_usdz")
        if not reference:
            raise ValueError(f"scene {scene_id} has no scene_usdz reference")
        return _resolve_inside(self.root, str(reference)).parent

    def scene_asset(self, scene_id: str) -> Path:
        path = _resolve_inside(self.root, str(self.scene_entry(scene_id)["scene_usdz"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def scene_metadata(self, scene_id: str) -> dict[str, Any]:
        entry = self.scene_entry(scene_id)
        return _read_json(_resolve_inside(self.root, str(entry["scene_metainfo"])))

    def scene_build(self, scene_id: str) -> dict[str, Any]:
        entry = self.scene_entry(scene_id)
        return _read_json(_resolve_inside(self.root, str(entry["scene_build"])))

    def navigation_path(self, scene_id: str) -> Path:
        return _resolve_inside(self.root, str(self.scene_entry(scene_id)["navigation"]))

    def manipulation(self, scene_id: str) -> dict[str, Any]:
        entry = self.scene_entry(scene_id)
        return _read_json(_resolve_inside(self.root, str(entry["manipulation"])))

    def sessions(self, scene_id: str) -> list[SessionRecord]:
        entry = self.scene_entry(scene_id)
        directory = _resolve_inside(self.root, str(entry["sessions_dir"]))
        records = []
        for path in sorted(directory.glob("session_*")):
            metadata_path = path / "session.json"
            if metadata_path.is_file():
                metadata = _read_json(metadata_path)
                if metadata.get("dataset_schema") != "scene_package_v2":
                    raise ValueError(f"unsupported dataset_schema in {metadata_path}")
                _validate_instruction(metadata.get("instruction"), metadata_path)
                records.append(SessionRecord(scene_id, path, metadata))
        return records

    def canonical_state(self, scene_id: str) -> dict[str, Any]:
        return canonical_state_from_metadata(self.scene_metadata(scene_id), scene_id)
