#!/usr/bin/env python3
"""Validate a CL_session_builder ``scene_package_v2`` dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ABSOLUTE = re.compile(r"(?:/media/|/home/|/mnt/|[A-Za-z]:[\\/])")
TASK_TYPES = {"navigation", "manipulation", "qa"}
TASK_FAMILIES = {
    "object_transfer": "manipulation", "inspect_then_act": "manipulation",
    "facility_control": "manipulation", "multi_site_workflow": "navigation",
    "navigation_qa": "qa", "named_place_revisit": "navigation",
    "spatial_relation_qa": "qa", "point_navigation": "navigation",
    "relative_pose_navigation": "navigation", "object_navigation": "navigation",
    "semantic_navigation": "navigation",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        return [f"missing {manifest_path}"]
    try:
        manifest = _json(manifest_path)
    except Exception as exc:  # noqa: BLE001
        return [f"invalid dataset_manifest.json: {exc}"]
    if manifest.get("schema_version") != 1:
        errors.append("dataset_manifest.json must use schema_version 1")
    scenes = manifest.get("scenes", [])
    if not scenes:
        return errors + ["manifest contains no scenes"]
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".yaml", ".usda"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
            if ABSOLUTE.search(text):
                errors.append(f"absolute path in {path.relative_to(root)}")
            if path.suffix.lower() == ".json":
                json.loads(text)
            elif path.suffix.lower() == ".jsonl":
                for line in text.splitlines():
                    if line.strip():
                        json.loads(line)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"invalid {path.relative_to(root)}: {exc}")
    for entry in scenes:
        scene_id = str(entry.get("scene_id"))
        required_keys = ("scene_usdz", "scene_metainfo", "sessions_dir", "scene_build", "navigation", "manipulation")
        for key in required_keys:
            if not entry.get(key):
                errors.append(f"{scene_id}: manifest missing {key}")
        paths = {key: root / str(entry[key]) for key in required_keys if entry.get(key)}
        for key, path in paths.items():
            if not path.exists():
                errors.append(f"{scene_id}: missing {key}: {path.relative_to(root)}")
        if not all(path.exists() for path in paths.values()):
            continue
        try:
            metadata = _json(paths["scene_metainfo"])
            object_ids = {str(item.get("object_id")) for item in metadata.get("objects", [])}
            if len(object_ids) != len(metadata.get("objects", [])):
                errors.append(f"{scene_id}: duplicate or missing object_id")
            for item in metadata.get("objects", []):
                if not item.get("prim_path"):
                    errors.append(f"{scene_id}: object lacks prim_path: {item.get('object_id')}")
            build = _json(paths["scene_build"])
            for section, path in (("scene_metainfo", paths["scene_metainfo"]), ("scene_usdz", paths["scene_usdz"])):
                expected = build.get("inputs", {}).get(section, {}).get("sha256")
                if expected and expected != _sha256(path):
                    errors.append(f"{scene_id}: {section} SHA-256 mismatch")
            for section, path in (("navigation", paths["navigation"]), ("manipulation", paths["manipulation"])):
                expected = build.get("derived", {}).get(section, {}).get("sha256")
                if expected and expected != _sha256(path):
                    errors.append(f"{scene_id}: {section} SHA-256 mismatch")
            sessions_dir = paths["sessions_dir"]
            sessions = sorted(sessions_dir.glob("session_*"))
            if not sessions:
                errors.append(f"{scene_id}: no sessions")
            parent = None
            session_data = {}
            for index, session_dir in enumerate(sessions):
                session_path = session_dir / "session.json"
                gt_path = session_dir / "ground_truth.json"
                if not session_path.exists() or not gt_path.exists():
                    errors.append(f"{session_dir.relative_to(root)}: missing session.json or ground_truth.json")
                    continue
                session = _json(session_path)
                gt = _json(gt_path)
                session_data[str(session.get("session_id"))] = session
                if session.get("dataset_schema") != "scene_package_v2":
                    errors.append(f"{session_path.relative_to(root)}: unsupported dataset_schema")
                if session.get("parent_session_id") != parent:
                    errors.append(f"{session_path.relative_to(root)}: invalid parent_session_id")
                tasks = session.get("tasks", [])
                gt_tasks = gt.get("tasks", [])
                if {task.get("task_id") for task in tasks} != {task.get("task_id") for task in gt_tasks}:
                    errors.append(f"{session_path.relative_to(root)}: session/GT task IDs differ")
                if index == 0 and session.get("mutation_file"):
                    errors.append(f"{session_path.relative_to(root)}: session_000 cannot have mutation")
                counts = {kind: sum(task.get("task_type") == kind for task in tasks) for kind in TASK_TYPES}
                for kind, count in counts.items():
                    if count < 2:
                        errors.append(f"{session_path.relative_to(root)}: requires at least two {kind} tasks")
                for task in tasks:
                    task_type = task.get("task_type")
                    family = task.get("task_family")
                    if task_type not in TASK_TYPES:
                        errors.append(f"{session_path.relative_to(root)}: unsupported task_type {task_type!r}")
                    if TASK_FAMILIES.get(family) != task_type:
                        errors.append(f"{session_path.relative_to(root)}: task_family/type mismatch")
                    unknown_objects = set(task.get("target", {}).get("object_ids", [])) - object_ids
                    if unknown_objects:
                        errors.append(f"{session_path.relative_to(root)}: unknown target objects {sorted(unknown_objects)}")
                    text = task.get("instruction", task.get("question", ""))
                    if not isinstance(text, str) or not text or any(ord(char) > 127 for char in text):
                        errors.append(f"{session_path.relative_to(root)}: task text must be ASCII English")
                    positions = {item.get("task_id"): position for position, item in enumerate(tasks)}
                    if any(dep not in positions or positions[dep] >= positions[task.get("task_id")] for dep in task.get("depends_on", [])):
                        errors.append(f"{session_path.relative_to(root)}: invalid task dependency")
                mutation_name = session.get("mutation_file")
                if mutation_name:
                    mutation_path = session_dir / mutation_name
                    if not mutation_path.is_file():
                        errors.append(f"{session_path.relative_to(root)}: missing mutation file")
                        parent = session.get("session_id")
                        continue
                    mutation = _json(mutation_path)
                    if mutation.get("visible_to_agent") is not False:
                        errors.append(f"{session_path.relative_to(root)}: mutation must be hidden")
                    for change in mutation.get("changes", []):
                        if change.get("object_id") not in object_ids:
                            errors.append(f"{session_path.relative_to(root)}: mutation references unknown object")
                        for key in ("pose_before", "pose_after", "relation_before", "relation_after"):
                            if not change.get(key):
                                errors.append(f"{session_path.relative_to(root)}: mutation lacks {key}")
                for private in gt_tasks:
                    if not private.get("expected_outcome") or not private.get("success_criteria"):
                        errors.append(f"{gt_path.relative_to(root)}: incomplete GT for {private.get('task_id')}")
                parent = session.get("session_id")
            for owner_id, session in session_data.items():
                for task in session.get("tasks", []):
                    if task.get("task_family") != "named_place_revisit":
                        continue
                    pair = task.get("paired_task", {})
                    paired = next((item for item in session_data.get(str(pair.get("session_id")), {}).get("tasks", [])
                                   if item.get("task_id") == pair.get("task_id")), None)
                    expected = {"session_id": owner_id, "task_id": task.get("task_id")}
                    if not paired or paired.get("paired_task") != expected:
                        errors.append(f"{scene_id}: named-place pair is not reciprocal")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{scene_id}: validation error: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    errors = validate(args.dataset.resolve())
    if errors:
        print("Dataset validation failed:\n" + "\n".join(f"- {error}" for error in errors))
        return 1
    print(f"Dataset validation passed: {args.dataset.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
