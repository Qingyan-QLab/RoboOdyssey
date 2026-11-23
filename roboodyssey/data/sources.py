"""Independent Scene and Session inputs for benchmark execution."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from ..artifacts.layout import initialize_run, sha256_file
from ..contracts.navigation_io import read_navigation_npz
from ..scenes.dataset_loader import SessionRecord, canonical_state_from_metadata
from .archive import resource_root


SCENE_RESOURCES = (
    "scene_metainfo", "navigation", "manipulation", "environment_setup",
    "initial_state", "runtime_scene_cache",
)
SESSION_RESOURCES = ("ground_truth", "templates", "mutations")
RUNTIME_CONTRACT = {"name": "contextual_requests", "version": 3}


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resource_paths(manifest: Path, entries: dict[str, Any], required: Sequence[str]) -> dict[str, Path]:
    if not isinstance(entries, dict):
        raise ValueError(f"resources must be an object: {manifest}")
    missing = sorted(set(required) - set(entries))
    if missing:
        raise ValueError(f"missing resources in {manifest}: {', '.join(missing)}")
    result = {}
    for name, relative in entries.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"invalid resource path for {name}")
        path = (manifest.parent / relative).resolve()
        if not path.is_relative_to(manifest.parent):
            raise ValueError(f"resource path escapes its directory: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = path
    return result


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    asset: Path
    resources: dict[str, Path]
    manifest_path: Path


@dataclass(frozen=True)
class SessionSpec:
    scene_id: str
    sequence_id: str
    session_id: str
    path: Path
    resources: dict[str, Path]
    metadata: dict[str, Any]
    manifest_path: Path


def load_scene(source: str | Path) -> SceneSpec:
    """Load a standalone Scene directory or tar with scene_resources.json."""
    root = resource_root(source, "scene_resources.json")
    manifest = root / "scene_resources.json"
    data = _object(manifest)
    if data.get("schema_version") != 1 or data.get("kind") != "roboodyssey_scene":
        raise ValueError("unsupported Scene resource manifest")
    asset_name = data.get("asset")
    if not isinstance(asset_name, str) or not asset_name:
        raise ValueError("Scene manifest requires asset")
    asset = (root / asset_name).resolve()
    if not asset.is_relative_to(root):
        raise ValueError("Scene asset path escapes its directory")
    if asset.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
        raise ValueError("Scene asset must be a USD, USDA, USDC or USDZ file")
    if not asset.is_file():
        raise FileNotFoundError(asset)
    scene_id = data.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("Scene requires scene_id")
    expected = data.get("asset_sha256")
    if not isinstance(expected, str) or sha256_file(asset) != expected:
        raise ValueError("Scene asset SHA-256 mismatch")
    paths = _resource_paths(manifest, data.get("resources"), SCENE_RESOURCES)
    hashes = data.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(paths):
        raise ValueError("Scene manifest must hash every resource")
    for name, digest in hashes.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"Scene resource SHA-256 mismatch: {name}")
    metadata = _object(paths["scene_metainfo"])
    if metadata.get("scene", {}).get("scene_id") != scene_id:
        raise ValueError("Scene metadata identity mismatch")
    cache = _object(paths["runtime_scene_cache"])
    if cache.get("scene_id") != scene_id or set(cache.get("objects", {})) != {
        item["object_id"] for item in metadata.get("objects", [])
    }:
        raise ValueError("Scene runtime cache identity or objects differ from metadata")
    setup = json.loads(paths["environment_setup"].read_text(encoding="utf-8"))
    navigation = read_navigation_npz(paths["navigation"])
    if navigation.get("environment_setup") and navigation["environment_setup"] != setup:
        raise ValueError("Scene navigation and environment setup differ")
    if "robot_base_pose" not in _object(paths["initial_state"]):
        raise ValueError("Scene initial state requires robot_base_pose")
    return SceneSpec(scene_id, asset, paths, manifest)


def load_session(source: str | Path) -> SessionSpec:
    """Load one Session directory or tar with session_resources.json."""
    root = resource_root(source, "session_resources.json")
    manifest = root / "session_resources.json"
    data = _object(manifest)
    session_name = data.get("session", "session.json")
    if not isinstance(session_name, str) or not session_name:
        raise ValueError("Session manifest requires session path")
    path = (root / session_name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Session path escapes its directory")
    metadata = _object(path)
    if data.get("schema_version") != 1 or data.get("kind") != "roboodyssey_session":
        raise ValueError("unsupported Session resource manifest")
    if sha256_file(path) != data.get("session_sha256"):
        raise ValueError("Session file SHA-256 mismatch")
    for name in ("scene_id", "sequence_id", "session_id"):
        if not isinstance(data.get(name), str) or not data[name]:
            raise ValueError(f"Session manifest requires {name}")
        if name in {"scene_id", "session_id"} and metadata.get(name) != data[name]:
            raise ValueError(f"Session {name} differs from resource manifest")
    tasks = metadata.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Session requires a nonempty tasks list")
    task_ids = [task.get("task_id") for task in tasks]
    if any(not isinstance(task_id, str) or not task_id for task_id in task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("Session task IDs must be unique nonempty strings")
    if "context" not in metadata or "time_context" not in metadata:
        raise ValueError("Session requires context and time_context")
    contract = data.get("runtime_contract", RUNTIME_CONTRACT)
    if contract != RUNTIME_CONTRACT:
        raise ValueError("unsupported Session runtime contract")
    paths = _resource_paths(manifest, data.get("resources"), SESSION_RESOURCES)
    hashes = data.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(paths):
        raise ValueError("Session manifest must hash every resource")
    for name, digest in hashes.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"Session resource SHA-256 mismatch: {name}")
    for name in SESSION_RESOURCES:
        values = json.loads(paths[name].read_text(encoding="utf-8"))
        if not isinstance(values, list) or (name != "mutations" and len(values) != len(tasks)):
            raise ValueError(f"Session {name} does not match task count")
        if name != "mutations" and [entry.get("task_id") for entry in values] != task_ids:
            raise ValueError(f"Session {name} task order differs from public tasks")
    metadata = dict(metadata, sequence_id=data["sequence_id"], runtime_contract=contract)
    return SessionSpec(data["scene_id"], data["sequence_id"], data["session_id"], path, paths, metadata, manifest)


class _ExternalResources:
    def __init__(self, scene: SceneSpec):
        self.scene = scene

    def path(self, name: str) -> Path:
        return self.scene.resources[name]


class ExternalDataset:
    """Adapter from independent Scene/Session inputs to the current runner."""

    def __init__(self, scene: SceneSpec, sessions: Sequence[SessionSpec]):
        if not sessions:
            raise ValueError("at least one Session is required")
        if any(s.scene_id != scene.scene_id for s in sessions):
            raise ValueError("Scene and Session identities differ")
        if len({s.sequence_id for s in sessions}) != 1:
            raise ValueError("Sessions in one run must share sequence_id")
        if len({s.session_id for s in sessions}) != len(sessions):
            raise ValueError("duplicate Session ID")
        self.scene = scene
        self._sessions = tuple(sessions)
        self.root = scene.manifest_path.parent
        self.manifest = {"schema_version": 1, "runtime_contract": RUNTIME_CONTRACT,
                         "sequence_id": sessions[0].sequence_id, "scene_id": scene.scene_id}
        self._requests = _ExternalResources(scene)

    def scene_ids(self) -> list[str]:
        return [self.scene.scene_id]

    def scene_asset(self, scene_id: str) -> Path:
        self._check_scene(scene_id)
        return self.scene.asset

    def _check_scene(self, scene_id: str) -> None:
        if scene_id != self.scene.scene_id:
            raise KeyError(f"unknown scene_id: {scene_id}")

    def scene_metadata(self, scene_id: str) -> dict[str, Any]:
        self._check_scene(scene_id)
        return _object(self.scene.resources["scene_metainfo"])

    def navigation_path(self, scene_id: str) -> Path:
        self._check_scene(scene_id)
        return self.scene.resources["navigation"]

    def manipulation(self, scene_id: str) -> dict[str, Any]:
        self._check_scene(scene_id)
        return _object(self.scene.resources["manipulation"])

    def canonical_state(self, scene_id: str) -> dict[str, Any]:
        state = canonical_state_from_metadata(self.scene_metadata(scene_id), scene_id)
        state["robot_base_pose"] = _object(self.scene.resources["initial_state"])["robot_base_pose"]
        return state

    def _mutation_context(self) -> Any:
        metadata = self.scene_metadata(self.scene.scene_id)
        navigation = read_navigation_npz(self.scene.resources["navigation"])
        manipulation = self.manipulation(self.scene.scene_id)
        picks = {c["object_id"] for c in manipulation.get("pick_place_candidates", [])}
        return SimpleNamespace(metadata=metadata, engine=SimpleNamespace(
            rooms=metadata["rooms"], navigation=navigation, pick_ids=picks))

    def sessions(self, scene_id: str) -> list[SessionRecord]:
        self._check_scene(scene_id)
        context = self._mutation_context()
        return [SessionRecord(scene_id, s.path.parent, s.metadata,
                              {**s.resources, "runtime_scene_cache": self.scene.resources["runtime_scene_cache"]},
                              runtime_context=context) for s in self._sessions]

    def sequence_identity(self) -> dict[str, Any]:
        return {"scene_id": self.scene.scene_id, "sequence_id": self._sessions[0].sequence_id,
                "scene_reference": {"kind": "external", "sha256": sha256_file(self.scene.asset)}}


class BenchmarkSuite:
    """A Scene and ordered Sessions ready for interaction and evaluation."""

    def __init__(self, scene: SceneSpec, sessions: Sequence[SessionSpec]):
        self.dataset = ExternalDataset(scene, sessions)

    @classmethod
    def from_published(cls, path: str | Path, *, sequence_id: str | None = None) -> "BenchmarkSuite":
        """Use the same interaction/evaluation interface for a frozen archive."""
        suite = cls.__new__(cls)
        suite.dataset = load_published_dataset(path, sequence_id=sequence_id)
        return suite

    def validate(self, capability: str = "interaction") -> None:
        if capability not in {"interaction", "evaluation", "synthesis"}:
            raise ValueError(f"unknown capability: {capability}")
        if capability == "synthesis":
            return
        for scene_id in self.dataset.scene_ids():
            self.dataset.sessions(scene_id)

    def interact(self, agent: Any, output: str | Path,
                 environment_factory: Callable[..., Any], *, max_steps: int = 2000) -> Path:
        """Run through the shared Session runner with an injected environment."""
        from ..runtime.session_runner import SessionRunner

        self.validate("interaction")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        initialize_run(output, self.dataset)
        scene_id = self.dataset.scene_ids()[0]
        SessionRunner(output, agent, max_steps=max_steps).run_scene(
            scene_id, self.dataset.sessions(scene_id), self.dataset.canonical_state(scene_id),
            environment_factory)
        return output

    def evaluate(self, output: str | Path) -> dict[str, Any]:
        """Aggregate the evidence recorded by an interaction run."""
        from ..evaluation.report import write_summary

        self.validate("evaluation")
        return write_summary(self.dataset, output)


def load_published_dataset(path: str | Path, *, sequence_id: str | None = None):
    """Preserve the published archive loader behind a named public API."""
    from ..scenes.dataset_loader import BenchmarkDataset

    return BenchmarkDataset(path, sequence_id=sequence_id)
