"""Aggregate task, capability, session, and memory-horizon metrics."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from ..scenes.dataset_loader import BenchmarkDataset


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def write_summary(dataset: BenchmarkDataset, result_root: str | Path) -> dict[str, Any]:
    result_root = Path(result_root)
    sessions: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    by_type: dict[str, list[bool]] = defaultdict(list)
    by_capability: dict[str, list[bool]] = defaultdict(list)
    by_horizon: dict[int, list[bool]] = defaultdict(list)
    resets = 0
    interrupted = 0
    for scene_id in dataset.scene_ids():
        records = dataset.sessions(scene_id)
        session_index = {record.session_id: index for index, record in enumerate(records)}
        object_sessions: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            for task in record.tasks:
                for object_id in task.get("target", {}).get("object_ids", []):
                    object_sessions[str(object_id)].append(index)
        for current_index, record in enumerate(records):
            path = result_root / "scenes" / scene_id / "sessions" / record.session_id / "metrics.json"
            if not path.is_file():
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            sessions.append(metrics)
            if metrics.get("run_status") == "interrupted":
                interrupted += 1
            resets += int(metrics.get("reset_count", 0))
            authored = {str(task["task_id"]): task for task in record.tasks}
            private = {str(task["task_id"]): task for task in record.ground_truth()["tasks"]}
            for result in metrics.get("task_results", []):
                success = bool(result.get("success"))
                tasks.append(result)
                by_type[str(result["task_type"])].append(success)
                for capability in result.get("capabilities", []):
                    by_capability[str(capability)].append(success)
                if "memory" in result.get("capabilities", []):
                    evidence = private.get(str(result["task_id"]), {}).get("evidence", {})
                    source = evidence.get("source_session_id")
                    if source in session_index:
                        by_horizon[session_index[record.session_id] - session_index[source]].append(success)
                    elif authored.get(str(result["task_id"]), {}).get("paired_task"):
                        pair = authored[str(result["task_id"])]["paired_task"]["session_id"]
                        if pair in session_index:
                            by_horizon[abs(session_index[record.session_id] - session_index[pair])].append(success)
                    else:
                        target_ids = authored.get(str(result["task_id"]), {}).get("target", {}).get("object_ids", [])
                        previous = [index for object_id in target_ids
                                    for index in object_sessions.get(str(object_id), []) if index < current_index]
                        if previous:
                            by_horizon[current_index - max(previous)].append(success)
    summary = {
        "schema_version": 1,
        "dataset_id": dataset.manifest.get("dataset_id"),
        "sessions_evaluated": len(sessions),
        "sessions_interrupted": interrupted,
        "session_success_rate": _rate([bool(item["success"]) for item in sessions]),
        "tasks_evaluated": len(tasks),
        "task_success_rate": _rate([bool(item["success"]) for item in tasks]),
        "task_type_success_rate": {key: _rate(values) for key, values in sorted(by_type.items())},
        "capability_success_rate": {key: _rate(values) for key, values in sorted(by_capability.items())},
        "memory_success_rate_by_horizon": {str(key): _rate(values) for key, values in sorted(by_horizon.items())},
        "reset_count": resets,
        "session_metrics": sessions,
    }
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
