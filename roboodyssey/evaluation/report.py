"""Shared denominator rules and complete legacy/v5 result aggregation."""

from __future__ import annotations

from collections import defaultdict, Counter
import json
from pathlib import Path
from typing import Any

from ..scenes.dataset_loader import BenchmarkDataset
from ..artifacts.layout import session_dir, manifest, write_json


def is_scored(result):
    # Old results and missing results retain their original scoring behavior.
    return result.get("scored", True) is not False


def task_totals(results):
    scored = [r for r in results if is_scored(r)]
    score = sum(bool(r.get("success")) for r in scored) / len(scored) if scored else None
    return {
        "tasks_total": len(results),
        "tasks_evaluated": len(scored),
        "tasks_excluded": len(results) - len(scored),
        "task_score": score,
        "success": all(bool(r.get("success")) for r in scored) if scored else None,
    }


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def write_summary(dataset: BenchmarkDataset, result_root: str | Path) -> dict[str, Any]:
    if dataset.manifest.get("schema_version") == 5:
        return write_request_summary(dataset, result_root)
    result_root = Path(result_root)
    sessions: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    all_tasks: list[dict[str, Any]] = []
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
            path = (
                result_root / "scenes" / scene_id / "sessions" / record.session_id / "metrics.json"
            )
            metrics = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.is_file()
                else {
                    "schema_version": 1,
                    "scene_id": scene_id,
                    "session_id": record.session_id,
                    "run_status": "missing",
                    "reset_count": 0,
                    "task_results": [],
                }
            )
            existing = {str(item.get("task_id")): item for item in metrics.get("task_results", [])}
            complete_results = []
            for task in record.tasks:
                task_id = str(task["task_id"])
                result = existing.get(task_id)
                if result is None:
                    result = {
                        "task_id": task_id,
                        "task_type": task["task_type"],
                        "task_family": task.get("task_family"),
                        "capabilities": task.get("capabilities", []),
                        "success": False,
                        "metrics": {"reason": "missing_result"},
                    }
                complete_results.append(result)
            metrics["task_results"] = complete_results
            metrics.update(task_totals(complete_results))
            score = metrics["task_score"]
            metrics["reset_aware_score"] = (
                score / (1 + int(metrics.get("reset_count", 0))) if score is not None else None
            )
            sessions.append(metrics)
            if metrics.get("run_status") in {"interrupted", "missing"}:
                interrupted += 1
            resets += int(metrics.get("reset_count", 0))
            authored = {str(task["task_id"]): task for task in record.tasks}
            private = {str(task["task_id"]): task for task in record.ground_truth()["tasks"]}
            for result in complete_results:
                all_tasks.append(result)
                if not is_scored(result):
                    continue
                success = bool(result.get("success"))
                tasks.append(result)
                by_type[str(result["task_type"])].append(success)
                for capability in result.get("capabilities", []):
                    by_capability[str(capability)].append(success)
                if "memory" in result.get("capabilities", []):
                    truth = private.get(str(result["task_id"]), {})
                    evidence = truth.get("evidence", {})
                    source = evidence.get("source_session_id") or truth.get(
                        "memory_evidence", {}
                    ).get("event", {}).get("session_id")
                    pair = authored.get(str(result["task_id"]), {}).get("paired_task") or truth.get(
                        "paired_task"
                    )
                    if source in session_index:
                        by_horizon[session_index[record.session_id] - session_index[source]].append(
                            success
                        )
                    elif pair:
                        pair = pair["session_id"]
                        if pair in session_index:
                            by_horizon[
                                abs(session_index[record.session_id] - session_index[pair])
                            ].append(success)
                    else:
                        target_ids = (
                            authored.get(str(result["task_id"]), {})
                            .get("target", {})
                            .get("object_ids", [])
                        )
                        previous = [
                            index
                            for object_id in target_ids
                            for index in object_sessions.get(str(object_id), [])
                            if index < current_index
                        ]
                        if previous:
                            by_horizon[current_index - max(previous)].append(success)
    summary = {
        "schema_version": 1,
        "dataset_id": dataset.manifest.get("dataset_id"),
        "sessions_evaluated": sum(s["tasks_evaluated"] > 0 for s in sessions),
        "sessions_total": len(sessions),
        "sessions_interrupted": interrupted,
        "session_success_rate": _rate(
            [bool(item["success"]) for item in sessions if item["tasks_evaluated"]]
        ),
        "tasks_total": len(all_tasks),
        "tasks_excluded": len(all_tasks) - len(tasks),
        "exclusions_by_reason": dict(
            Counter(t.get("reason_code", "unspecified") for t in all_tasks if not is_scored(t))
        ),
        "already_satisfied_tasks": sum(
            t.get("termination_reason") == "already_satisfied" for t in all_tasks
        ),
        "tasks_evaluated": len(tasks),
        "task_success_rate": _rate([bool(item["success"]) for item in tasks]),
        "task_type_success_rate": {key: _rate(values) for key, values in sorted(by_type.items())},
        "capability_success_rate": {
            key: _rate(values) for key, values in sorted(by_capability.items())
        },
        "memory_success_rate_by_horizon": {
            str(key): _rate(values) for key, values in sorted(by_horizon.items())
        },
        "reset_count": resets,
        "runtime_repair": {
            "repaired_tasks": sum(bool(t.get("instruction_repaired")) for t in all_tasks),
            "unavailable_tasks": sum(
                t.get("repair_status", "ready") not in {"ready", "already_satisfied"}
                for t in all_tasks
            ),
            "dependency_unsatisfied": sum(
                t.get("repair_status") == "dependency_unsatisfied" for t in all_tasks
            ),
            "unavailable_tasks_in_score_denominator": any(
                is_scored(t)
                for t in all_tasks
                if t.get("repair_status", "ready") not in {"ready", "already_satisfied"}
            ),
            "unavailable_by_reason": dict(
                Counter(
                    t.get("reason_code", t.get("repair_status"))
                    for t in all_tasks
                    if t.get("repair_status", "ready") not in {"ready", "already_satisfied"}
                )
            ),
        },
        "session_metrics": sessions,
    }
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def write_request_summary(dataset, result_root):
    from .requests import request_totals

    root = Path(result_root)
    sessions = []
    tasks = []
    errors = []
    groups = defaultdict(list)
    version = dataset.manifest["runtime_contract"]["version"]
    for scene_id in dataset.scene_ids():
        for record in dataset.sessions(scene_id):
            path = session_dir(root, scene_id, record.session_id) / "metrics.json"
            metrics = (
                json.loads(path.read_text())
                if path.exists()
                else {"run_status": "missing", "task_results": []}
            )
            actual = metrics.get("task_results", [])
            if (
                metrics.get("evaluation_valid") is False
                or metrics.get("run_status") != "completed"
                or [r.get("task_id") for r in actual] != [t["task_id"] for t in record.tasks]
            ):
                errors.append(
                    {
                        "scene_id": scene_id,
                        "session_id": record.session_id,
                        "reason": metrics.get("error", metrics.get("run_status")),
                    }
                )
            sessions.append(metrics)
            tasks.extend(actual)
            for item in actual:
                if item.get("scored", True):
                    for capability in item.get("capabilities", []):
                        groups[capability].append(item["success"] is True)
    totals = request_totals(tasks, valid=not errors) if version == 3 else task_totals(tasks)
    if errors:
        totals.update(evaluation_valid=False, task_score=None, success=None)
    value = {
        "schema_version": 1,
        "runtime_contract": dataset.manifest["runtime_contract"],
        **totals,
        "sessions_total": len(sessions),
        "sessions": sessions,
        "integration_errors": errors,
        "by_capability": {k: sum(v) / len(v) if not errors else None for k, v in groups.items()},
    }
    capability_scores = {}
    for capability in groups:
        subset = [
            t for t in tasks if t.get("scored", True) and capability in t.get("capabilities", [])
        ]
        capability_scores[capability] = request_totals(subset, valid=not errors).get("score_100")
    value.update(
        task_success_rate=totals.get("task_success_rate", totals["task_score"]),
        capability_score_100=capability_scores,
        capability_success_rate=value["by_capability"],
        session_metrics=sessions,
        sessions_evaluated=sum(s.get("tasks_evaluated", 0) > 0 for s in sessions),
        sessions_interrupted=len(errors),
        reset_count=sum(s.get("reset_count", 0) for s in sessions),
        session_success_rate=None
        if errors
        else sum(s.get("success") is True for s in sessions) / len(sessions)
        if sessions
        else None,
    )
    value["sequence_identity"] = dataset.sequence_identity()
    if manifest(root):
        for entry in manifest(root)["sequences"]:
            if "/" not in entry["sessions_root"]:
                continue  # Sequence child: its root summary is already this run's summary.
            subset = [s for s in sessions if s.get("scene_id") == entry["scene_id"]]
            write_json(
                root / entry["sessions_root"].rsplit("/", 1)[0] / "summary.json",
                {
                    "scene_id": entry["scene_id"],
                    "sequence_id": entry["sequence_id"],
                    **request_totals(
                        [r for s in subset for r in s["task_results"]], valid=not errors
                    ),
                },
            )
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return value
