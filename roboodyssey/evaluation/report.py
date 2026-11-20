"""Complete request-result aggregation with explicit validity checks."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from ..artifacts.layout import session_dir, manifest, write_json


def write_summary(dataset, result_root):
    from .requests import request_totals

    root = Path(result_root)
    sessions = []
    tasks = []
    errors = []
    groups = defaultdict(list)
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
    totals = request_totals(tasks, valid=not errors)
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
