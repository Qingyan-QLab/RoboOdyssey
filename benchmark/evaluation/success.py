"""Deterministic task success checks for ``scene_package_v2``."""
from __future__ import annotations

import math
from typing import Any

from .metrics.qa import score_qa


def _distance(a: Any, b: Any) -> float:
    if a is None or b is None:
        return float("inf")
    return math.dist([float(value) for value in a], [float(value) for value in b])


def _distance_to_aabb_xy(point: Any, geometry: dict[str, Any] | None) -> float:
    """Return the shortest 2D distance from a point to an axis-aligned box."""
    if point is None or not isinstance(geometry, dict):
        return float("inf")
    bounds = geometry.get("world_aabb_xy")
    if not isinstance(bounds, dict):
        return float("inf")
    minimum, maximum = bounds.get("min"), bounds.get("max")
    if not minimum or not maximum or len(point) < 2:
        return float("inf")
    dx = max(float(minimum[0]) - float(point[0]), 0.0, float(point[0]) - float(maximum[0]))
    dy = max(float(minimum[1]) - float(point[1]), 0.0, float(point[1]) - float(maximum[1]))
    return math.hypot(dx, dy)


def _runtime_geometry(object_id: str | None, state: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve an object's current AABB without changing scene metainfo."""
    if not object_id:
        return None
    entity = state.get("objects", {}).get(str(object_id), {})
    bounds = entity.get("geometry", {}).get("world_aabb", {})
    minimum, maximum = bounds.get("min"), bounds.get("max")
    pose = entity.get("pose", {}).get("position")
    matrix = entity.get("geometry", {}).get("world_matrix4x4")
    if not minimum or not maximum:
        return None
    if pose is None or not isinstance(matrix, list) or len(matrix) < 4:
        return {"world_aabb_xy": {"min": list(minimum[:2]), "max": list(maximum[:2])}}
    base = [float(value) for value in matrix[3][:3]]
    delta = [float(pose[i]) - base[i] for i in range(3)]
    return {"world_aabb_xy": {"min": [float(minimum[0]) + delta[0], float(minimum[1]) + delta[1]],
                              "max": [float(maximum[0]) + delta[0], float(maximum[1]) + delta[1]]}}


def evaluate_embodied_task(
    task: dict[str, Any], ground_truth: dict[str, Any], events: list[dict[str, Any]],
    state: dict[str, Any],
) -> dict[str, Any]:
    kind = task["task_type"]
    criteria = ground_truth.get("success_criteria", {})
    if kind == "navigation":
        goal = ground_truth.get("expected_outcome", {}).get("goal_pose", {}).get("position")
        actual = events[-1].get("after", {}).get("position") if events else None
        actual_xy = actual[:2] if actual else None
        target_geometry = ground_truth.get("expected_outcome", {}).get("target_geometry")
        target_ids = ground_truth.get("target", {}).get("object_ids", [])
        runtime_target = _runtime_geometry(target_ids[0] if target_ids else None, state)
        if runtime_target is not None:
            target_geometry = runtime_target
        distance_type = str(criteria.get("target_distance_type", "point"))
        if distance_type == "aabb_2d" and isinstance(target_geometry, dict):
            distance = _distance_to_aabb_xy(actual_xy, target_geometry)
        else:
            distance = _distance(actual_xy, goal[:2] if goal else None)
        tolerance = float(criteria.get("position_tolerance_m", 0.8))
        collisions = sum(bool(event.get("result", {}).get("collision")) for event in events)
        collision_ok = bool(criteria.get("collision_allowed", False)) or collisions == 0
        reference = ground_truth.get("reference_trajectory", {})
        path_length = sum(
            _distance(before[:2], after[:2])
            for event in events
            if (before := event.get("before", {}).get("position"))
            and (after := event.get("after", {}).get("position"))
        )
        shortest = float(reference.get("length_m", 0.0))
        required_order = criteria.get("required_site_order") or []
        visited = []
        if required_order:
            geometries = ground_truth.get("expected_outcome", {}).get("site_geometries", {})
            for event in events:
                point = (event.get("after") or {}).get("position")
                point_xy = point[:2] if point else None
                for object_id in required_order:
                    if object_id in visited:
                        continue
                    geometry = geometries.get(str(object_id))
                    geometry = _runtime_geometry(object_id, state) or geometry
                    site_distance = (_distance_to_aabb_xy(point_xy, geometry)
                                     if geometry else float("inf"))
                    if site_distance <= tolerance:
                        visited.append(object_id)
                        break
        order_ok = not required_order or visited == list(required_order)
        success = distance <= tolerance and collision_ok and order_ok
        return {"success": success, "final_distance_m": distance, "target_distance_type": distance_type,
                "tolerance_m": tolerance,
                "collisions": collisions, "path_length_m": path_length,
                "reference_length_m": shortest,
                "visited_sites": visited, "required_site_order": list(required_order),
                "completion_reason": "target_reached" if success else "target_not_reached",
                "spl": shortest / max(shortest, path_length, 1e-9) if success else 0.0,
                "steps": len(events)}
    if kind == "manipulation":
        expected = ground_truth.get("expected_outcome", {}).get("state_changes", [])
        objects = state.get("objects", {})
        checks = []
        for change in expected:
            actual = objects.get(str(change.get("object_id")), {})
            relation = change.get("relation_after") or {}
            relation_type = relation.get("type") or relation.get("relation_type")
            if relation_type == "in_room":
                passed = actual.get("room_id") == relation.get("room_id")
            elif relation_type in {"on", "supported_by"}:
                passed = actual.get("support_object_id") == relation.get("object_id")
            else:
                passed = not relation
            if change.get("state_after") is not None:
                expected_state = change["state_after"]
                if expected_state in {"open", "closed"}:
                    passed = passed and bool(actual.get("door_open")) == (expected_state == "open")
                else:
                    passed = passed and actual.get("state") == expected_state
            checks.append({"object_id": change.get("object_id"), "success": passed})
        return {"success": bool(checks) and all(item["success"] for item in checks),
                "checks": checks, "steps": len(events)}
    raise ValueError(f"not an embodied task: {kind}")


def evaluate_qa(prediction: Any, ground_truth: dict[str, Any]) -> dict[str, Any]:
    outcome = ground_truth.get("expected_outcome", {})
    metrics = score_qa(prediction, outcome.get("answer", ""), outcome.get("accepted_answers", []))
    metrics["success"] = bool(metrics["semantic_match"])
    return metrics
