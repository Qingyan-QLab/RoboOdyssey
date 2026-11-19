"""Resolve navigation references from the actual task-start state."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
from typing import Any


def _world_cell(navigation: dict[str, Any], point: list[float]) -> tuple[int, int]:
    resolution = float(navigation["resolution"])
    origin = navigation["origin_xy"]
    return (int((float(point[0]) - float(origin[0])) / resolution),
            int((float(point[1]) - float(origin[1])) / resolution))


def _cell_world(navigation: dict[str, Any], cell: tuple[int, int]) -> list[float]:
    resolution = float(navigation["resolution"])
    origin = navigation["origin_xy"]
    return [float(origin[0]) + (cell[0] + 0.5) * resolution,
            float(origin[1]) + (cell[1] + 0.5) * resolution,
            float(navigation.get("floor_height", 0.0))]


def _nearest_free(blocked: Any, target: tuple[int, int]) -> tuple[int, int] | None:
    height, width = blocked.shape
    choices = ((math.hypot(x - target[0], y - target[1]), (x, y))
               for y in range(height) for x in range(width) if not blocked[y, x])
    return min(choices, default=(0.0, None))[1]


def _path(blocked: Any, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    height, width = blocked.shape
    if not (0 <= start[0] < width and 0 <= start[1] < height) or blocked[start[1], start[0]]:
        start = _nearest_free(blocked, start)
    if not (0 <= goal[0] < width and 0 <= goal[1] < height) or blocked[goal[1], goal[0]]:
        goal = _nearest_free(blocked, goal)
    if start is None or goal is None:
        return None
    queue = deque([start])
    parents = {start: None}
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (1, -1), (-1, 1), (-1, -1))
    while queue:
        current = queue.popleft()
        if current == goal:
            route = [current]
            while parents[route[-1]] is not None:
                route.append(parents[route[-1]])
            return list(reversed(route))
        for dx, dy in directions:
            nxt = current[0] + dx, current[1] + dy
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if blocked[nxt[1], nxt[0]] or nxt in parents:
                continue
            if dx and dy and (blocked[current[1], nxt[0]] or blocked[nxt[1], current[0]]):
                continue
            parents[nxt] = current
            queue.append(nxt)
    return None


def _object_center(state: dict[str, Any], object_id: str) -> list[float] | None:
    item = state.get("objects", {}).get(str(object_id), {})
    pose = item.get("pose", {}).get("position")
    if pose:
        return [float(pose[0]), float(pose[1])]
    bounds = item.get("geometry", {}).get("world_aabb", {})
    if bounds.get("min") and bounds.get("max"):
        return [(float(bounds["min"][0]) + float(bounds["max"][0])) / 2,
                (float(bounds["min"][1]) + float(bounds["max"][1])) / 2]
    return None


def resolve_runtime_reference(task: dict[str, Any], ground_truth: dict[str, Any], env: Any) -> dict[str, Any]:
    """Return private GT whose route begins at the agent's real task-start pose."""
    result = deepcopy(ground_truth)
    if task.get("task_type") != "navigation":
        return result
    navigation = getattr(env, "_navigation", None)
    static = getattr(env, "_static", None)
    dynamic = getattr(env, "_dynamic", None)
    if not isinstance(navigation, dict) or static is None or dynamic is None:
        return result
    blocked = (static != 0) | (dynamic != 0)
    start_position = env.get_agent_pose().get("position")
    if not start_position:
        return result
    start = _world_cell(navigation, start_position)
    outcome = result.setdefault("expected_outcome", {})
    target_ids = list(result.get("target", {}).get("object_ids", []))
    goals: list[list[float]] = []
    required_order = result.get("success_criteria", {}).get("required_site_order") or []
    if required_order:
        goals.extend(point for object_id in required_order
                     if (point := _object_center(env.runtime_state(), str(object_id))) is not None)
    elif task.get("task_family") == "relative_pose_navigation":
        offset = task.get("relative_goal", {}).get("position_offset", [0.0, 0.0])
        goals.append([float(start_position[0]) + float(offset[0]),
                      float(start_position[1]) + float(offset[1])])
    elif result.get('fixed_target_geometry', {}).get('world_aabb'):
        bounds = result['fixed_target_geometry']['world_aabb']
        goals.append([(bounds['min'][i] + bounds['max'][i]) / 2 for i in (0, 1)])
    elif target_ids:
        point = _object_center(env.runtime_state(), str(target_ids[0]))
        if point is not None:
            goals.append(point)
    else:
        goal = outcome.get("goal_pose", {}).get("position")
        if goal:
            goals.append([float(goal[0]), float(goal[1])])
    if not goals:
        return result
    route: list[tuple[int, int]] = []
    cursor = start
    for point in goals:
        segment = _path(blocked, cursor, _world_cell(navigation, point))
        if not segment:
            return result
        route.extend(segment if not route else segment[1:])
        cursor = segment[-1]
    points = [_cell_world(navigation, cell) for cell in route]
    length = sum(math.dist(a[:2], b[:2]) for a, b in zip(points, points[1:]))
    result["reference_trajectory"] = {
        "coordinate_frame": "world",
        "points": [{"position": point, "orientation": None} for point in points],
        "length_m": length,
        "planner": {"name": "runtime_grid_shortest_path", "version": 1,
                    "connectivity": 8, "task_start_state": "actual"},
    }
    if task.get("task_family") == "relative_pose_navigation":
        outcome["goal_pose"] = {"position": points[-1], "orientation": None}
    return result
