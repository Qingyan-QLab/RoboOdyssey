"""Navigation cells, approach routes and connectivity validation. See LICENSE."""

from __future__ import annotations

from collections import deque
import math
import re


_PLACEHOLDER_DESCRIPTION = re.compile(r"^unknown(?:\s*\(|$)", re.IGNORECASE)


def entity_label(entity: dict, fallback: str = "object") -> str:
    """Return a safe display label, preferring description over category."""
    description = str(entity.get("description") or "").strip()
    if description and not _PLACEHOLDER_DESCRIPTION.match(description):
        return description.replace("_", " ")
    category = str(entity.get("category") or "").strip()
    if category:
        return category.replace("_", " ")
    return fallback


def _parents_from_goal(
    grid: dict, goal: tuple[int, int]
) -> dict[tuple[int, int], tuple[int, int] | None]:
    blocked = [
        [bool(a or b) for a, b in zip(row_a, row_b)]
        for row_a, row_b in zip(grid["static_occupancy"], grid["dynamic_occupancy"])
    ]
    height, width = len(blocked), len(blocked[0])
    if blocked[goal[1]][goal[0]]:
        return {}
    queue, parent = deque([goal]), {goal: None}
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    while queue:
        current = queue.popleft()
        for dx, dy in directions:
            nxt = (current[0] + dx, current[1] + dy)
            if (
                0 <= nxt[0] < width
                and 0 <= nxt[1] < height
                and not blocked[nxt[1]][nxt[0]]
                and nxt not in parent
                and (
                    not (dx and dy)
                    or (not blocked[current[1]][nxt[0]] and not blocked[nxt[1]][current[0]])
                )
            ):
                parent[nxt] = current
                queue.append(nxt)
    return parent


def _path_from_parents(parents: dict, start: tuple[int, int]) -> list[tuple[int, int]] | None:
    if start not in parents:
        return None
    route = [start]
    while parents[route[-1]] is not None:
        route.append(parents[route[-1]])
    return route


def cell_world(grid: dict, cell: tuple[int, int]) -> list[float]:
    return [
        grid["origin_xy"][0] + (cell[0] + 0.5) * grid["resolution"],
        grid["origin_xy"][1] + (cell[1] + 0.5) * grid["resolution"],
        grid["floor_height"],
    ]


def nearest_free(
    grid: dict, point: tuple[float, float], room_id: str | None = None
) -> tuple[int, int] | None:
    room_index = None
    if room_id and room_id in grid["room_ids"]:
        room_index = grid["room_ids"].index(room_id) + 1
    choices = []
    for y, row in enumerate(grid["static_occupancy"]):
        for x, static in enumerate(row):
            if (
                static
                or grid["dynamic_occupancy"][y][x]
                or (room_index is not None and grid["room_labels"][y][x] != room_index)
            ):
                continue
            world = cell_world(grid, (x, y))
            choices.append((math.dist(world[:2], point), (x, y)))
    return min(choices)[1] if choices else None


def _object_center(entity: dict) -> tuple[float, float]:
    aabb = entity.get("geometry", {}).get("world_aabb", {})
    return ((aabb["min"][0] + aabb["max"][0]) / 2, (aabb["min"][1] + aabb["max"][1]) / 2)


def _target_geometry(entity: dict) -> dict | None:
    """Return the target object's 2D AABB for distance-based evaluation."""
    aabb = entity.get("geometry", {}).get("world_aabb") or entity.get("world_aabb")
    if not aabb or not aabb.get("min") or not aabb.get("max"):
        return None
    return {
        "type": "aabb_2d",
        "world_aabb_xy": {
            "min": [float(aabb["min"][0]), float(aabb["min"][1])],
            "max": [float(aabb["max"][0]), float(aabb["max"][1])],
        },
        "source": "scene_metainfo.geometry.world_aabb",
    }


def navigation_task(
    entity: dict,
    grid: dict,
    index: int,
    *,
    task_family: str = "object_navigation",
    room: dict | None = None,
    start: tuple[int, int] | None = None,
) -> tuple[dict, dict] | None:
    goal = nearest_free(grid, _object_center(entity), str(entity.get("room_id")))
    if goal is None:
        return None
    parents = _parents_from_goal(grid, goal)
    if start is None:
        for y, row in enumerate(grid["static_occupancy"]):
            for x, static in enumerate(row):
                if static or grid["dynamic_occupancy"][y][x] or (x, y) == goal:
                    continue
                candidate = _path_from_parents(parents, (x, y))
                if candidate and len(candidate) > 2:
                    start, route = (x, y), candidate
                    break
            if start is not None:
                break
        if start is None:
            return None
    else:
        route = _path_from_parents(parents, start)
        if not route:
            return None
    task_id = f"navigation_{index:03d}"
    target = {"object_ids": [entity["object_id"]], "room_ids": [entity.get("room_id")]}
    description = entity_label(entity)
    room_label = str(
        (room or {}).get("room_type") or entity.get("room_id") or "target area"
    ).replace("_", " ")
    instruction = f"Go to the {description}."
    if task_family == "semantic_navigation":
        instruction = f"Go to the {room_label}."
        target["object_ids"] = []
    elif task_family == "point_navigation":
        world_goal = cell_world(grid, goal)
        instruction = f"Go to the point at ({world_goal[0]:.2f}, {world_goal[1]:.2f}) metres."
        target["object_ids"] = []
    elif task_family == "relative_pose_navigation":
        world_start, world_goal = cell_world(grid, start), cell_world(grid, goal)
        offset = [world_goal[0] - world_start[0], world_goal[1] - world_start[1], 0.0]
        instruction = (
            f"From your current position, move {offset[0]:.2f} metres along world X "
            f"and {offset[1]:.2f} metres along world Y."
        )
        target["object_ids"] = []
    task = {
        "task_id": task_id,
        "task_type": "navigation",
        "task_family": task_family,
        "capabilities": ["navigation"],
        "depends_on": [],
        "target": target,
        "instruction": instruction,
    }
    if task_family == "relative_pose_navigation":
        task["reference_pose"] = {
            "coordinate_frame": "world",
            "position": cell_world(grid, start),
            "orientation": [0.0, 0.0, 0.0, 1.0],
        }
        task["relative_goal"] = {"position_offset": offset, "orientation_offset": None}
    expected_outcome = {
        "reached_target": True,
        "goal_pose": {"position": cell_world(grid, goal), "orientation": None},
    }
    if target.get("object_ids") and (target_geometry := _target_geometry(entity)):
        expected_outcome["target_geometry"] = target_geometry
    gt = {
        "task_id": task_id,
        "task_type": "navigation",
        "task_family": task_family,
        "target": target,
        "preconditions": [],
        "expected_outcome": expected_outcome,
        "reference_trajectory": {
            "coordinate_frame": "world",
            "points": [{"position": cell_world(grid, cell), "orientation": None} for cell in route],
            "length_m": sum(
                grid["resolution"] * math.sqrt(2 if a[0] != b[0] and a[1] != b[1] else 1)
                for a, b in zip(route, route[1:])
            ),
            "planner": {
                "name": "grid_shortest_path",
                "version": 1,
                "connectivity": grid["connectivity"],
                "robot_radius_m": grid["robot_radius"],
            },
        },
        "success_criteria": {
            "target_distance_type": "aabb_2d" if target.get("object_ids") else "point",
            "position_tolerance_m": 0.8,
            "orientation_tolerance_deg": None,
            "collision_allowed": True,
        },
    }
    return task, gt


def navigation_components(grid: dict) -> list[dict]:
    # Four-neighbor connectivity equals eight-neighbor connectivity when diagonal
    # corner cutting is forbidden, as it is in the navigation planner.
    free = {
        (x, y)
        for y, row in enumerate(grid["static_occupancy"])
        for x, blocked in enumerate(row)
        if not blocked and not grid["dynamic_occupancy"][y][x]
    }
    components = []
    while free:
        start = min(free)
        free.remove(start)
        queue = deque([start])
        labels = set()
        count = 0
        while queue:
            x, y = queue.popleft()
            count += 1
            labels.add(int(grid["room_labels"][y][x]))
            for cell in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if cell in free:
                    free.remove(cell)
                    queue.append(cell)
        components.append(
            {
                "free_cells": count,
                "room_ids": [
                    grid["room_ids"][i - 1]
                    for i in sorted(labels)
                    if 0 < i <= len(grid["room_ids"])
                ],
            }
        )
    return sorted(components, key=lambda c: (-len(c["room_ids"]), -c["free_cells"]))


def validate_connectivity(engine):
    required = {r["room_id"] for r in engine.rooms if r.get("navigable", True)}
    if not any(required <= set(c["room_ids"]) for c in navigation_components(engine.navigation)):
        raise ValueError("hidden change disconnects navigable rooms")
