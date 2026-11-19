"""Runtime planning for compact session room introductions.

The dataset stores only room ids, text and a render-region hint.  This module
turns those hints into a short walkable camera route after the current session
state and navigation grid have been applied.  No camera trajectory or image is
persisted in the dataset.
"""
from __future__ import annotations

from collections import deque
import math
from typing import Any


def _cell_world(navigation: dict[str, Any], cell: tuple[int, int], height: float) -> list[float]:
    origin = navigation["origin_xy"]
    resolution = float(navigation["resolution"])
    return [float(origin[0]) + (cell[0] + 0.5) * resolution,
            float(origin[1]) + (cell[1] + 0.5) * resolution,
            float(height)]


def _quat_for_yaw(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def _point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    inside = False
    for first, second in zip(polygon, polygon[1:] + polygon[:1]):
        x1, y1 = float(first[0]), float(first[1])
        x2, y2 = float(second[0]), float(second[1])
        if ((y1 > y) != (y2 > y)) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            inside = not inside
    return inside


def _path(
    grid: dict[str, Any], start: tuple[int, int], goal: tuple[int, int],
    allowed: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    queue = deque([start])
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    width = len(grid["room_labels"][0])
    height = len(grid["room_labels"])
    while queue:
        current = queue.popleft()
        if current == goal:
            route = [current]
            while parents[route[-1]] is not None:
                route.append(parents[route[-1]])
            return list(reversed(route))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if nxt not in allowed or nxt in parents:
                continue
            parents[nxt] = current
            queue.append(nxt)
    return None


def plan_room_trajectory(env: Any, room_id: str, *, sample_count: int = 8,
                         waypoint_spacing_m: float = 0.25) -> dict[str, Any] | None:
    """Plan a deterministic, walkable route inside ``room_id``.

    The planner uses the environment's current effective static/dynamic grid,
    room boundary from scene metadata, and current camera height.  Consequently
    mutations and already-open doors affect the route without changing
    ``session.json``.
    """
    raw = getattr(env, "_navigation", None)
    metadata = getattr(env, "_metadata", {})
    if not isinstance(raw, dict) or not room_id:
        return None
    room_ids = [str(value) for value in raw.get("room_ids", [])]
    if room_id not in room_ids:
        return None
    room_index = room_ids.index(room_id) + 1
    labels = raw.get("room_labels")
    static = getattr(env, "_static_base", None)
    dynamic = getattr(env, "_dynamic", None)
    if labels is None or static is None or dynamic is None:
        return None
    boundary = next((item.get("boundary_polygon") for item in metadata.get("rooms", [])
                     if str(item.get("room_id")) == room_id), None)
    if not isinstance(boundary, list) or len(boundary) < 3:
        boundary = None
    height, width = static.shape
    allowed: set[tuple[int, int]] = set()
    for y in range(height):
        for x in range(width):
            if int(labels[y, x]) != room_index or bool(static[y, x]) or bool(dynamic[y, x]):
                continue
            world = _cell_world(raw, (x, y), 0.0)
            if boundary is not None and not _point_in_polygon(world[0], world[1], boundary):
                continue
            pose = [world[0], world[1], float(getattr(env, "_floor_height", 0.0)) +
                    float(getattr(getattr(env, "config", None), "camera_height", 1.5))]
            if env._walkable(pose):
                allowed.add((x, y))
    if len(allowed) < 2:
        return None

    center = [sum(_cell_world(raw, cell, 0.0)[axis] for cell in allowed) / len(allowed)
              for axis in (0, 1)]
    sectors: dict[int, tuple[float, tuple[int, int]]] = {}
    for cell in allowed:
        world = _cell_world(raw, cell, 0.0)
        angle = math.atan2(world[1] - center[1], world[0] - center[0])
        sector = int(((angle + math.pi) / (2.0 * math.pi)) * sample_count) % sample_count
        distance = math.dist(world[:2], center)
        if sector not in sectors or distance > sectors[sector][0]:
            sectors[sector] = (distance, cell)
    anchors = [sectors[index][1] for index in sorted(sectors)]
    if len(anchors) < 2:
        return None
    route: list[tuple[int, int]] = []
    for index, anchor in enumerate(anchors):
        part = _path(raw, anchor, anchors[(index + 1) % len(anchors)], allowed)
        if not part:
            continue
        route.extend(part if not route else part[1:])
    if len(route) < 2:
        return None

    spacing = max(1, int(round(float(waypoint_spacing_m) / float(raw["resolution"]))))
    waypoints = [route[0]]
    last_direction = None
    distance = 0
    for previous, current in zip(route, route[1:]):
        direction = (current[0] - previous[0], current[1] - previous[1])
        distance += 1
        if ((last_direction is not None and direction != last_direction)
                or distance >= spacing):
            if waypoints[-1] != previous:
                waypoints.append(previous)
            distance = 1
        last_direction = direction
    if waypoints[-1] != route[-1]:
        waypoints.append(route[-1])

    camera_height = float(getattr(getattr(env, "config", None), "camera_height", 1.5))
    floor = float(getattr(env, "_floor_height", 0.0))
    poses = []
    for cell in waypoints:
        position = _cell_world(raw, cell, floor + camera_height)
        yaw = math.atan2(center[1] - position[1], center[0] - position[0])
        poses.append({"coordinate_frame": "world", "position": position,
                      "orientation": _quat_for_yaw(yaw)})
    return {"coordinate_frame": "world", "interpolation": "grid_walkable",
            "loop": False, "waypoint_spacing_m": float(waypoint_spacing_m),
            "frame_interval_s": float(getattr(
                getattr(env, "config", None), "step_seconds", 1 / 30)),
            "poses": poses}


def session_start_pose(
    session: Any, ground_truth: dict[str, Any], env: Any,
) -> dict[str, Any] | None:
    """Resolve the task start pose without exposing ground truth to the agent."""
    instruction = session.metadata.get("instruction")
    candidates = [session.metadata.get("start_pose"),
                  instruction.get("start_pose") if isinstance(instruction, dict) else None]
    for value in candidates:
        if isinstance(value, dict) and value.get("position") is not None:
            return _normalize_pose(value, env)
    tasks = ground_truth.get("tasks", []) if isinstance(ground_truth, dict) else []
    if tasks:
        points = tasks[0].get("reference_trajectory", {}).get("points", [])
        if points and isinstance(points[0], dict) and points[0].get("position") is not None:
            return _normalize_pose(points[0], env)
        reference = tasks[0].get("reference_pose")
        if isinstance(reference, dict) and reference.get("position") is not None:
            return _normalize_pose(reference, env)
    return None


def _normalize_pose(value: dict[str, Any], env: Any) -> dict[str, Any]:
    position = [float(item) for item in value["position"][:3]]
    floor = float(getattr(env, "_floor_height", 0.0))
    camera_height = float(getattr(getattr(env, "config", None), "camera_height", 1.5))
    if len(position) < 3:
        position.append(floor + camera_height)
    elif position[2] <= floor + 0.05:
        position[2] = floor + camera_height
    orientation = value.get("orientation")
    return {
        "coordinate_frame": "world",
        "position": position,
        "orientation": ([float(item) for item in orientation]
                         if orientation else [0.0, 0.0, 0.0, 1.0]),
    }
