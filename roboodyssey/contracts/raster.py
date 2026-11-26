"""Frozen-data runtime helpers. Adapted from MIT-licensed sources; see LICENSE."""
from __future__ import annotations

from copy import deepcopy

import math

def _point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    inside = False
    for i, point in enumerate(polygon):
        x1, y1 = point
        x2, y2 = polygon[i - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            inside = not inside
    return inside

def _aabb(obj: dict) -> tuple[list[float], list[float]] | None:
    value = obj.get("geometry", {}).get("world_aabb") or obj.get("world_aabb")
    if not value or not value.get("min") or not value.get("max"):
        return None
    return list(value["min"]), list(value["max"])

def _mark_aabb(grid: list[list[int]], bounds: tuple[list[float], list[float]], origin: list[float], resolution: float, inflate: int) -> None:
    minimum, maximum = bounds
    height, width = len(grid), len(grid[0])
    x0 = max(0, int(math.floor((minimum[0] - origin[0]) / resolution)) - inflate)
    x1 = min(width - 1, int(math.ceil((maximum[0] - origin[0]) / resolution)) + inflate)
    y0 = max(0, int(math.floor((minimum[1] - origin[1]) / resolution)) - inflate)
    y1 = min(height - 1, int(math.ceil((maximum[1] - origin[1]) / resolution)) + inflate)
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            grid[y][x] = 1

def _clear_aabb(grid: list[list[int]], bounds: tuple[list[float], list[float]], origin: list[float], resolution: float) -> None:
    """Clear only the panel footprint, leaving the surrounding frame/wall."""
    minimum, maximum = bounds
    height, width = len(grid), len(grid[0])
    x0 = max(0, int(math.floor((minimum[0] - origin[0]) / resolution)))
    x1 = min(width - 1, int(math.ceil((maximum[0] - origin[0]) / resolution)))
    y0 = max(0, int(math.floor((minimum[1] - origin[1]) / resolution)))
    y1 = min(height - 1, int(math.ceil((maximum[1] - origin[1]) / resolution)))
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            grid[y][x] = 0

def _inflate_occupancy(grid: list[list[int]], cells: int) -> list[list[int]]:
    if cells <= 0:
        return [list(row) for row in grid]
    height, width = len(grid), len(grid[0])
    result = [list(row) for row in grid]
    for y, row in enumerate(grid):
        for x, occupied in enumerate(row):
            if not occupied:
                continue
            for dy in range(-cells, cells + 1):
                for dx in range(-cells, cells + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        result[ny][nx] = 1
    return result

def derive_navigation(rooms: list[dict], objects: list[dict], *, resolution: float = .05,
                      robot_radius: float | None = None, environment_setup: dict | None = None) -> dict:
    profile = None
    if robot_radius is None:
        from ..scenes.robot_body import default_robot_identity
        profile = default_robot_identity()
        robot_radius = profile['robot_radius_m']
    if resolution <= 0 or robot_radius < 0:
        raise ValueError('navigation dimensions must be positive')
    points = [point for room in rooms for point in room.get("boundary_polygon", [])]
    if not points:
        raise ValueError("scene metadata contains no room boundary polygons")
    origin = [min(point[i] for point in points) - resolution for i in (0, 1)]
    maximum = [max(point[i] for point in points) + resolution for i in (0, 1)]
    width = int(math.ceil((maximum[0] - origin[0]) / resolution)) + 1
    height = int(math.ceil((maximum[1] - origin[1]) / resolution)) + 1
    room_ids = [str(room["room_id"]) for room in rooms]
    labels = [[0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            wx = origin[0] + (x + .5) * resolution
            wy = origin[1] + (y + .5) * resolution
            for index, room in enumerate(rooms, 1):
                if room.get("navigable", True) and _point_in_polygon(wx, wy, room.get("boundary_polygon", [])):
                    labels[y][x] = index
                    break
    inflate = max(0, int(math.ceil(robot_radius / resolution)))
    # Keep the uninflated grid as the authoritative geometric occupancy.  The
    # inflated copy is still useful to the offline planner, but the runtime
    # collision checker must apply the circular agent footprint exactly once.
    static_base = [[int(label == 0) for label in row] for row in labels]
    static = static_base
    dynamic = [[0] * width for _ in range(height)]
    open_doors = {d['object_id']: d for d in (environment_setup or {}).get('doors', [])}
    if environment_setup:
        from .doors import POLICY
        if environment_setup.get('policy') != POLICY:
            raise ValueError('unsupported navigation door policy')
        expected = {o['object_id'] for o in objects if str(o.get('category', '')).lower() == 'door'}
        if set(open_doors) != expected:
            raise ValueError('open-door geometry must match all doors in navigation')
        # First open just the authored portals in the room-polygon wall mask.
        # All frame parts, open leaves and other obstacles are then added back.
        for door in open_doors.values():
            portal = door['portal_aabb']
            _clear_aabb(static_base, (portal['min'], portal['max']), origin, resolution)
    for obj in objects:
        category = str(obj.get("category", "")).lower()
        if obj['object_id'] in open_doors:
            floor = next((r.get('floor_height', 0.) for r in rooms if r['room_id'] == obj.get('room_id')), 0.)
            for collider in open_doors[obj['object_id']]['colliders']:
                box = collider['world_aabb']
                if box['max'][2] > floor + .05 and box['min'][2] < floor + 1.5:
                    _mark_aabb(static_base, (box['min'], box['max']), origin, resolution, 0)
            continue
        bodies = obj.get("physics", {}).get("rigid_bodies", [])
        movable = category == "door" or any(item.get("enabled") and not item.get("kinematic", False) for item in bodies)
        bounds = _aabb(obj)
        if bounds:
            # Dynamic bodies remain uninflated.  The runtime applies the
            # agent's circular footprint while sweeping, so inflating a door
            # here would double-count the robot radius and narrow the doorway.
            _mark_aabb(dynamic if movable else static_base, bounds, origin, resolution, 0)
            if movable and category == "door":
                # Room polygons can rasterize the wall behind a closed door as
                # outside-room cells.  Reserve the panel footprint as the
                # doorway; the closed panel remains the dynamic blocker.
                _clear_aabb(static_base, bounds, origin, resolution)
    static = _inflate_occupancy(static_base, inflate)
    return {"format_version": 1, "method": "room_polygon_aabb_grid", "scene_id": None,
            "resolution": resolution, "origin_xy": origin, "floor_height": 0.0,
            "robot_radius": robot_radius, "connectivity": 8,
            "static_occupancy": static, "static_base_occupancy": static_base,
            "dynamic_occupancy": dynamic,
            "room_labels": labels, "room_ids": room_ids,
            **({'environment_setup': deepcopy(environment_setup)} if environment_setup else {}),
            **({'robot_profile': profile} if profile is not None else {})}
