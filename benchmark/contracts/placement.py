"""Frozen-data runtime helpers. Adapted from MIT-licensed sources; see LICENSE."""
from __future__ import annotations

from copy import deepcopy

import math

from .navigation import cell_world, navigation_task, _parents_from_goal

from .raster import _point_in_polygon, derive_navigation

from .doors import collision_bounds

class PlacementError(ValueError):
    pass

def translated_entity(entity: dict, bounds: dict, room_id: str) -> dict:
    result = deepcopy(entity)
    previous = entity["geometry"]["world_aabb"]
    offset = [(bounds["min"][i] + bounds["max"][i]
               - previous["min"][i] - previous["max"][i]) / 2 for i in range(3)]
    result["room_id"] = room_id
    result["geometry"]["world_aabb"] = deepcopy(bounds)
    matrix = result["geometry"].get("world_matrix4x4")
    if matrix:
        for i in range(3):
            matrix[3][i] += offset[i]
    return result

def _intersects(a: dict, b: dict, clearance: float) -> bool:
    return all(a["min"][i] < b["max"][i] + clearance
               and a["max"][i] > b["min"][i] - clearance for i in range(3))

def floor_placement(entity: dict, destination: dict, entities: list[dict], rooms: list[dict],
                    navigation: dict, *, start: tuple[int, int] | None = None,
                    clearance: float = .03, max_attempts: int = 80) -> dict:
    """Select a reproducible free floor pose and prove its grid approach.

    The source must be reachable from the actual planned start. Destination
    reachability is checked from that pickup approach, both before and after
    inserting the moved object's footprint. Exact selected geometry is private GT.
    """
    if clearance < 0 or max_attempts < 1:
        raise PlacementError("invalid placement search parameters")
    bounds = entity.get("geometry", {}).get("world_aabb")
    if not bounds:
        raise PlacementError("cannot move an object with unknown geometry")
    sizes = [float(bounds["max"][i]) - float(bounds["min"][i]) for i in range(3)]
    if any(not math.isfinite(size) or size <= 0 for size in sizes):
        raise PlacementError("invalid object dimensions")
    room_id = str(destination["room_id"])
    if room_id == entity.get("room_id"):
        raise PlacementError("region transfer must change room")
    source = navigation_task(entity, navigation, 0, start=start)
    if source is None:
        raise PlacementError("pickup object is not reachable from current start")
    pickup = source[1]["reference_trajectory"]["points"][-1]["position"]
    pickup_cell = tuple(int((pickup[i] - navigation["origin_xy"][i]) / navigation["resolution"])
                        for i in (0, 1))
    others = [item for item in entities if item["object_id"] != entity["object_id"]]
    base_grid = derive_navigation(rooms, others, resolution=navigation["resolution"],
                                  robot_radius=navigation["robot_radius"],
                                  environment_setup=navigation.get('environment_setup'))
    base_grid['scene_id'] = navigation.get('scene_id')
    polygon = destination.get("boundary_polygon", [])
    if not polygon:
        raise PlacementError("destination has no floor polygon")
    center = [sum(p[i] for p in polygon) / len(polygon) for i in (0, 1)]
    label = base_grid["room_ids"].index(room_id) + 1
    reachable = _parents_from_goal(base_grid, pickup_cell)
    cells = [(x, y) for y, row in enumerate(base_grid["room_labels"]) for x, value in enumerate(row)
             if value == label and not base_grid["static_occupancy"][y][x]
             and not base_grid["dynamic_occupancy"][y][x] and (x, y) in reachable]
    if not cells:
        raise PlacementError(f"destination floor is disconnected from pickup in {room_id}")
    cells.sort(key=lambda cell: (math.dist(cell_world(base_grid, cell)[:2], center), cell))
    # Sample spread-out positions instead of spending every attempt in one cluster.
    step = max(1, len(cells) // max_attempts)
    for cell in cells[::step][:max_attempts]:
        point = cell_world(base_grid, cell)
        minimum = [point[0] - sizes[0] / 2, point[1] - sizes[1] / 2,
                   float(destination.get("floor_height", 0))]
        maximum = [minimum[i] + sizes[i] for i in range(3)]
        candidate = {"min": minimum, "max": maximum}
        # Check the full footprint perimeter against potentially concave rooms.
        spacing = base_grid["resolution"] / 2
        perimeter = []
        for axis in (0, 1):
            count = max(1, math.ceil((sizes[axis] + 2 * clearance) / spacing))
            for j in range(count + 1):
                along = minimum[axis] - clearance + (sizes[axis] + 2 * clearance) * j / count
                for across in (minimum[1-axis] - clearance, maximum[1-axis] + clearance):
                    perimeter.append((along, across) if axis == 0 else (across, along))
        if not all(_point_in_polygon(x, y, polygon) for x, y in perimeter):
            continue
        if any(_intersects(candidate, box, clearance)
               for other in others for box in collision_bounds(other, navigation)):
            continue
        moved = translated_entity(entity, candidate, room_id)
        approach = navigation_task(moved, base_grid, 0, start=pickup_cell)
        if approach is None:
            continue
        final_grid = derive_navigation(rooms, others + [moved], resolution=base_grid["resolution"],
                                       robot_radius=base_grid["robot_radius"],
                                       environment_setup=navigation.get('environment_setup'))
        final_grid['scene_id'] = navigation.get('scene_id')
        final = navigation_task(moved, final_grid, 0, start=pickup_cell)
        if final is None:
            continue
        return {"entity": moved, "navigation": final_grid,
                "pickup_trajectory": source[1]["reference_trajectory"],
                "placement_trajectory": final[1]["reference_trajectory"],
                "validation": {"method": "floor_polygon_aabb_and_grid_approach_v1",
                               "clearance_m": clearance, "physics_execution_verified": False},
                "relation_after": {"type": "in_room", "room_id": room_id}}
    raise PlacementError(f"no collision-free, approachable floor placement in {room_id}")
