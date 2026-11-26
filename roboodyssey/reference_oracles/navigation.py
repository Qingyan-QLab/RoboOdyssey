"""Point-to-point reference distances delegated to restools on frozen maps."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
from importlib.metadata import version
import json
import math

import numpy as np
from restools.navigation import GridMap, Goal2D, PlannerSession, Pose2D, is_segment_collision_free

from .types import Point, StateSource, Status
REFERENCE_VERSION = "restools_holonomic_regions_v1"


def point_xy(value) -> Point:
    if len(value) != 2:
        raise ValueError("a point needs exactly two world coordinates")
    point = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in point):
        raise ValueError("point coordinates must be finite")
    return point


@dataclass(frozen=True)
class NavigationSnapshot:
    """Uninflated occupancy, metres, world XY, rows increasing with world Y.

    Copy input arrays into immutable tuples. A new world state requires a new
    snapshot; cached searches never observe edits to the caller's arrays.
    A single circular footprint applies to every occupied cell in this model.
    """

    occupancy: tuple[tuple[bool, ...], ...]
    resolution_m: float
    robot_radius_m: float
    state_source: StateSource
    origin_xy: Point = (0.0, 0.0)
    safety_margin_m: float = 0.0
    revision: str = "0"
    occupancy_kind: str = "raw"
    digest: str = field(init=False)

    def __post_init__(self):
        raw = np.asarray(self.occupancy)
        if raw.ndim != 2 or not all(raw.shape) or not np.isin(raw, [0, 1]).all():
            raise ValueError("occupancy must be a nonempty binary 2D map")
        if self.occupancy_kind != "raw":
            raise ValueError("provide raw occupancy; preinflated maps would count radius twice")
        if self.state_source not in ("planned", "runtime"):
            raise ValueError("state_source must be planned or runtime")
        for name in ("resolution_m", "robot_radius_m", "safety_margin_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0 or (name == "resolution_m" and value == 0):
                raise ValueError("invalid " + name)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "origin_xy", point_xy(self.origin_xy))
        object.__setattr__(self, "occupancy", tuple(tuple(bool(v) for v in row) for row in raw))
        if not isinstance(self.revision, str) or not self.revision:
            raise ValueError("revision must be a nonempty string")
        description = dict(shape=raw.shape, resolution=self.resolution_m,
                           radius=self.robot_radius_m, margin=self.safety_margin_m,
                           origin=self.origin_xy, revision=self.revision,
                           state_source=self.state_source, occupancy_kind=self.occupancy_kind,
                           reference_version=REFERENCE_VERSION)
        data = json.dumps(description, sort_keys=True, allow_nan=False).encode()
        object.__setattr__(self, "digest", hashlib.sha256(data + raw.astype(np.uint8).tobytes()).hexdigest())


@dataclass(frozen=True)
class SearchLimits:
    max_expansions: int = 200_000
    max_search_calls: int = 20_000
    max_candidates_per_goal: int = 256
    cached_goals: int = 4

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(name + " must be a positive integer")


@dataclass
class SearchBudget:
    limits: SearchLimits
    calls: int = 0

    def consume(self) -> bool:
        if self.calls >= self.limits.max_search_calls:
            return False
        self.calls += 1
        return True


@dataclass(frozen=True)
class PathResult:
    status: Status
    cost_m: float | None = None
    points: tuple[Point, ...] = ()
    reason: str = ""


class PointNavigator:
    """Exact minimum on the configured restools graph plus local connectors.

    Cell-centre endpoints use their own graph node. Other endpoints connect to
    all visible centres in their surrounding 3x3 cells, with Euclidean costs.
    No distant snapping, path smoothing, heading cost or proximity cost is used.
    This reference model is not the legacy Benchmark distance implementation.
    Instances are private to a calculation and are not thread-safe.
    """

    def __init__(self, snapshot: NavigationSnapshot, limits: SearchLimits | None = None,
                 *, budget: SearchBudget | None = None):
        self.snapshot = snapshot
        self.limits = limits or SearchLimits()
        self.budget = budget or SearchBudget(self.limits)
        self._grid = GridMap(snapshot.occupancy, snapshot.resolution_m, snapshot.origin_xy)
        self._sessions = OrderedDict()
        self._solutions = {}

    @property
    def backend_version(self) -> str:
        return version("restools")

    def walkable(self, point: Point) -> bool:
        point = point_xy(point)
        return self._grid.collision_free(*point, self.snapshot.robot_radius_m,
                                         self.snapshot.safety_margin_m)

    def centres(self):
        for row in range(self._grid.shape[0]):
            for col in range(self._grid.shape[1]):
                yield self._grid.center(row, col)

    def _connectors(self, point):
        row, col = self._grid.index(*point)
        centre = self._grid.center(row, col)
        if math.dist(point, centre) <= 1e-12:
            return [(centre, math.dist(point, centre))]
        result = []
        for rr in range(row - 1, row + 2):
            for cc in range(col - 1, col + 2):
                if not self._grid.contains(rr, cc):
                    continue
                centre = self._grid.center(rr, cc)
                if self.walkable(centre) and is_segment_collision_free(
                        self._grid, point, centre, self.snapshot.robot_radius_m,
                        self.snapshot.safety_margin_m):
                    result.append((centre, math.dist(point, centre)))
        return result

    def _session(self, goal):
        if goal not in self._sessions:
            self._sessions[goal] = PlannerSession(
                self._grid, self.snapshot.robot_radius_m, self.snapshot.safety_margin_m,
                motion_model="holonomic", soft_clearance_m=0.0, proximity_weight=0.0)
            if len(self._sessions) > self.limits.cached_goals:
                self._sessions.popitem(last=False)
        self._sessions.move_to_end(goal)
        return self._sessions[goal]

    def distance(self, start: Point, goal: Point) -> PathResult:
        start, goal = point_xy(start), point_xy(goal)
        key = start, goal
        if key in self._solutions:
            return self._solutions[key][0]
        if not self.walkable(start) or not self.walkable(goal):
            return PathResult("invalid", reason="endpoint_in_collision")
        if start == goal:
            result = PathResult("ok", 0.0, (start,))
            self._solutions[key] = result, None
            return result
        starts, goals = self._connectors(start), self._connectors(goal)
        if not starts or not goals:
            return PathResult("unreachable", reason="no_visible_local_connector")
        best, anchors = math.inf, None
        for target, target_offset in goals:
            session = self._session(target)
            for source, source_offset in starts:
                if not self.budget.consume():
                    return PathResult("indeterminate", reason="search_call_limit")
                query = session.cost_to_goal(Pose2D(*source), Goal2D(*target),
                                             max_expansions=self.limits.max_expansions)
                if query.reason == "expansion limit":
                    return PathResult("indeterminate", reason="expansion_limit")
                if query.reason and query.reason != "unreachable":
                    return PathResult("invalid", reason=query.reason)
                cost = source_offset + query.cost + target_offset
                if cost < best:
                    best, anchors = float(cost), (source, target)
        result = (PathResult("ok", best) if anchors is not None else
                  PathResult("unreachable", reason="no_route"))
        self._solutions[key] = result, anchors
        return result

    def path(self, start: Point, goal: Point) -> PathResult:
        start, goal = point_xy(start), point_xy(goal)
        result = self.distance(start, goal)
        if result.status != "ok" or result.points:
            return result
        source, target = self._solutions[start, goal][1]
        if not self.budget.consume():
            return PathResult("indeterminate", reason="search_call_limit")
        plan = self._session(target).plan(Pose2D(*source), Goal2D(*target),
                                          max_expansions=self.limits.max_expansions,
                                          use_lookahead=False)
        if not plan.success:
            return PathResult("indeterminate", reason="path_extraction: " + plan.reason)
        points = [start]
        for point in [(p.x, p.y) for p in plan.poses] + [goal]:
            if point != points[-1]:
                points.append(point)
        length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        if not math.isclose(length, result.cost_m, rel_tol=1e-9, abs_tol=1e-9):
            return PathResult("indeterminate", reason="path_cost_mismatch")
        return PathResult("ok", result.cost_m, tuple(points))
