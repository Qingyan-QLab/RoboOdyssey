"""High-level finite-goal task planning; all spatial searches use restools."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
import math

from .navigation import (NavigationSnapshot, Point, PointNavigator, REFERENCE_VERSION,
                         SearchBudget, SearchLimits, StateSource, Status, point_xy)


@dataclass(frozen=True)
class TargetPose:
    position: Point
    yaw_rad: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "position", point_xy(self.position))
        if self.yaw_rad is not None and not math.isfinite(self.yaw_rad):
            raise ValueError("target yaw must be finite radians")


@dataclass(frozen=True)
class GoalRegion:
    """XY AABB expanded by Euclidean distance, including its boundary."""

    minimum: Point
    maximum: Point
    tolerance_m: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "minimum", point_xy(self.minimum))
        object.__setattr__(self, "maximum", point_xy(self.maximum))
        if any(a > b for a, b in zip(self.minimum, self.maximum)):
            raise ValueError("inverted goal AABB")
        if not math.isfinite(self.tolerance_m) or self.tolerance_m < 0:
            raise ValueError("invalid goal tolerance")

    def contains(self, point: Point) -> bool:
        return math.hypot(*(max(lo - p, 0.0, p - hi) for lo, hi, p in
                            zip(self.minimum, self.maximum, point))) <= self.tolerance_m


@dataclass(frozen=True)
class CandidateGoal:
    """Explicit legal target poses, for example simulator-validated placements.

    The caller certifies interaction geometry; navigation verifies only the
    circular robot's occupancy. Yaw is a required terminal orientation when
    supplied, but does not contribute to the translational reference cost.
    """

    poses: tuple[TargetPose, ...]

    def __post_init__(self):
        poses = tuple(self.poses)
        if not poses or not all(isinstance(pose, TargetPose) for pose in poses):
            raise ValueError("candidate goal needs explicit TargetPose samples")
        object.__setattr__(self, "poses", tuple(dict.fromkeys(poses)))


@dataclass(frozen=True)
class Interaction:
    action: str
    object_id: str
    cost: float = 1.0

    def __post_init__(self):
        if self.action not in ("pick", "place") or not self.object_id:
            raise ValueError("interaction needs pick/place and a target object")
        if not math.isfinite(self.cost) or self.cost < 0:
            raise ValueError("interaction cost must be finite and nonnegative")


@dataclass(frozen=True)
class ReferenceStage:
    snapshot: NavigationSnapshot
    goal: GoalRegion | CandidateGoal
    after: Interaction | None = None


@dataclass(frozen=True)
class ReferenceLeg:
    target: TargetPose
    points: tuple[Point, ...]
    length_m: float
    after: Interaction | None
    snapshot_digest: str


@dataclass(frozen=True)
class TaskReference:
    status: Status
    cost: float | None
    cost_unit: str
    metres_per_cost_unit: float
    legs: tuple[ReferenceLeg, ...]
    reason: str
    state_source: StateSource
    snapshot_digests: tuple[str, ...]
    search_calls: int
    limits: SearchLimits
    backend_version: str
    reference_version: str = REFERENCE_VERSION
    optimality: str = "finite_goal_samples_on_restools_grid_with_local_connectors"

    def to_dict(self) -> dict:
        """JSON-serializable private reference; never an Agent observation."""
        return asdict(self)


class ReferenceOracle:
    """Jointly choose stage endpoints before extracting point-to-point paths.

    This computes reference plans/costs, not executable simulator commands or
    success labels. It does not implement OraclePolicy.decide(). Callers retain
    task lifecycle, history binding, live state readback and scoring authority.
    """

    def __init__(self, limits: SearchLimits | None = None):
        self.limits = limits or SearchLimits()

    def navigation(self, snapshot: NavigationSnapshot, start: Point,
                   goals: list[GoalRegion | CandidateGoal]) -> TaskReference:
        return self.plan_stages(start, [ReferenceStage(snapshot, goal) for goal in goals])

    def transfer(self, approach: NavigationSnapshot, carrying: NavigationSnapshot,
                 start: Point, *, object_id: str, object_origin_xy: Point,
                 placements: CandidateGoal, pick_radius_m: float = 1.5,
                 move_step_m: float = 0.25, already_held: bool = False) -> TaskReference:
        """Approach -> pick -> carry -> place, optimized jointly over candidates.

        ``carrying`` must remove the target's old occupancy without removing
        overlapping obstacles. ``placements`` must already satisfy actual room,
        floor, object-pivot/forward-offset and orientation constraints. This
        layer cannot derive those facts from a 2D occupancy map.
        """
        if not isinstance(placements, CandidateGoal) or any(p.yaw_rad is None for p in placements.poses):
            raise ValueError("transfer requires legal placements with explicit yaw")
        if not math.isfinite(pick_radius_m) or pick_radius_m <= 0:
            raise ValueError("pick radius must be positive")
        if approach.state_source != carrying.state_source:
            raise ValueError("cannot mix planned and runtime transfer snapshots")
        pickup = GoalRegion(object_origin_xy, object_origin_xy, pick_radius_m)
        stages = [] if already_held else [ReferenceStage(approach, pickup, Interaction("pick", object_id))]
        stages.append(ReferenceStage(carrying, placements, Interaction("place", object_id)))
        return self.plan_stages(start, stages, metres_per_cost_unit=move_step_m,
                                cost_unit="equivalent_actions")

    def _candidates(self, goal, navigator, previous):
        if isinstance(goal, CandidateGoal):
            poses = (p for p in goal.poses if navigator.walkable(p.position))
        elif isinstance(goal, GoalRegion):
            # Include actual previous endpoints for zero-distance visits in
            # overlapping regions, in addition to the declared grid sampling.
            from itertools import chain
            poses = (TargetPose(p) for p in chain(previous, navigator.centres())
                     if goal.contains(p) and navigator.walkable(p))
        else:
            raise TypeError("expected GoalRegion or CandidateGoal")
        result = dict()
        for pose in poses:
            result[pose] = None
            if len(result) > self.limits.max_candidates_per_goal:
                return None
        return tuple(result)

    def plan_stages(self, start: Point, stages: list[ReferenceStage], *,
                    metres_per_cost_unit: float = 1.0, cost_unit: str = "m") -> TaskReference:
        """Dynamic programming over all candidates; never greedy or truncated.

        Optimality is relative to the finite goal samples and the declared
        restools graph. Any unresolved relevant search prevents an optimality
        claim and returns indeterminate, even if another route was found.
        """
        start = point_xy(start)
        stages = tuple(stages)
        if not stages:
            raise ValueError("a reference needs at least one stage")
        if not math.isfinite(metres_per_cost_unit) or metres_per_cost_unit <= 0:
            raise ValueError("metres_per_cost_unit must be positive")
        if cost_unit not in ("m", "equivalent_actions"):
            raise ValueError("unsupported cost unit")
        if cost_unit == "m" and (metres_per_cost_unit != 1 or any(s.after for s in stages)):
            raise ValueError("metre references cannot include operation costs or scaling")
        sources = {s.snapshot.state_source for s in stages}
        if len(sources) != 1:
            raise ValueError("cannot mix planned and runtime snapshots")
        budget = SearchBudget(self.limits)
        maps = {}
        navigators = []
        for stage in stages:
            if stage.snapshot.digest not in maps:
                maps[stage.snapshot.digest] = PointNavigator(stage.snapshot, self.limits, budget=budget)
            navigators.append(maps[stage.snapshot.digest])

        def finish(status, cost=None, legs=(), reason=""):
            return TaskReference(status, cost, cost_unit, metres_per_cost_unit, tuple(legs), reason,
                                 stages[0].snapshot.state_source,
                                 tuple(s.snapshot.digest for s in stages), budget.calls,
                                 self.limits, version("restools"))

        if not navigators[0].walkable(start):
            return finish("invalid", reason="start_in_collision")
        previous = (TargetPose(start),)
        costs = [0.0]
        layers, parents = [], []
        for index, (stage, navigator) in enumerate(zip(stages, navigators)):
            candidates = self._candidates(stage.goal, navigator, (p.position for p in previous))
            if candidates is None:
                return finish("indeterminate", reason=f"stage_{index}: candidate_limit")
            if not candidates:
                return finish("unreachable", reason=f"stage_{index}: no_legal_goal_samples")
            next_costs, predecessors = [], []
            action_cost = stage.after.cost if stage.after else 0.0
            for target in candidates:
                best, parent = math.inf, None
                for prev_index, (source, cost) in enumerate(zip(previous, costs)):
                    if not math.isfinite(cost) or not navigator.walkable(source.position):
                        continue
                    distance = navigator.distance(source.position, target.position)
                    if distance.status in ("indeterminate", "invalid"):
                        return finish(distance.status, reason=f"stage_{index}: {distance.reason}")
                    if distance.status == "unreachable":
                        continue
                    candidate_cost = cost + distance.cost_m / metres_per_cost_unit + action_cost
                    if candidate_cost < best:
                        best, parent = candidate_cost, prev_index
                next_costs.append(best)
                predecessors.append(parent)
            if not any(math.isfinite(cost) for cost in next_costs):
                return finish("unreachable", reason=f"stage_{index}: no_route")
            layers.append(candidates)
            parents.append(predecessors)
            previous, costs = candidates, next_costs

        selected = min(range(len(costs)), key=costs.__getitem__)
        total = costs[selected]
        targets = []
        for candidates, predecessors in zip(reversed(layers), reversed(parents)):
            targets.append(candidates[selected])
            selected = predecessors[selected]
        targets.reverse()
        legs, previous_point = [], start
        for target, navigator, stage in zip(targets, navigators, stages):
            path = navigator.path(previous_point, target.position)
            if path.status != "ok":
                return finish(path.status, reason=path.reason)
            legs.append(ReferenceLeg(target, path.points, path.cost_m, stage.after, stage.snapshot.digest))
            previous_point = target.position
        return finish("ok", total, legs)
