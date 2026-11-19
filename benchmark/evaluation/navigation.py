"""CPU grid geodesics from a saved, actual task-start collision map.

Distances are shortest paths on an eight-neighbour grid, not Isaac navmesh
distances. No diagonal corner cutting or snapping through obstacles is allowed.
"""
from heapq import heappop, heappush
import math
import numpy as np


class GridDistance:
    def __init__(self, blocked, origin, resolution, radius=0.0, door_blocked=None, carried_radius=0.0):
        self.raw = np.asarray(blocked, dtype=bool)
        self.origin = np.asarray(origin, dtype=float)[:2]
        self.resolution, self.radius = float(resolution), float(radius)
        self.doors = np.zeros_like(self.raw) if door_blocked is None else np.asarray(door_blocked,dtype=bool)
        self.carried_radius = float(carried_radius)
        if (self.raw.ndim != 2 or not self.raw.size or self.origin.shape != (2,)
                or not np.isfinite(self.origin).all() or not math.isfinite(self.resolution)
                or not math.isfinite(self.radius) or self.resolution <= 0 or self.radius < 0
                or self.doors.shape != self.raw.shape or not math.isfinite(self.carried_radius)
                or self.carried_radius < 0):
            raise ValueError('invalid navigation snapshot')
        self.height, self.width = self.raw.shape
        yy, xx = np.indices(self.raw.shape)
        self.xx = self.origin[0] + (xx + .5) * self.resolution
        self.yy = self.origin[1] + (yy + .5) * self.resolution
        self.has_doors = bool(self.doors.any())
        self.blocked = self._inflate(self.raw, self.radius, boundary=True)
        if self.has_doors:
            self.blocked |= self._inflate(self.doors, self.carried_radius, boundary=False)

    def _inflate(self, occupied, radius, *, boundary):
        # Same circle-to-occupied-cell test as IsaacVLNEnv._walkable.
        blocked = occupied.copy()
        pad = math.ceil(radius / self.resolution + .5)
        padded = np.pad(occupied, pad, constant_values=boundary)
        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                gap = math.hypot(max(abs(dx)-.5, 0), max(abs(dy)-.5, 0)) * self.resolution
                if gap <= radius:
                    blocked |= padded[pad+dy:pad+dy+self.height, pad+dx:pad+dx+self.width]
        return blocked

    def cell(self, point):
        point = np.asarray(point, dtype=float)[:2]
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError('nonfinite navigation point')
        x, y = np.floor((point - self.origin) / self.resolution).astype(int)
        return int(x), int(y)

    def walkable(self, point):
        return (self._clear(self.raw, self.radius, point, boundary=True)
                and (not self.has_doors or self._clear(self.doors, self.carried_radius, point, boundary=False)))

    def _clear(self, occupied, radius, point, *, boundary):
        x, y = point[:2]
        lo = np.floor((np.asarray([x, y])-radius-self.origin)/self.resolution).astype(int)
        hi = np.floor((np.asarray([x, y])+radius-self.origin)/self.resolution).astype(int)
        for cy in range(lo[1], hi[1]+1):
            for cx in range(lo[0], hi[0]+1):
                if not (0 <= cx < self.width and 0 <= cy < self.height):
                    if boundary:
                        return False
                    continue
                if occupied[cy, cx]:
                    left, bottom = self.origin + np.array([cx, cy])*self.resolution
                    dx = max(left-x, 0, x-left-self.resolution)
                    dy = max(bottom-y, 0, y-bottom-self.resolution)
                    if dx*dx + dy*dy <= radius*radius:
                        return False
        return True

    def connected_segment(self, a, b):
        steps = max(1, math.ceil(math.dist(a[:2], b[:2]) / min(.05, self.resolution/2)))
        previous = None
        for i in range(steps+1):
            point = [a[0]+(b[0]-a[0])*i/steps, a[1]+(b[1]-a[1])*i/steps]
            if not self.walkable(point):
                return False
            x,y = self.cell(point)
            if previous is not None:
                px,py = previous
                if x != px and y != py and (self.raw[py,x] or self.raw[y,px]):
                    return False
            previous = x,y
        return True

    def connectors(self, point):
        """Visible local grid centres; never project to a distant free cell."""
        x, y = self.cell(point)
        if not self.walkable(point):
            return []
        result = []
        for cy in range(max(0, y-1), min(self.height, y+2)):
            for cx in range(max(0, x-1), min(self.width, x+2)):
                centre = [self.xx[cy, cx], self.yy[cy, cx]]
                if not self.blocked[cy, cx] and self.connected_segment(point, centre):
                    result.append((cy, cx, math.dist(point[:2], centre)))
        return result

    def field(self, seeds):
        distance = np.asarray(seeds, dtype=float).copy()
        distance[self.blocked] = np.inf
        heap = [(float(distance[y, x]), y, x) for y, x in np.argwhere(np.isfinite(distance))]
        from heapq import heapify
        heapify(heap)
        while heap:
            cost, y, x = heappop(heap)
            if cost > distance[y, x]:
                continue
            for dy, dx in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
                ny, nx = y+dy, x+dx
                if not (0 <= nx < self.width and 0 <= ny < self.height) or self.blocked[ny, nx]:
                    continue
                if dx and dy and (self.blocked[y, nx] or self.blocked[ny, x]):
                    continue
                candidate = cost + self.resolution * (math.sqrt(2) if dx and dy else 1)
                if candidate < distance[ny, nx]:
                    distance[ny, nx] = candidate
                    heappush(heap, (candidate, ny, nx))
        return distance

    def query(self, distance, point):
        return min((float(distance[y,x])+offset for y,x,offset in self.connectors(point)), default=math.inf)

    def between(self, a, b):
        if self.connected_segment(a, b):
            return math.dist(a[:2], b[:2])
        seeds = np.full(self.raw.shape, np.inf)
        for y, x, offset in self.connectors(b):
            seeds[y,x] = offset
        return self.query(self.field(seeds), a)

    def save(self, path):
        np.savez_compressed(path, blocked=self.raw, origin=self.origin,
                            resolution=self.resolution, radius=self.radius,
                            door_blocked=self.doors, carried_radius=self.carried_radius)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data['blocked'], data['origin'], data['resolution'], data['radius'],
                       data['door_blocked'] if 'door_blocked' in data else None,
                       data['carried_radius'] if 'carried_radius' in data else 0)


def box_distance(point, box):
    return math.hypot(*(max(box['min'][i]-point[i], 0, point[i]-box['max'][i]) for i in (0,1)))


class OrderedRoute:
    """Backward dynamic programming finds a shortest route through goal regions.

    Choosing each nearest region endpoint greedily can worsen the next leg;
    terminal costs include all later regions before Dijkstra propagation.
    """
    def __init__(self, grid, goals):
        self.grid, self.goals = grid, goals
        self.boxes, self.tolerances = [], []
        self.masks, self.single, self.chain = [], [], [None] * len(goals)
        for goal in goals:
            box = (goal['fixed_target_geometry']['world_aabb'] if goal['goal_kind']=='revisit_place'
                   else goal['expected_outcome']['target_geometry']['world_aabb_xy'])
            tolerance = float(goal['success_criteria'].get('position_tolerance_m', .8))
            if tolerance < 0 or not math.isfinite(tolerance) or any(
                    len(box[k]) < 2 or not all(math.isfinite(v) for v in box[k][:2]) for k in ('min','max')):
                raise ValueError('invalid runtime goal geometry/tolerance')
            if any(box['min'][i] > box['max'][i] for i in (0,1)):
                raise ValueError('inverted runtime goal box')
            dx = np.maximum(np.maximum(box['min'][0]-grid.xx, 0), grid.xx-box['max'][0])
            dy = np.maximum(np.maximum(box['min'][1]-grid.yy, 0), grid.yy-box['max'][1])
            mask = (np.hypot(dx,dy) <= tolerance) & ~grid.blocked
            if not mask.any():
                raise ValueError('no traversable grid centre in runtime goal region')
            self.boxes.append(box); self.tolerances.append(tolerance); self.masks.append(mask)
            self.single.append(grid.field(np.where(mask, 0.0, np.inf)))
        for i in reversed(range(len(goals))):
            terminal = 0.0 if i == len(goals)-1 else self.chain[i+1]
            self.chain[i] = grid.field(np.where(self.masks[i], terminal, np.inf))

    def reached(self, point, stage):
        return box_distance(point, self.boxes[stage]) <= self.tolerances[stage]

    def distance(self, point, stage, *, remaining_route=False):
        if stage >= len(self.goals):
            return 0.0
        if self.reached(point, stage):
            return self.distance(point, stage+1, remaining_route=True) if remaining_route else 0.0
        return self.grid.query((self.chain if remaining_route else self.single)[stage], point)
