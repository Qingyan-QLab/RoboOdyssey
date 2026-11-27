"""CPU grid geodesics from a saved, actual task-start collision map.

Distances are shortest paths on an eight-neighbour grid, not Isaac navmesh
distances. No diagonal corner cutting or snapping through obstacles is allowed.
"""
from heapq import heappop, heappush
from copy import deepcopy
import json
import math
import numpy as np


NAVIGATION_REFERENCE_VERSION = 'grid_distance_v2_obstacle_aware_arrival'

class GridDistance:
    def __init__(self, blocked, origin, resolution, radius=0.0, door_blocked=None,
                 carried_radius=0.0, target_blocked=None):
        self.raw = np.asarray(blocked, dtype=bool)
        self.origin = np.asarray(origin, dtype=float)[:2]
        self.resolution, self.radius = float(resolution), float(radius)
        self.doors = np.zeros_like(self.raw) if door_blocked is None else np.asarray(door_blocked,dtype=bool)
        self.carried_radius = float(carried_radius)
        self.target_blocked = {str(key): np.asarray(value, dtype=bool)
                               for key, value in (target_blocked or {}).items()}
        if (self.raw.ndim != 2 or not self.raw.size or self.origin.shape != (2,)
                or not np.isfinite(self.origin).all() or not math.isfinite(self.resolution)
                or not math.isfinite(self.radius) or self.resolution <= 0 or self.radius < 0
                or self.doors.shape != self.raw.shape or not math.isfinite(self.carried_radius)
                or self.carried_radius < 0):
            raise ValueError('invalid navigation snapshot')
        if any(value.shape != self.raw.shape for value in self.target_blocked.values()):
            raise ValueError('target approach grid shape differs from navigation map')
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

    def unobstructed_segment(self, a, b, occupied):
        """Test every occupied cell intersected by a geometric segment.

        The target-only occupancy exclusion is supplied by the environment.
        This method never clears an AABB in the collision map. Closed-cell
        intersection also forbids slipping through the corner of two walls.
        """
        a, b = np.asarray(a, dtype=float)[:2], np.asarray(b, dtype=float)[:2]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return False
        lo = np.floor((np.minimum(a, b)-self.origin)/self.resolution).astype(int)
        hi = np.floor((np.maximum(a, b)-self.origin)/self.resolution).astype(int)
        if (lo < 0).any() or hi[0] >= self.width or hi[1] >= self.height:
            return False
        # Include adjacent cells when the ray lies exactly on a cell edge.
        lo = np.maximum(lo-1, 0)
        hi = np.minimum(hi+1, [self.width-1, self.height-1])
        delta = b-a
        for cy, cx in np.argwhere(occupied[lo[1]:hi[1]+1, lo[0]:hi[0]+1]):
            cx, cy = int(cx+lo[0]), int(cy+lo[1])
            lower = self.origin+np.array([cx, cy])*self.resolution
            upper = lower+self.resolution
            enter, leave = 0.0, 1.0
            for axis in (0, 1):
                if abs(delta[axis]) < 1e-12:
                    if a[axis] < lower[axis]-1e-12 or a[axis] > upper[axis]+1e-12:
                        enter, leave = 1.0, 0.0
                        break
                else:
                    first = (lower[axis]-a[axis])/delta[axis]
                    last = (upper[axis]-a[axis])/delta[axis]
                    enter = max(enter, min(first, last))
                    leave = min(leave, max(first, last))
            if enter <= leave+1e-12:
                return False
        return True

    def unobstructed_corridor(self, a, b, occupied):
        """Verify a robot-width contact corridor, ending at the target surface.

        A shortened sweep can miss a slit immediately in front of the target.
        A full capsule instead reaches behind it and rejects small objects
        backed by a wall. The flat-ended rectangle checks the whole approach
        while excluding space beyond the contacted surface. This is a target-
        height geometric relation; it does not authorize physical movement.
        """
        a, b = np.asarray(a, dtype=float)[:2], np.asarray(b, dtype=float)[:2]
        length = math.dist(a, b)
        if length <= 1e-12 or self.radius == 0:
            return self.unobstructed_segment(a, b, occupied)
        direction = (b-a)/length
        normal = np.array([-direction[1], direction[0]])
        corners = np.asarray([a+normal*self.radius, a-normal*self.radius,
                              b+normal*self.radius, b-normal*self.radius])
        minimum, maximum = corners.min(axis=0), corners.max(axis=0)
        if ((minimum < self.origin).any()
                or (maximum >= self.origin+np.array([self.width, self.height])*self.resolution).any()):
            return False
        lo = np.maximum(np.floor((minimum-self.origin)/self.resolution).astype(int)-1, 0)
        hi = np.minimum(np.floor((maximum-self.origin)/self.resolution).astype(int)+1,
                        [self.width-1, self.height-1])
        centre = (a+b)/2
        # Separating-axis theorem for the oriented corridor and an axis-aligned
        # occupied cell. Touching a boundary counts as blocked, including the
        # diagonal corner case. No step-size sampling can tunnel through cells.
        axes = [direction, normal, np.array([1., 0.]), np.array([0., 1.])]
        extents = [length/2*abs(np.dot(direction, axis))
                   + self.radius*abs(np.dot(normal, axis))
                   + self.resolution/2*(abs(axis[0])+abs(axis[1])) for axis in axes]
        for cy, cx in np.argwhere(occupied[lo[1]:hi[1]+1, lo[0]:hi[0]+1]):
            cell_centre = self.origin+np.array([cx+lo[0]+.5, cy+lo[1]+.5])*self.resolution
            delta = cell_centre-centre
            if all(abs(np.dot(delta, axis)) <= extent+1e-12 for axis, extent in zip(axes, extents)):
                return False
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

    def next_waypoint(self, distance, point):
        """Return one collision-checked descent waypoint, or None when unreachable.

        Callers decide task/phase completion from actual feedback. The field
        can include terminal costs for later stages and need not end at zero.
        """
        field = np.asarray(distance, dtype=float)
        if field.shape != self.raw.shape:
            raise ValueError('distance field shape differs from navigation map')
        choices = [(float(field[y, x])+offset, offset, y, x)
                   for y, x, offset in self.connectors(point) if math.isfinite(field[y, x])]
        if not choices:
            return None
        _, offset, y, x = min(choices)
        if offset > 1e-6:
            return [float(self.xx[y, x]), float(self.yy[y, x])]
        lower = []
        for dy, dx in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
            ny, nx = y+dy, x+dx
            if not (0 <= nx < self.width and 0 <= ny < self.height) or field[ny, nx] >= field[y, x]-1e-9:
                continue
            target = [float(self.xx[ny, nx]), float(self.yy[ny, nx])]
            if self.connected_segment(point, target):
                lower.append((float(field[ny, nx])+math.dist(point[:2], target), ny, nx, target))
        return min(lower)[-1] if lower else [float(self.xx[y, x]), float(self.yy[y, x])]

    def between(self, a, b):
        if self.connected_segment(a, b):
            return math.dist(a[:2], b[:2])
        seeds = np.full(self.raw.shape, np.inf)
        for y, x, offset in self.connectors(b):
            seeds[y,x] = offset
        return self.query(self.field(seeds), a)

    def save(self, path):
        target_ids = sorted(self.target_blocked)
        np.savez_compressed(path, blocked=self.raw, origin=self.origin,
                            resolution=self.resolution, radius=self.radius,
                            door_blocked=self.doors, carried_radius=self.carried_radius,
                            target_ids_json=json.dumps(target_ids),
                            **{f'target_blocked_{i}': self.target_blocked[key]
                               for i, key in enumerate(target_ids)})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            target_ids = json.loads(str(data['target_ids_json'])) if 'target_ids_json' in data else []
            return cls(data['blocked'], data['origin'], data['resolution'], data['radius'],
                       data['door_blocked'] if 'door_blocked' in data else None,
                       data['carried_radius'] if 'carried_radius' in data else 0,
                       {key: data[f'target_blocked_{i}'] for i, key in enumerate(target_ids)})


def box_distance(point, box):
    return math.hypot(*(max(box['min'][i]-point[i], 0, point[i]-box['max'][i]) for i in (0,1)))


class GoalRegion:
    """One authoritative arrival predicate and its grid representation.

    Object visits require a collision-free standing pose and an unobstructed
    local approach to a target surface. Named places instead use distance to
    the original place through the navigable space. Tolerance belongs to this
    region; route distances must not apply it a second time.
    """
    def __init__(self, grid, goal):
        self.grid = grid
        self.is_place = goal['goal_kind'] == 'revisit_place'
        self.box = deepcopy(goal['fixed_target_geometry']['world_aabb'] if self.is_place
                            else goal['expected_outcome']['target_geometry']['world_aabb_xy'])
        self.tolerance = float(goal['success_criteria'].get('position_tolerance_m', .8))
        if self.tolerance < 0 or not math.isfinite(self.tolerance) or any(
                len(self.box[key]) < 2 or not all(math.isfinite(v) for v in self.box[key][:2])
                for key in ('min', 'max')):
            raise ValueError('invalid runtime goal geometry/tolerance')
        if any(self.box['min'][i] > self.box['max'][i] for i in (0, 1)):
            raise ValueError('inverted runtime goal box')
        self.lower = np.asarray(self.box['min'][:2], dtype=float)
        self.upper = np.asarray(self.box['max'][:2], dtype=float)
        oid = goal.get('target', {}).get('object_ids', [None])[0]
        self.approach_blocked = grid.target_blocked.get(oid, grid.raw)
        self.place_field = self._place_field() if self.is_place else None
        self.surfaces = self._surface_points()
        dx = np.maximum(np.maximum(self.lower[0]-grid.xx, 0), grid.xx-self.upper[0])
        dy = np.maximum(np.maximum(self.lower[1]-grid.yy, 0), grid.yy-self.upper[1])
        candidates = (np.hypot(dx, dy) <= self.tolerance+1e-9) & ~grid.blocked
        self.mask = np.zeros_like(grid.raw)
        for y, x in np.argwhere(candidates):
            self.mask[y, x] = self.reached([grid.xx[y, x], grid.yy[y, x]])

    def _surface_points(self):
        points = []
        for axis in (0, 1):
            other = 1-axis
            count = max(1, math.ceil((self.upper[other]-self.lower[other]) /
                                     (self.grid.resolution/2)))
            for fixed in (self.lower[axis], self.upper[axis]):
                for value in np.linspace(self.lower[other], self.upper[other], count+1):
                    p = np.zeros(2)
                    p[axis], p[other] = fixed, value
                    points.append(p)
        return points

    def _place_field(self):
        grid = self.grid
        seeds = np.full(grid.raw.shape, np.inf)
        # Connect the original, undilated place to local grid centres. Building
        # seeds from a Euclidean tolerance band would reintroduce wall leakage.
        margin = grid.resolution*1.5
        candidates = ((grid.xx >= self.lower[0]-margin) & (grid.xx <= self.upper[0]+margin)
                      & (grid.yy >= self.lower[1]-margin) & (grid.yy <= self.upper[1]+margin)
                      & ~grid.blocked)
        for y, x in np.argwhere(candidates):
            point = np.array([grid.xx[y, x], grid.yy[y, x]])
            anchor = np.clip(point, self.lower, self.upper)
            if grid.connected_segment(point, anchor):
                seeds[y, x] = math.dist(point, anchor)
        return grid.field(seeds)

    def place_distance(self, point):
        point = np.asarray(point, dtype=float)[:2]
        anchor = np.clip(point, self.lower, self.upper)
        if self.grid.connected_segment(point, anchor):
            return math.dist(point, anchor)
        return self.grid.query(self.place_field, point)

    def reached(self, point):
        point = np.asarray(point, dtype=float)[:2]
        if (point.shape != (2,) or not np.isfinite(point).all()
                or box_distance(point, self.box) > self.tolerance+1e-9
                or not self.grid.walkable(point)):
            return False
        if self.is_place:
            return self.place_distance(point) <= self.tolerance+1e-9
        closest = np.clip(point, self.lower, self.upper)
        # Test the nearest surface first. Other sampled surfaces allow a clear
        # approach around a partly occluded object, without requiring the
        # camera to be aimed at it.
        for target in [closest, *self.surfaces]:
            length = math.dist(point, target)
            if length > self.tolerance+1e-9:
                continue
            if not self.grid.unobstructed_segment(point, target, self.approach_blocked):
                continue
            # The target-height layer may omit the supporting tabletop. This
            # contact corridor never replaces full-map standing/motion checks.
            if self.grid.unobstructed_corridor(point, target, self.approach_blocked):
                return True
        return False


class OrderedRoute:
    """Backward dynamic programming finds a shortest route through goal regions.

    Choosing each nearest region endpoint greedily can worsen the next leg;
    terminal costs include all later regions before Dijkstra propagation.
    """
    def __init__(self, grid, goals):
        self.grid, self.goals = grid, deepcopy(goals)
        self.boxes, self.tolerances = [], []
        self.regions = []
        self.masks, self.single, self.chain = [], [], [None] * len(self.goals)
        for goal in self.goals:
            region = GoalRegion(grid, goal)
            self.regions.append(region)
            mask = region.mask
            self.boxes.append(region.box); self.tolerances.append(region.tolerance); self.masks.append(mask)
            self.single.append(grid.field(np.where(mask, 0.0, np.inf)))
        for i in reversed(range(len(self.goals))):
            terminal = 0.0 if i == len(self.goals)-1 else self.chain[i+1]
            self.chain[i] = grid.field(np.where(self.masks[i], terminal, np.inf))

    def reached(self, point, stage):
        return self.regions[stage].reached(point)

    def distance(self, point, stage, *, remaining_route=False):
        if stage >= len(self.goals):
            return 0.0
        if self.reached(point, stage):
            return self.distance(point, stage+1, remaining_route=True) if remaining_route else 0.0
        return self.grid.query((self.chain if remaining_route else self.single)[stage], point)
