"""Reproducible grid cost-to-go for the avatar's actual pick/place mechanics.

Lengths use the configured default move step as one equivalent action; each
pick/place attempt costs one action. Turns, observations and time have no cost.
The reference is private and never supplied to the agent.
"""
from ..interaction.protocol import required_action_name as _name
from copy import deepcopy
from collections import OrderedDict
import hashlib
import json
import math

import numpy as np

from .navigation import GridDistance


TRANSFER_POLICY = {
    'reference_version': 1, 'cost_unit': 'default_move_step_equivalent_action',
    'pick_attempt_cost': 1, 'place_attempt_cost': 1,
    'placement_heading_samples': 72,
    'progress': 'clip(1-remaining_cost/initial_cost,0,1)',
    'efficiency': 'initial_cost/max(initial_cost,charged_cost+remaining_cost)',
}


def _box(obj):
    return np.asarray([obj['geometry']['world_aabb'][k] for k in ('min', 'max')], dtype=float)


def _pivot(obj):
    value = np.asarray(obj.get('interaction_position'), dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('transfer reference requires actual object interaction_position')
    return value


def _held(state):
    value = state.get('held_object')
    if any(obj.get('held') is not (oid == value) for oid, obj in state['objects'].items()):
        raise ValueError('transfer held flags disagree with held_object')
    return value


def collision_digest(snapshot):
    """Audit each actual map transition without storing a full grid per action."""
    if isinstance(snapshot, GridDistance):
        raw, doors = snapshot.raw, snapshot.doors
        origin, resolution, radius, carried = snapshot.origin, snapshot.resolution, snapshot.radius, snapshot.carried_radius
    else:
        raw = np.asarray(snapshot['blocked'], dtype=bool)
        doors = np.asarray(snapshot.get('door_blocked', np.zeros_like(raw)), dtype=bool)
        origin, resolution, radius, carried = snapshot['origin'], snapshot['resolution'], snapshot['radius'], snapshot.get('carried_radius', 0)
    metadata = {'shape': list(raw.shape), 'origin': np.round(origin, 9).tolist(),
        'resolution': round(float(resolution), 9), 'radius': round(float(radius), 9),
        'carried_radius': round(float(carried), 9)}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    digest.update(np.asarray(raw, dtype=np.uint8).tobytes())
    digest.update(np.asarray(doors, dtype=np.uint8).tobytes())
    return digest.hexdigest()


class TransferReference(GridDistance):
    """Task-start grid plus the removable target and measured interaction data."""

    def __init__(self, navigation, base_blocked, room_labels, metadata):
        super().__init__(**navigation)
        self.base = np.asarray(base_blocked, dtype=bool)
        self.labels = np.asarray(room_labels, dtype=int)
        self.metadata = deepcopy(metadata)
        self.step = float(metadata['move_step_m'])
        self.pick_radius = float(metadata['pick_radius_m'])
        self.place_offset = float(metadata['place_offset_m'])
        self.floor = float(metadata['placement_floor_m'])
        self.place_height = float(metadata['placement_height_m'])
        self.room_ids = metadata['room_ids']
        self.target = metadata['target_id']
        if (metadata.get('reference_version') != TRANSFER_POLICY['reference_version']
                or self.base.shape != self.raw.shape or self.labels.shape != self.raw.shape
                or not all(math.isfinite(v) and v > 0 for v in (self.step, self.pick_radius, self.place_offset))
                or not math.isfinite(self.floor) or not math.isfinite(self.place_height)
                or self.place_height <= 0 or not np.isfinite(metadata['held_offset']).all()):
            raise ValueError('invalid transfer reference')
        self.initial = metadata['initial_state']
        _pivot(self.initial['objects'][self.target])
        initial_grid = self.grid(self.initial)
        if not np.array_equal(initial_grid.raw, self.raw) or not np.array_equal(initial_grid.blocked, self.blocked):
            raise ValueError('transfer reference does not reproduce actual task-start collision map')
        self._fields = OrderedDict()

    def save(self, path):
        np.savez_compressed(path, blocked=self.raw, origin=self.origin,
            resolution=self.resolution, radius=self.radius, door_blocked=self.doors,
            carried_radius=self.carried_radius, base_blocked=self.base,
            room_labels=self.labels, transfer_metadata=json.dumps(self.metadata, allow_nan=False))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            navigation = {k: data[k].copy() for k in
                ('blocked', 'origin', 'resolution', 'radius', 'door_blocked', 'carried_radius')}
            return cls(navigation, data['base_blocked'], data['room_labels'],
                       json.loads(str(data['transfer_metadata'])))

    def grid(self, state, *, carrying=False):
        """Reinsert the released target without erasing overlapping obstacles."""
        held = _held(state)
        if held not in (None, self.target):
            raise ValueError('transfer reference cannot plan while holding an unrelated object')
        obj = state['objects'][self.target]
        raw = self.base.copy()
        if held != self.target and not carrying:
            box = _box(obj)
            lower = np.maximum(0, np.floor((box[0, :2]-self.origin)/self.resolution).astype(int))
            upper = np.minimum([self.width-1, self.height-1],
                np.ceil((box[1, :2]-self.origin)/self.resolution).astype(int))
            if np.all(lower <= upper):
                raw[lower[1]:upper[1]+1, lower[0]:upper[0]+1] = True
        radius = 0.0
        if carrying or held == self.target:
            size = _box(obj)[1]-_box(obj)[0]
            radius = max(self.radius, .5*math.hypot(*size[:2]))
        return GridDistance(raw, self.origin, self.resolution, self.radius, self.doors, radius)

    def room_labels_at(self, x, y):
        cx = np.floor((np.asarray(x)-self.origin[0])/self.resolution).astype(int)
        cy = np.floor((np.asarray(y)-self.origin[1])/self.resolution).astype(int)
        valid = (cx >= 0) & (cy >= 0) & (cx < self.width) & (cy < self.height)
        return np.where(valid, self.labels[np.clip(cy, 0, self.height-1),
                                          np.clip(cx, 0, self.width-1)], 0)

    def placement_mask(self, obj, room, x, y, yaw=None):
        """Test the same pivot offset, room readback and floor rule as place."""
        if room not in self.room_ids:
            raise ValueError('transfer destination missing from room labels')
        box, pivot = _box(obj), _pivot(obj)
        offset = (box[0]+box[1])/2-pivot
        bottom = self.floor + max(.025, self.place_height/2) + box[0, 2]-pivot[2]
        floor = self.metadata['floor_heights'].get(room)
        if floor is None:
            raise ValueError('transfer destination floor evidence missing')
        label = self.room_ids.index(room)+1
        mask = np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape, dtype=bool)
        if abs(bottom-float(floor)) > .04:
            return mask
        angles = np.arange(TRANSFER_POLICY['placement_heading_samples'])*2*math.pi/TRANSFER_POLICY['placement_heading_samples']
        if yaw is not None:
            angles = np.append(angles, math.radians(yaw))
        for angle in angles:
            destination = self.room_labels_at(np.asarray(x)+self.place_offset*math.cos(angle)+offset[0],
                                             np.asarray(y)+self.place_offset*math.sin(angle)+offset[1])
            # Runtime readback preserves the assigned room if the centre has no label.
            mask |= (destination == label) | (destination == 0)
        return mask & (self.room_labels_at(x, y) == label)

    def completed(self, goal, state, placed):
        obj = state['objects'][self.target]
        room = goal['success_criteria']['room_id']
        return (placed and _held(state) != self.target and obj['room_id'] == room
                and abs(_box(obj)[0, 2]-self.metadata['floor_heights'][room]) <= .04)

    def remaining(self, goal, state, placed=False):
        if self.completed(goal, state, placed):
            return 0.0
        obj, point = state['objects'][self.target], state['robot_pose']['position'][:2]
        box, pivot = _box(obj), _pivot(obj)
        held = _held(state) == self.target
        # Translation does not change carry geometry. Cache fields while walking.
        key = (held, tuple(np.round((box-pivot).ravel(), 8)),
               () if held else tuple(pivot), goal['success_criteria']['room_id'])
        if key not in self._fields:
            carry = self.grid(state, carrying=True)
            place = self.placement_mask(obj, goal['success_criteria']['room_id'], self.xx, self.yy)
            delivery = carry.field(np.where(place & ~carry.blocked, 0.0, np.inf))
            approach = self.grid(state)
            grasp = (np.hypot(self.xx-pivot[0], self.yy-pivot[1]) <= self.pick_radius)
            seeds = np.where(grasp & ~carry.blocked, delivery+2*self.step, np.inf)
            pickup = approach.field(seeds) if not held else None
            self._fields[key] = carry, delivery, approach, pickup
            if len(self._fields) > 4:
                self._fields.popitem(last=False)
        self._fields.move_to_end(key)
        carry, delivery, approach, pickup = self._fields[key]
        distance = carry.query(delivery, point)
        if carry.walkable(point) and self.placement_mask(obj, goal['success_criteria']['room_id'],
                *point, yaw=state['robot_pose'].get('yaw_degrees')):
            distance = 0.0
        if held:
            return distance/self.step+1
        cost = approach.query(pickup, point)/self.step
        # A released large object can overlap the avatar. Pick itself has no
        # collision test; removing that target may make the current pose free.
        if math.dist(point, pivot[:2]) <= self.pick_radius and carry.walkable(point):
            cost = min(cost, distance/self.step+2)
        return cost

    def validate_state(self, state):
        if set(state['objects']) != set(self.initial['objects']):
            raise ValueError('transfer object inventory changed')
        _held(state)
        for oid, original in self.initial['objects'].items():
            if oid == self.target:
                continue
            obj = state['objects'][oid]
            if any(obj.get(k) != original.get(k) for k in
                   ('geometry', 'held', 'room_id', 'door_open', 'interaction_position')):
                raise ValueError('transfer background world changed: '+oid)
        _pivot(state['objects'][self.target])
        if _held(state) == self.target:
            pose = state['robot_pose']
            if 'yaw_degrees' not in pose:
                raise ValueError('transfer held pose lacks actual yaw evidence')
            angle = math.radians(pose['yaw_degrees'])
            forward, right, vertical = self.metadata['held_offset']
            expected = np.asarray(pose['position']) + [forward*math.cos(angle)-right*math.sin(angle),
                forward*math.sin(angle)+right*math.cos(angle), vertical]
            if not np.allclose(_pivot(state['objects'][self.target]), expected, atol=1e-5, rtol=0):
                raise ValueError('transfer held object does not match the verified attachment model')


def transfer_metric(goal, reference, before, after, actions, expected_success):
    """Replay actual deltas; final state and remaining work can lose progress."""
    oid = goal['target']['object_ids'][0]
    if oid != reference.target or oid not in reference.metadata['pickable_ids']:
        raise ValueError('transfer target lacks a verified pickable reference')
    reference.validate_state(before)
    for key in ('objects', 'held_object', 'robot_pose', 'floor_heights'):
        if before.get(key) != reference.initial.get(key):
            raise ValueError('transfer reference differs from actual initial state: '+key)
    initial = reference.remaining(goal, before)
    if not math.isfinite(initial) or initial <= 0:
        raise ValueError('initial transfer route is unreachable or has no valid placement')
    state, placed, walk, reset_cost, attempts, resets = deepcopy(before), False, 0.0, 0.0, 0, 0
    timeline = []
    def record(index):
        remaining = reference.remaining(goal, state, placed)
        charged = (walk+reset_cost)/reference.step+attempts
        progress = max(0.0, min(1.0, 1-remaining/initial))
        efficiency = initial/max(initial, charged+remaining)
        timeline.append({'action_index': index, 'remaining_cost': remaining if math.isfinite(remaining) else None,
            'remaining_reachable': math.isfinite(remaining), 'charged_cost': charged,
            'progress_score': 100*progress, 'path_efficiency': efficiency, 'score': 100*progress*efficiency})
    record(-1)
    for index, event in enumerate(actions):
        if state['robot_pose'] != event['before']:
            raise ValueError('transfer trajectory is discontinuous')
        previous = deepcopy(state)
        grid = reference.grid(state)
        if event.get('transfer_collision_before_sha256') != collision_digest(grid):
            raise ValueError('transfer before-action collision map evidence missing or inconsistent')
        displacement = math.dist(event['before']['position'][:2], event['after']['position'][:2])
        name = _name(event)
        result = event['result']
        if name == 'reset':
            applied = result.get('reset_applied')
            if type(applied) is not bool or (not applied and displacement > 1e-6):
                raise ValueError('transfer reset lacks actual execution evidence')
            if applied:
                distance = grid.between(event['before']['position'][:2], event['after']['position'][:2])
                if not math.isfinite(distance):
                    raise ValueError('transfer reset endpoints are disconnected')
                reset_cost += max(distance, displacement, grid.resolution)
                resets += 1
        else:
            if displacement > 1e-6 and not grid.connected_segment(event['before']['position'][:2], event['after']['position'][:2]):
                raise ValueError('transfer movement crosses actual collision map')
            walk += displacement
        if name in {'pick', 'place'}:
            attempts += 1
        for delta in event.get('state_changes', {}).get('objects', []):
            obj = state['objects'][delta['object_id']]
            for key, change in delta['changes'].items():
                if obj.get(key) != change['before']:
                    raise ValueError('transfer object delta has conflicting before state')
                obj[key] = deepcopy(change['after'])
        held = event.get('state_changes', {}).get('held_object')
        if held:
            if state.get('held_object') != held['before']:
                raise ValueError('transfer held delta has conflicting before state')
            state['held_object'] = held['after']
        state['robot_pose'] = deepcopy(event['after'])
        controlled = _held(previous) == oid or (name == 'pick' and result.get('success') is True and result.get('object_id') == oid)
        changed = previous['objects'][oid] != state['objects'][oid]
        if changed and not controlled:
            raise ValueError('transfer target changed without actual control evidence')
        if name == 'pick' and result.get('success') is True:
            if result.get('object_id') != oid or _held(previous) is not None or _held(state) != oid:
                raise ValueError('transfer pick evidence conflicts with actual held state')
            if math.dist(event['before']['position'][:2], _pivot(previous['objects'][oid])[:2]) > reference.pick_radius+1e-6:
                raise ValueError('transfer pick is outside the verified interaction range')
        if name == 'place' and result.get('success') is True:
            if result.get('object_id') != oid or _held(previous) != oid or _held(state) is not None:
                raise ValueError('transfer place evidence conflicts with actual held state')
            pose = event['before']
            if 'yaw_degrees' not in pose:
                raise ValueError('transfer place lacks actual yaw evidence')
            angle = math.radians(pose['yaw_degrees'])
            expected = np.asarray(pose['position']) + [reference.place_offset*math.cos(angle),
                reference.place_offset*math.sin(angle), 0.0]
            expected[2] = reference.floor + max(.025, reference.place_height/2)
            if not np.allclose(_pivot(state['objects'][oid]), expected, atol=1e-5, rtol=0):
                raise ValueError('transfer place does not match the verified placement model')
            placed = True
        reference.validate_state(state)
        if event.get('transfer_collision_after_sha256') != collision_digest(reference.grid(state)):
            raise ValueError('transfer after-action collision map evidence missing or inconsistent')
        record(index)
    if any(state.get(k) != after.get(k) for k in ('objects', 'held_object', 'robot_pose', 'floor_heights')):
        raise ValueError('transfer final state disagrees with actual action deltas')
    if reference.completed(goal, state, placed) != expected_success:
        raise ValueError('transfer completion disagrees with success evidence')
    return {'goal_id': goal['goal_id'], 'cost_unit': TRANSFER_POLICY['cost_unit'],
        'reference_cost': initial, **timeline[-1], 'walking_length_m': walk,
        'reset_cost_m': reset_cost, 'reset_count': resets, 'interaction_attempts': attempts,
        'move_step_m': reference.step, 'best_progress_score': max(r['progress_score'] for r in timeline),
        'timeline': timeline}
