"""CPU expected-state projection using common geometry, never execution evidence.

This conservative grid/AABB model does not model physical grasps or the avatar's
fixed-offset place action. Its planned paths/costs are not formal score references.
"""
from copy import deepcopy
import math

from .snapshots import StateBranch
from .contextual import instantiate_request_v3
from .navigation import navigation_task
from .placement import floor_placement
from .raster import derive_navigation, derive_arrival_navigation
from .state import geometry
from .navigation_io import read_navigation_npz

PROJECTION_VERSION = 'grid_aabb_expected_success_v2'


class PlannedProjection:
    def __init__(self, sequence):
        from ..data import load_sequence
        from ..data.frozen import read
        from ..scenes.dataset_loader import BenchmarkDataset
        self.sequence = load_sequence(sequence)
        self.metadata = read(self.sequence.path('scene_metainfo'))
        self.cache = read(self.sequence.path('runtime_scene_cache'))
        self.navigation = read_navigation_npz(self.sequence.path('navigation'))
        from ..scenes.robot_body import scene_robot_identity
        declared = scene_robot_identity(self.sequence)
        if declared is not None:
            self.navigation['robot_profile'] = declared
        initial = BenchmarkDataset(self.sequence).canonical_state(self.metadata['scene']['scene_id'])
        initial['held_object'] = None
        initial['robot_pose'] = deepcopy(initial['robot_base_pose'])
        initial['floor_heights'] = {room['room_id']: room.get('floor_height', 0.) for room in self.metadata['rooms']}
        self.ledger = {}
        self.branch = StateBranch(self.cache, initial, source='planned')

    def _grid(self, state, *, arrival_object_ids=None):
        entities = []
        for raw in state['objects'].values():
            entity = deepcopy(raw)
            entity['geometry'] = geometry(entity)
            entity.pop('pose', None)
            entities.append(entity)
        grid = derive_navigation(self.metadata['rooms'], entities,
            resolution=self.navigation['resolution'], robot_radius=self.navigation['robot_radius'],
            environment_setup=self.navigation.get('environment_setup'))
        grid['scene_id'] = self.metadata['scene']['scene_id']
        if self.navigation.get('robot_profile') is not None:
            grid['robot_profile'] = deepcopy(self.navigation['robot_profile'])
        if arrival_object_ids is not None:
            from ..evaluation.navigation import GridDistance
            grid['arrival_grid'] = GridDistance(**derive_arrival_navigation(
                self.metadata['rooms'], entities, arrival_object_ids,
                resolution=self.navigation['resolution'], robot_radius=self.navigation['robot_radius'],
                environment_setup=self.navigation.get('environment_setup')))
        position = state['robot_pose']['position']
        start = tuple(math.floor((position[i]-grid['origin_xy'][i])/grid['resolution']) for i in (0, 1))
        return grid, entities, start

    def _bound(self, spec):
        binding = spec.get('history_binding') or spec.get('place_binding')
        if not binding:
            return None
        matches = [event for event in self.branch.history()
                   if all(event.get(k) == binding[k] for k in ('session_id', 'task_id', 'goal_id'))]
        if len(matches) != 1:
            raise ValueError('planned history binding is missing or ambiguous')
        return matches[0]

    def _move(self, state, oid, room_id):
        grid, entities, start = self._grid(state)
        entity = next(item for item in entities if item['object_id'] == oid)
        destination = next(room for room in self.metadata['rooms'] if room['room_id'] == room_id)
        placement = floor_placement(entity, destination, entities, self.metadata['rooms'], grid, start=start)
        moved = placement['entity']
        moved.update(held=False, support_object_id=None)
        state['objects'][oid] = moved
        return placement

    def mutate(self, mutation, *, event_id):
        """Apply a planned boundary move, with conservative shared placement."""
        before = self.branch.snapshot()
        state = before.state
        if len(mutation.get('changes', [])) != 1:
            raise ValueError('projection supports one object per boundary mutation')
        change = mutation['changes'][0]
        if state['objects'][change['object_id']]['room_id'] != change['relation_before']['room_id']:
            raise ValueError('planned mutation has a stale location precondition')
        binding = mutation.get('inspection_binding')
        if not binding or not self._bound({'history_binding': binding}).get('assumed_location_experience'):
            raise ValueError('boundary projection requires a planned location experience')
        placement = self._move(state, change['object_id'], change['relation_after']['room_id'])
        grid, _, start = self._grid(state)
        x, y = start
        if not (0 <= y < len(grid['static_occupancy']) and 0 <= x < len(grid['static_occupancy'][y])) or grid['static_occupancy'][y][x] or grid['dynamic_occupancy'][y][x]:
            raise ValueError('projected boundary mutation blocks the robot')
        after = self.branch.commit(state, expected_version=before.version, event_id=event_id)
        return {'state_source': 'planned', 'event_id': event_id, 'before_version': before.version,
                'after_version': after.version, 'placement_validation': placement['validation']}

    def task(self, session_id, public, truth, template):
        """Atomically project a frozen request; errors leave the branch unchanged."""
        before = self.branch.snapshot()
        state = before.state
        navigation_ids = [goal['target']['object_ids'][0] for goal in truth['goals']
                          if goal['goal_kind'] in {'visit', 'establish_place', 'revisit_place'}]
        arrival = self._grid(state, arrival_object_ids=navigation_ids)[0]['arrival_grid'] if navigation_ids else None
        repair = instantiate_request_v3(public, truth, template, self.cache, state,
                                        self.branch.history(), self.ledger, state_source='planned',
                                        navigation=arrival)
        if repair['status'] == 'unavailable':
            raise ValueError('planned request could not be instantiated: '+str(repair['repair_record']))
        truth = repair['runtime_ground_truth']
        if repair['status'] == 'already_satisfied':
            for goal in truth['goals']:
                self.branch.append_history({'state_source': 'planned', 'session_id': session_id,
                    'task_id': public['task_id'], 'goal_id': goal['goal_id'],
                    'object_id': goal['target']['object_ids'][0], 'assumed_dispatch': False,
                    'assumed_success': None, 'reason_code': 'already_satisfied',
                    'assumed_location_experience': False, 'decision_state': before.state}, source='planned')
            return {'session_id': session_id, 'task_id': public['task_id'], 'state_source': 'planned',
                    'status': 'already_satisfied', 'before': before.state, 'after': before.state,
                    'before_version': before.version, 'after_version': before.version,
                    'goals': [], 'planned_repair': repair['repair_record']}
        goals, events = [], []
        for goal, spec in zip(truth['goals'], template['goals']):
            oid, kind = goal['target']['object_ids'][0], goal['goal_kind']
            bound = self._bound(spec)
            row = {'goal_id': goal['goal_id'], 'goal_kind': kind, 'object_id': oid,
                   'state_source': 'planned', 'paths': [], 'cost_unit': 'm'}
            entity = deepcopy(state['objects'][oid])
            entity['geometry'] = geometry(entity)
            if kind in {'visit', 'establish_place', 'revisit_place'}:
                if kind == 'revisit_place':
                    if bound is None or 'planned_place_geometry' not in bound:
                        raise ValueError('planned named place is unavailable')
                    entity['geometry'] = deepcopy(bound['planned_place_geometry'])
                    entity.pop('pose', None)
                grid, _, start = self._grid(state, arrival_object_ids=[oid])
                route = navigation_task(entity, grid, 0, start=start,
                    arrival_grid=grid['arrival_grid'], arrival_goal=goal,
                    start_position=state['robot_pose']['position'])
                if route is None:
                    raise ValueError('goal has no route in the declared grid model')
                path = route[1]['reference_trajectory']
                row['paths'].append(path)
                state['robot_pose']['position'] = deepcopy(path['points'][-1]['position'])
                if kind == 'establish_place':
                    position = state['robot_pose']['position']
                    row['planned_place_geometry'] = {'world_aabb': {
                        'min': [value-.05 for value in position],
                        'max': [value+.05 for value in position]}}
            elif kind == 'transfer':
                rule = spec['contract']
                room = goal['success_criteria']['room_id']
                placement = self._move(state, oid, room)
                row['paths'] = [placement['pickup_trajectory'], placement['placement_trajectory']]
                row['placement_validation'] = placement['validation']
                state['robot_pose']['position'] = deepcopy(placement['placement_trajectory']['points'][-1]['position'])
                row['destination_room_id'] = room
            elif kind == 'answer':
                outcome = goal['expected_outcome']
                row['expected_answer'] = outcome['answer'] if isinstance(outcome['answer'], bool) else outcome['accepted_answers'][0]
            else:
                raise ValueError('unsupported projected goal: '+kind)
            row['path_length_m'] = sum(path['length_m'] for path in row['paths'])
            goals.append(row)
            event = {'state_source': 'planned', 'session_id': session_id, 'task_id': public['task_id'],
                     'goal_id': goal['goal_id'], 'object_id': oid, 'assumed_dispatch': True, 'assumed_success': True,
                     'assumed_location_experience': kind != 'answer', 'before': before.state}
            if 'planned_place_geometry' in row:
                event['planned_place_geometry'] = deepcopy(row['planned_place_geometry'])
                event['assumed_place_established'] = True
            events.append(event)
        after = self.branch.commit(state, expected_version=before.version, event_id=session_id+'/'+public['task_id'])
        self.ledger = repair['name_ledger']
        for event in events:
            event['after'] = after.state
            self.branch.append_history(event, source='planned')
        return {'session_id': session_id, 'task_id': public['task_id'], 'state_source': 'planned',
                'status': 'projected', 'before': before.state, 'after': after.state,
                'before_version': before.version, 'after_version': after.version, 'goals': goals,
                'planned_repair': repair['repair_record']}


def project_sequence(sequence, *, scene=None):
    """Return a fresh CPU plan, halting explicitly if its model cannot project."""
    from ..data import load_sequence
    from ..data.frozen import read, inside
    source = load_sequence(sequence, scene=scene)
    projection = PlannedProjection(source)
    tasks, mutations = [], []
    value = {'schema_version': 1, 'state_source': 'planned', 'model': PROJECTION_VERSION,
             'physics_execution_verified': False, 'formal_score_reference': False,
             'sequence_identity': {key: source.manifest[key] for key in ('scene_id', 'sequence_id', 'scene_reference')},
             'tasks': tasks, 'mutations': mutations, 'status': 'projected'}
    for session in source.manifest['sessions']:
        sid = session['session_id']
        stopped_task = None
        try:
            for index, mutation in enumerate(read(inside(source.root, session['mutations']))):
                mutations.append(projection.mutate(mutation, event_id=f'{sid}/mutation/{index}'))
            public = read(inside(source.root, session['session']))
            truths = read(inside(source.root, session['ground_truth']))
            templates = read(inside(source.root, session['templates']))
            for task, truth, template in zip(public['tasks'], truths, templates):
                stopped_task = task['task_id']
                tasks.append(projection.task(sid, task, truth, template))
        except (ValueError, KeyError, StopIteration) as exc:
            value.update(status='indeterminate', reason=str(exc), stopped_session_id=sid, stopped_task_id=stopped_task)
            break
    value['planned_history'] = projection.branch.history()
    return value
