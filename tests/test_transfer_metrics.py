from tests.frozen_data import result_session
from benchmark.artifacts.layout import session_file
"""CPU tests against explicit grasp, attachment, placement and obstacle states."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from benchmark.evaluation.transfer import TransferReference, TRANSFER_POLICY, transfer_metric, collision_digest
from benchmark.evaluation.task_metrics import evaluate_metrics, prepare_reference, load_reference, METRIC_VERSION
from benchmark.runtime.request_events import state_changes


def pose(x, y=4.5, yaw=0):
    return {'position': [x, y, 1.5], 'yaw_degrees': yaw}


def object_at(x, y=4.5, z=.1, held=False, room='hall'):
    return {'object_id': 'book', 'category': 'book', 'held': held, 'room_id': room,
        'interaction_position': [x, y, z],
        'geometry': {'world_aabb': {'min': [x-.1, y-.1, z-.1], 'max': [x+.1, y+.1, z+.1]}}}


def fixture(start=1.5, held=False, blocked=None, labels=None, step=1):
    raw = np.zeros((9, 40), bool) if blocked is None else blocked.copy()
    labels = np.ones(raw.shape, int) if labels is None else labels.copy()
    labels[:, 25:] = 2
    state = {'objects': {'book': object_at(12.5)}, 'held_object': None,
             'robot_pose': pose(start), 'floor_heights': {'hall': 0, 'study': 0}}
    if held:
        state['held_object'] = 'book'
        state['objects']['book'] = object_at(start+.75, 4.85, 1.28, True,
            'study' if start+.75 >= 25 else 'hall')
    occupied = raw.copy()
    if not held:
        occupied[4:6, 12:14] = True
    metadata = {'reference_version': TRANSFER_POLICY['reference_version'],
        'move_step_m': step, 'pick_radius_m': 1.5, 'place_offset_m': .8,
        'held_offset': [.75, .35, -.22], 'placement_floor_m': 0,
        'placement_height_m': .2, 'floor_heights': state['floor_heights'],
        'room_ids': ['hall', 'study'], 'pickable_ids': ['book'],
        'target_id': 'book', 'initial_state': deepcopy(state)}
    snapshot = {'navigation': {'blocked': occupied, 'origin': [0, 0], 'resolution': 1, 'radius': 0},
        'base_blocked': raw, 'room_labels': labels, 'metadata': metadata}
    gt = {'task_id': 't', 'state_assumption': 'actual_predispatch_state', 'goals': [
        {'goal_id': 'g', 'goal_kind': 'transfer', 'target': {'object_ids': ['book']},
         'success_criteria': {'room_id': 'study'}}]}
    return gt, state, snapshot


def change(state, name, x=None, y=None, yaw=None, success=True):
    before = deepcopy(state)
    p = state['robot_pose']
    if x is not None:
        p['position'][0] = x
    if y is not None:
        p['position'][1] = y
    if yaw is not None:
        p['yaw_degrees'] = yaw
    angle = math.radians(p['yaw_degrees'])
    result = {'success': success}
    if name in {'pick', 'place'}:
        result['object_id'] = 'book'
    if name == 'pick' and success:
        state['held_object'] = 'book'
    if name == 'place' and success:
        px, py = p['position'][0]+.8*math.cos(angle), p['position'][1]+.8*math.sin(angle)
        state['objects']['book'] = object_at(px, py, room='study' if px >= 25 else 'hall')
        state['held_object'] = None
    if state['held_object'] == 'book':
        px = p['position'][0]+.75*math.cos(angle)-.35*math.sin(angle)
        py = p['position'][1]+.75*math.sin(angle)+.35*math.cos(angle)
        state['objects']['book'] = object_at(px, py, 1.28, True, 'study' if px >= 25 else 'hall')
    if name == 'reset':
        result['reset_applied'] = success
    return {'action': name, 'task_id': 't', 'task_step': 0,
        'before': before['robot_pose'], 'after': deepcopy(p), 'result': result,
        'state_changes': state_changes(before, state)}


def run_cost(before, snapshot, *commands):
    state, events = deepcopy(before), []
    reference = TransferReference(**snapshot)
    for name, kwargs in commands:
        digest_before = collision_digest(reference.grid(state))
        event = change(state, name, **kwargs)
        event.update(transfer_collision_before_sha256=digest_before,
            transfer_collision_after_sha256=collision_digest(reference.grid(state)))
        event['task_step'] = len(events)
        events.append(event)
    gt = fixture()[0]
    complete = reference.completed(gt['goals'][0], state,
        any(_event['action'] == 'place' and _event['result']['success'] for _event in events))
    metric = transfer_metric(gt['goals'][0], reference, before, state, events, complete)
    return metric, state, events


class TransferCostTests(unittest.TestCase):
    def test_continuous_approach_pick_carry_and_place_use_one_reference_cost(self):
        gt, before, snapshot = fixture()
        ref = TransferReference(**snapshot)
        self.assertEqual(ref.remaining(gt['goals'][0], before), 26)
        commands = [('move_forward', {'x': 6.5}), ('move_forward', {'x': 11.5}),
                    ('pick', {}), ('move_forward', {'x': 25.5}), ('place', {})]
        metric, _, _ = run_cost(before, snapshot, *commands)
        self.assertEqual(metric['score'], 100)
        self.assertEqual(metric['reference_cost'], 26)
        self.assertEqual(metric['charged_cost'], 26)
        self.assertEqual(metric['path_efficiency'], 1)
        remaining = [r['remaining_cost'] for r in metric['timeline']]
        self.assertEqual(remaining, [26, 21, 16, 15, 1, 0])
        self.assertAlmostEqual(metric['timeline'][1]['progress_score'], 500/26)
        self.assertAlmostEqual(metric['timeline'][-2]['score'], 2500/26)

    def test_detours_and_repeated_attempts_discount_the_same_final_progress(self):
        _, before, snapshot = fixture()
        direct, _, _ = run_cost(before, snapshot, ('move_forward', {'x': 6.5}))
        detour, _, _ = run_cost(before, snapshot, ('move_forward', {'x': .5}), ('move_forward', {'x': 6.5}))
        failed, _, _ = run_cost(before, snapshot, ('pick', {'success': False}), ('move_forward', {'x': 6.5}))
        self.assertEqual(direct['progress_score'], detour['progress_score'])
        self.assertGreater(direct['score'], detour['score'])
        self.assertGreater(direct['score'], failed['score'])
        completed, _, _ = run_cost(before, snapshot, ('move_backward', {'x': .5}),
            ('move_forward', {'x': 11.5}), ('pick', {}), ('move_forward', {'x': 25.5}), ('place', {}))
        self.assertEqual(completed['progress_score'], 100)
        self.assertAlmostEqual(completed['score'], 2600/28)

    def test_final_progress_can_decrease_and_pickup_is_not_permanent_credit(self):
        _, before, snapshot = fixture()
        metric, _, _ = run_cost(before, snapshot, ('move_forward', {'x': 10.5}), ('move_backward', {'x': 1.5}))
        self.assertEqual(metric['score'], 0)
        self.assertGreater(metric['best_progress_score'], 0)
        dropped, state, events = run_cost(before, snapshot,
            ('move_forward', {'x': 11.5}), ('pick', {}), ('place', {}))
        self.assertLess(dropped['timeline'][-1]['progress_score'], dropped['timeline'][-2]['progress_score'])
        self.assertEqual(state['objects']['book']['room_id'], 'hall')
        self.assertGreater(dropped['remaining_cost'], 1)

    def test_initially_held_target_omits_approach_and_pick_cost(self):
        _, before, snapshot = fixture(start=11.5, held=True)
        metric, _, _ = run_cost(before, snapshot, ('move_forward', {'x': 25.5}))
        self.assertEqual(metric['reference_cost'], 15)
        self.assertEqual(metric['remaining_cost'], 1)
        self.assertEqual(metric['interaction_attempts'], 0)
        self.assertAlmostEqual(metric['score'], 1400/15)

    def test_reset_costs_include_geodesic_teleport_and_minimum_cell_length(self):
        _, before, snapshot = fixture()
        normal, _, _ = run_cost(before, snapshot, ('move_forward', {'x': 6.5}))
        reset, _, _ = run_cost(before, snapshot, ('reset', {'x': 1.5}), ('reset', {'x': 6.5}))
        self.assertEqual(reset['reset_cost_m'], 6)
        self.assertEqual(reset['walking_length_m'], 0)
        self.assertLess(reset['score'], normal['score'])
        self.assertEqual(reset['reset_count'], 2)

    def test_walls_determine_route_and_unreachable_destinations_are_explicit(self):
        blocked = np.zeros((9, 40), bool); blocked[:8, 20] = True
        gt, before, snapshot = fixture(blocked=blocked)
        self.assertGreater(TransferReference(**snapshot).remaining(gt['goals'][0], before), 26)
        blocked[8, 20] = True
        gt, before, snapshot = fixture(blocked=blocked)
        with self.assertRaisesRegex(ValueError, 'unreachable'):
            transfer_metric(gt['goals'][0], TransferReference(**snapshot), before, before, [], False)
        result = {'goal_results': [{'goal_id': 'g', 'success': False}]}
        value = evaluate_metrics(gt, before, before, [], [], result, grid=TransferReference(**snapshot))
        self.assertIsNone(value['score'])
        self.assertEqual(value['status'], 'unavailable')

    def test_attachment_origins_room_boundaries_and_floor_constraints_are_used(self):
        gt, before, snapshot = fixture()
        reference = TransferReference(**snapshot)
        obj = before['objects']['book']
        # The robot is inside the destination but every released centre is outside it.
        shifted = deepcopy(obj)
        shifted['interaction_position'][0] += 3
        self.assertFalse(reference.placement_mask(shifted, 'study', 25.5, 4.5))
        self.assertTrue(reference.placement_mask(obj, 'study', 25.5, 4.5))
        raised = deepcopy(obj)
        raised['geometry']['world_aabb']['min'][2] += .1
        raised['geometry']['world_aabb']['max'][2] += .1
        self.assertFalse(reference.placement_mask(raised, 'study', 25.5, 4.5))
        invalid = deepcopy(before); invalid['objects']['book'].pop('interaction_position')
        with self.assertRaisesRegex(ValueError, 'interaction_position'):
            reference.remaining(gt['goals'][0], invalid)

    def test_wrong_room_release_retains_remaining_delivery_work(self):
        _, before, snapshot = fixture()
        metric, state, _ = run_cost(before, snapshot, ('move_forward', {'x': 11.5}),
            ('pick', {}), ('move_forward', {'x': 18.5}), ('place', {}))
        self.assertEqual(state['objects']['book']['room_id'], 'hall')
        self.assertGreater(metric['score'], 0)
        self.assertLess(metric['score'], 100)
        self.assertGreaterEqual(metric['remaining_cost'], 2)

    def test_overlapping_released_target_can_be_picked_up_without_moving_first(self):
        gt, before, snapshot = fixture(start=11.5, held=True)
        reference = TransferReference(**snapshot)
        after = deepcopy(before)
        change(after, 'place')
        box = after['objects']['book']['geometry']['world_aabb']
        box['min'][0] -= 1; box['max'][0] += 1
        self.assertFalse(reference.grid(after).walkable(after['robot_pose']['position']))
        self.assertTrue(math.isfinite(reference.remaining(gt['goals'][0], after, placed=True)))

    def test_unsupported_and_missing_reference_have_different_denominator_behavior(self):
        gt, before, _ = fixture()
        result = {'task_id': 't', 'goal_results': [{'goal_id': 'g', 'success': False}]}
        self.assertIsNone(evaluate_metrics(gt, before, before, [], [], result)['score'])
        self.assertEqual(evaluate_metrics(gt, before, before, [], [], result | {'agent_status': 'unsupported'})['score'], 0)

    def test_control_trajectory_and_reference_tampering_are_rejected(self):
        gt, before, snapshot = fixture()
        ref = TransferReference(**snapshot)
        changed = deepcopy(before); changed['objects']['book']['room_id'] = 'study'
        with self.assertRaisesRegex(ValueError, 'final state'):
            transfer_metric(gt['goals'][0], ref, before, changed, [], False)
        bad = deepcopy(before)
        event = change(bad, 'no_op')
        event['after']['position'][0] = 30.5
        with self.assertRaisesRegex(ValueError, 'collision|final state'):
            transfer_metric(gt['goals'][0], ref, before, bad, [event], False)
        metric, after, events = run_cost(before, snapshot, ('move_forward', {'x': 6.5}))
        events[0]['transfer_collision_after_sha256'] = 'incorrect'
        with self.assertRaisesRegex(ValueError, 'after-action collision'):
            transfer_metric(gt['goals'][0], ref, before, after, events, False)
        with tempfile.TemporaryDirectory() as tmp:
            reference, live = prepare_reference(SimpleNamespace(evaluation_transfer=lambda goals: snapshot), gt, tmp)
            loaded = load_reference(reference, tmp)
            self.assertIsInstance(loaded, TransferReference)
            self.assertEqual(loaded.remaining(gt['goals'][0], before), live.remaining(gt['goals'][0], before))
            (Path(tmp)/reference['file']).write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_reference(reference, tmp)

    def test_released_object_outside_the_grid_does_not_block_unrelated_cells(self):
        _, before, snapshot = fixture()
        reference = TransferReference(**snapshot)
        outside = deepcopy(before)
        outside['objects']['book'] = object_at(-10, -10)
        np.testing.assert_array_equal(reference.grid(outside).raw, reference.base)

    def test_reference_cost_uses_configured_default_step_instead_of_action_count(self):
        gt, before, snapshot = fixture(step=.5)
        reference = TransferReference(**snapshot)
        self.assertEqual(reference.remaining(gt['goals'][0], before), 50)
        metric, _, events = run_cost(before, snapshot, ('move_forward', {'x': 6.5}))
        self.assertEqual(len(events), 1)
        self.assertEqual(metric['charged_cost'], 10)
        self.assertEqual(metric['remaining_cost'], 40)

    def test_real_environment_capture_reproduces_raster_without_removing_background(self):

        try:
            from benchmark.contracts.raster import derive_navigation
            from benchmark.contracts.doors import POLICY
        except ModuleNotFoundError:
            self.skipTest('Builder source is needed for the environment raster integration check')
        from benchmark.scenes.isaac_env import IsaacVLNEnv, EnvConfig
        gt, before, _ = fixture()
        rooms = [{'room_id': 'hall', 'boundary_polygon': [[0,0],[25,0],[25,9],[0,9]]},
                 {'room_id': 'study', 'boundary_polygon': [[25,0],[40,0],[40,9],[25,9]]}]
        obstacle = object_at(12.6, 5.5); obstacle.update(object_id='obstacle', category='cabinet')
        before['objects']['obstacle'] = obstacle
        setup = {'policy': POLICY, 'doors': []}
        grid = derive_navigation(rooms, list(before['objects'].values()), resolution=1,
                                 robot_radius=0, environment_setup=setup)
        env = IsaacVLNEnv.__new__(IsaacVLNEnv)
        env._np = np; env.config = EnvConfig(robot_radius=0, move_step=1)
        env._metadata = {'rooms': rooms}; env._environment_setup = setup
        env._objects = deepcopy(before['objects']); env._held_object = None
        env._pickable = {'book'}; env._floor_height = 0
        env._static_base = np.asarray(grid['static_base_occupancy'], bool); env._static = env._static_base.copy()
        env._dynamic = np.asarray(grid['dynamic_occupancy'], bool)
        env._origin = np.asarray(grid['origin_xy']); env._resolution = 1
        env._room_labels = np.asarray(grid['room_labels']); env._room_ids = grid['room_ids']
        env._door_open = set(); env._door_cells = {}
        env.runtime_state_v5 = lambda: deepcopy(before)
        reference = TransferReference(**env.evaluation_transfer(gt['goals']))
        carry = reference.grid(before, carrying=True)
        self.assertFalse(carry.walkable([12.6, 5.5]))
        self.assertGreater(reference.remaining(gt['goals'][0], before), 0)

    def test_partial_runtime_evidence_recomputes_offline_and_exports_cost_details(self):
        try:
            from benchmark.evaluation.requests import evaluate_request
        except ModuleNotFoundError:
            self.skipTest('Builder source is needed for the complete request integration check')
        from benchmark.evaluation.task_diagnostics import analyze_run
        from benchmark.reporting.task_diagnostics_report import write_diagnostics
        gt, before, snapshot = fixture()
        metric, after, actions = run_cost(before, snapshot,
            ('move_forward', {'x': 11.5}), ('pick', {}))
        outcome = evaluate_request(gt, after, [], actions, [], None)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'; session = result_session(root)
            reference, grid = prepare_reference(SimpleNamespace(evaluation_transfer=lambda goals: snapshot), gt, session)
            result = {'task_id': 't', 'scored': True, 'dispatched': True, 'success': False,
                'agent_status': 'ok', 'reason_code': 'agent_done', 'steps': len(actions),
                'goal_results': outcome['goals'], 'prediction': None}
            result['evaluation'] = evaluate_metrics(gt, before, after, [], actions, result, reference=reference, grid=grid)
            self.assertEqual(result['evaluation']['score'], metric['score'])
            public = {'task_id': 't', 'instruction': 'Bring the book to the study.'}
            task = {'original': public, 'original_ground_truth': gt, 'state_before': before,
                'metric_reference': reference, 'repair': {'status': 'ready', 'public_task': public,
                    'runtime_ground_truth': gt, 'repair_record': {}}}
            rows = {'runtime_tasks.jsonl': [task], 'action_changelog.jsonl': actions,
                'request_outcomes.jsonl': [result], 'agent_io.jsonl': [
                    {'direction': 'request', 'request_type': 'step', 'task_id': 't', 'payload': public}]}
            for name, records in rows.items():
                session_file(session, name).write_text(''.join(json.dumps(row)+'\n' for row in records))
            (root/'run_config.json').write_text(json.dumps({'official': False}))
            (session/'metrics.json').write_text(json.dumps({'run_status': 'completed',
                'evaluation_valid': True, 'task_results': [result]}))
            sources = {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            report = analyze_run(root)
            self.assertEqual(report['issues'], [])
            self.assertEqual(report['tasks'][0]['metric_version'], METRIC_VERSION)
            self.assertEqual(report['summary']['formal_rule_average'], metric['score'])
            self.assertEqual(report['summary']['transfer_metric']['goals_with_reference'], 1)
            output = Path(tmp)/'report'; write_diagnostics(report, output)
            self.assertIn('reference_cost', (output/'transfer_goals.csv').read_text())
            self.assertIn('剩余成本 J', (output/'report.html').read_text())
            for file, data in sources.items():
                self.assertEqual(file.read_bytes(), data)

    def test_request_runner_captures_reference_and_actual_collision_digests(self):
        try:
            from tests.test_contextual_requests import Env, Agent, fixture_v3, select_scenario, write_dataset, BenchmarkDataset
            from benchmark.contracts.raster import derive_navigation
            from benchmark.contracts.doors import POLICY
        except ModuleNotFoundError:
            self.skipTest('Builder source is needed for the request runner integration check')
        from benchmark.scenes.isaac_env import IsaacVLNEnv, EnvConfig
        from benchmark.runtime.session_runner import SessionRunner
        from benchmark.evaluation.report import write_summary
        from benchmark.evaluation.task_diagnostics import analyze_run
        from tests.test_observations import ManualTime
        compiler, inputs, plan = fixture_v3()
        rooms = inputs['metadata']['rooms']
        setup = {'policy': POLICY, 'doors': []}
        class ReferenceEnv(Env):
            def __init__(self, state):
                super().__init__(state)
                self.pose = pose(4.5, .5)
                self.state.update(robot_pose=deepcopy(self.pose), floor_heights={'study':0, 'kitchen':0})
                for obj in self.state['objects'].values():
                    box = obj['geometry']['world_aabb']
                    obj.update(held=False, interaction_position=np.mean([box['min'],box['max']],axis=0).tolist())
                avatar = self.avatar = IsaacVLNEnv.__new__(IsaacVLNEnv)
                grid = self.raster()
                avatar._np = np; avatar.config = EnvConfig(robot_radius=.1)
                avatar._metadata = {'rooms':rooms}; avatar._environment_setup = setup
                avatar._objects = self.state['objects']; avatar._held_object = None
                avatar._pickable = {'book_red','cup_blue'}; avatar._floor_height = 0
                avatar._resolution = .1; avatar._origin = np.asarray(grid['origin_xy'])
                avatar._room_labels = np.asarray(grid['room_labels']); avatar._room_ids = grid['room_ids']
                avatar.runtime_state_v5 = self.runtime_state_v5
                avatar.evaluation_navigation = self.evaluation_navigation
            def raster(self):
                return derive_navigation(rooms, list(self.state['objects'].values()), resolution=.1,
                    robot_radius=.1, environment_setup=setup)
            def evaluation_navigation(self):
                grid = self.raster()
                return {'blocked':np.asarray(grid['static_base_occupancy'],bool) | np.asarray(grid['dynamic_occupancy'],bool),
                    'origin':grid['origin_xy'], 'resolution':.1, 'radius':.1}
            def evaluation_transfer(self, goals):
                return self.avatar.evaluation_transfer(goals)
            def step(self, action):
                self.pose['position'][0] -= 1
                self.state['robot_pose'] = deepcopy(self.pose)
                return self.observe(), 0, False, False, {'success':True, 'collision':False}
        class Walker(Agent):
            def request(self, payload, on_wait=None):
                self.requests.append(deepcopy(payload))
                return {'sequence_reset_ack':payload['sequence_context']['run_id']} if payload['type']=='session_start' else {'action':'move_backward','done':True}
        with tempfile.TemporaryDirectory() as tmp:
            archive, output = Path(tmp)/'archive', Path(tmp)/'run'
            write_dataset(archive, select_scenario(compiler,plan), compiler, inputs)
            dataset = BenchmarkDataset(archive); state = dataset.canonical_state('synthetic_demo')
            env = ReferenceEnv(state)
            SessionRunner(output, Walker(), max_steps=1, observation_clock=ManualTime().sampler()).run_scene(
                'synthetic_demo', dataset.sessions('synthetic_demo'), state, lambda *args:env)
            summary = write_summary(dataset, output)
            self.assertTrue(summary['metrics_complete'])
            self.assertGreater(summary['score_100'], 0)
            self.assertLess(summary['score_100'], 100)
            (output/'run_config.json').write_text(json.dumps({'official':False}))
            report = analyze_run(output)
            self.assertEqual(report['issues'], [])
            self.assertEqual(report['summary']['formal_rule_average'], summary['score_100'])
            self.assertEqual(report['tasks'][0]['evaluation']['reference']['kind'], 'transfer_cost')


if __name__ == '__main__':
    unittest.main()
