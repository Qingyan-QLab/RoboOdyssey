"""Exercise dispatch, automatic stopping and saved replay through SessionRunner."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from roboodyssey.artifacts.layout import iter_sessions, session_file
from roboodyssey.evaluation.task_diagnostics import analyze_run
from roboodyssey.runtime.session_runner import SessionRunner
from roboodyssey.scenes.dataset_loader import BenchmarkDataset
from tests.frozen_data import fixture_v3, goal, task, select_scenario, write_dataset
from tests.test_contextual_requests import Agent


class MaplessEnv:
    def __init__(self, state, *, start_x=2.6, next_x=1.85, move_target=False):
        self.state = deepcopy(state)
        self.state['held_object'] = None
        self.pose = {'position': [start_x, 1.1, 0]}
        self.next_x = next_x
        self.move_target = move_target

    def runtime_state_v5(self):
        return deepcopy(self.state)

    def get_agent_pose(self):
        return deepcopy(self.pose)

    def set_task(self, value):
        self.task = value

    def observe(self):
        return {'images': {}}

    def step(self, action):
        self.pose['position'][0] = self.next_x
        if self.move_target:
            self.state['objects']['book_red']['geometry']['world_aabb']['min'][0] += .01
            self.state['objects']['book_red']['geometry']['world_aabb']['max'][0] += .01
        return self.observe(), 0, False, False, {'success': True, 'collision': False}

    def close(self):
        pass


class GridEnv(MaplessEnv):
    def __init__(self, state, *, wall=False, **kwargs):
        super().__init__(state, **kwargs)
        self.blocked = np.zeros((40, 60), bool)
        if wall:
            self.blocked[:30, 14:15] = True

    def evaluation_navigation(self):
        return {'blocked': self.blocked.copy(), 'origin': [0, 0], 'resolution': .1, 'radius': .29}


class Walker(Agent):
    def request(self, payload, on_wait=None):
        self.requests.append(deepcopy(payload))
        if payload['type'] == 'session_start':
            return {'sequence_reset_ack': payload['sequence_context']['run_id']}
        return {'action': 'move_forward'}


class ArrivalRuntimeTests(unittest.TestCase):
    def run_case(self, directory, env_type=GridEnv, **kwargs):
        compiler, inputs, plan = fixture_v3([task('visit', goal('book_red'))])
        root = Path(directory)
        archive, output = root/'dataset', root/'run'
        write_dataset(archive, select_scenario(compiler, plan), compiler, inputs)
        dataset = BenchmarkDataset(archive)
        state = dataset.canonical_state('synthetic_demo')
        env = env_type(state, **kwargs)
        SessionRunner(output, Walker(), max_steps=2).run_scene(
            'synthetic_demo', dataset.sessions('synthetic_demo'), state, lambda *args: env)
        (output/'run_config.json').write_text(json.dumps({'official': False}))
        _, _, session = next(iter_sessions(output))
        def one(name):
            return json.loads(session_file(session, name).read_text().splitlines()[0])
        return output, one('runtime_tasks.jsonl'), one('request_outcomes.jsonl')

    def test_wall_does_not_skip_or_stop_the_task_at_nearby_wrong_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, task_record, outcome = self.run_case(tmp, wall=True, start_x=1.85)
            self.assertEqual(task_record['repair']['status'], 'ready')
            self.assertTrue(outcome['dispatched'])
            self.assertFalse(outcome['success'])
            self.assertEqual(outcome['reason_code'], 'step_limit')
            self.assertEqual(outcome['steps'], 2)
            self.assertEqual(outcome['evaluation']['score'], 0)
            report = analyze_run(output)
            self.assertEqual(report['issues'], [])
            self.assertEqual(report['tasks'][0]['evaluation'], outcome['evaluation'])

    def test_unobstructed_arrival_stops_and_replays_at_the_same_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, _, outcome = self.run_case(tmp)
            self.assertTrue(outcome['success'])
            self.assertEqual(outcome['reason_code'], 'success')
            self.assertEqual(outcome['steps'], 1)
            self.assertEqual(outcome['goal_results'][0]['pose_index'], 1)
            report = analyze_run(output)
            self.assertEqual(report['issues'], [])
            self.assertEqual(report['tasks'][0]['evaluation'], outcome['evaluation'])
            self.assertEqual(report['tasks'][0]['navigation']['goals'][0]['reached_pose_index'], 1)

    def test_valid_predispatch_arrival_keeps_verifiable_exclusion_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, task_record, outcome = self.run_case(tmp, start_x=1.85)
            self.assertEqual(task_record['repair']['status'], 'already_satisfied')
            self.assertFalse(outcome['dispatched'])
            self.assertFalse(outcome['scored'])
            self.assertEqual(task_record['metric_reference']['status'], 'ready')
            report = analyze_run(output)
            self.assertEqual(report['issues'], [])
            self.assertTrue(report['summary']['task_score_evidence_valid'])

    def test_missing_map_never_skips_or_succeeds_from_proximity(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, task_record, outcome = self.run_case(tmp, MaplessEnv, start_x=1.85)
            self.assertEqual(task_record['repair']['status'], 'ready')
            self.assertFalse(outcome['success'])
            self.assertEqual(outcome['steps'], 2)
            self.assertEqual(outcome['goal_results'][0]['navigation_evidence_status'],
                             'navigation_obstacle_evidence_missing')
            self.assertIsNone(outcome['evaluation']['score'])
            report = analyze_run(output)
            self.assertFalse(report['summary']['task_score_evidence_valid'])

    def test_changed_target_invalidates_the_frozen_arrival_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, outcome = self.run_case(tmp, move_target=True)
            self.assertFalse(outcome['success'])
            self.assertEqual(outcome['goal_results'][0]['navigation_evidence_status'],
                             'navigation_world_changed_since_reference')
            self.assertEqual(outcome['evaluation']['status'], 'unavailable')
            self.assertIsNone(outcome['evaluation']['score'])

    def test_changed_collision_map_is_detected_without_object_state_changes(self):
        class ChangedGrid(GridEnv):
            def step(self, action):
                result = super().step(action)
                self.blocked[:30, 14:15] = True
                return result
        with tempfile.TemporaryDirectory() as tmp:
            output, _, outcome = self.run_case(tmp, ChangedGrid)
            self.assertFalse(outcome['success'])
            self.assertEqual(outcome['reason_code'], 'step_limit')
            self.assertEqual(outcome['goal_results'][0]['navigation_evidence_status'],
                             'navigation_world_changed_since_reference')
            self.assertIsNone(outcome['evaluation']['score'])
            _, _, session = next(iter_sessions(output))
            first = json.loads(session_file(session, 'action_changelog.jsonl').read_text().splitlines()[0])
            self.assertEqual(first['state_changes']['objects'], [])
            self.assertNotEqual(first['navigation_collision_before_sha256'],
                                first['navigation_collision_after_sha256'])
            report = analyze_run(output)
            self.assertFalse(report['summary']['task_score_evidence_valid'])
            self.assertIsNone(report['summary']['formal_rule_average'])
            self.assertTrue(any('world changed' in issue['error'] for issue in report['issues']))


if __name__ == '__main__':
    unittest.main()
