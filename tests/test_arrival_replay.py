"""The saved obstacle evidence must establish the same arrivals as runtime."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from roboodyssey.artifacts.layout import session_file
from roboodyssey.evaluation.navigation import GridDistance
from roboodyssey.evaluation.requests import SUCCESS_RULE_VERSION, evaluate_request
from roboodyssey.evaluation.task_diagnostics import analyze_run, navigation_diagnostics, recorded_success_rule
from roboodyssey.evaluation.task_metrics import prepare_reference, evaluate_metrics
from tests.frozen_data import result_session


class ArrivalReplayTests(unittest.TestCase):
    def fixture(self, root, *, excluded=False):
        session = result_session(root)
        blocked = np.zeros((40, 60), dtype=bool)
        blocked[:30, 20:22] = True  # Only a distant doorway connects the sides.
        box = {'min': [2.35, 1.6, 0.0], 'max': [2.55, 1.8, 1.0]}
        pose = {'position': [1.65, 1.65, 0.0]}
        goal = {'goal_id': 'g0', 'goal_kind': 'visit', 'target': {'object_ids': ['o0']},
                'success_criteria': {'position_tolerance_m': .8},
                'expected_outcome': {'target_geometry': {'world_aabb_xy': box}}}
        truth = {'task_id': 't', 'state_assumption': 'actual_predispatch_state',
                 'goals': [goal], 'success_criteria': {'goal_order': ['g0']}}
        state = {'robot_pose': pose, 'objects': {'o0': {'geometry': {'world_aabb': box}}}}
        snapshot = {'blocked': blocked, 'origin': [0, 0], 'resolution': .1, 'radius': .29}
        env = SimpleNamespace(evaluation_navigation=lambda: snapshot,
                              evaluation_arrival=lambda goals: snapshot)
        reference, grid = prepare_reference(env, truth, session)
        event = {'task_id': 't', 'task_step': 0, 'action': 'no_op', 'before': pose,
                 'after': pose, 'result': {'success': True, 'collision': False}}
        evaluated = evaluate_request(truth, state, [pose, pose], [event], [], None,
                                     navigation=grid)
        result = {'task_id': 't', 'scored': True, 'dispatched': True, 'agent_status': 'ok',
                  'success': evaluated['success'], 'reason_code': 'agent_done', 'steps': 1,
                  'goal_results': evaluated['goals'], 'success_rule_version': SUCCESS_RULE_VERSION}
        result['evaluation'] = evaluate_metrics(truth, state, state, [pose, pose], [event], result,
                                                 reference=reference, grid=grid)
        public = {'task_id': 't', 'instruction': 'Go to the object.'}
        task = {'original': public, 'original_ground_truth': truth, 'state_before': state,
                'metric_reference': reference, 'success_rule_version': SUCCESS_RULE_VERSION,
                'repair': {'status': 'ready', 'public_task': public, 'runtime_ground_truth': truth,
                           'repair_record': {'changed': False}}}
        if excluded:
            task['repair'].update(status='already_satisfied', public_task=None)
            result.update(scored=False, dispatched=False, success=None,
                          reason_code='already_satisfied', steps=0)
            result.pop('evaluation')
            result.pop('goal_results')
        def save(name, rows):
            session_file(session, name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
        def write():
            save('runtime_tasks.jsonl', [task])
            save('request_outcomes.jsonl', [result])
            save('action_changelog.jsonl', [] if excluded else [event])
            save('agent_io.jsonl', [] if excluded else [{'direction': 'request', 'request_type': 'step',
                 'task_id': 't', 'payload': public}])
            (session/'metrics.json').write_text(json.dumps({'run_status': 'completed',
                 'evaluation_valid': True, 'task_results': [result]}))
        (root/'run_config.json').write_text(json.dumps({'official': False}))
        write()
        return task, result, write

    def test_wrong_side_near_target_stays_failed_online_and_in_saved_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'
            _, result, _ = self.fixture(root)
            self.assertFalse(result['success'])
            before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            report = analyze_run(root)
            self.assertEqual(report['issues'], [])
            self.assertEqual(report['tasks'][0]['evaluation'], result['evaluation'])
            self.assertEqual(report['tasks'][0]['navigation']['goals_completed_in_order'], 0)
            self.assertLess(report['tasks'][0]['navigation']['goals'][0]['closest_distance_m'], .8)
            self.assertEqual(report['summary']['formal_rule_average'], 0)
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_missing_reference_never_reverts_to_geometric_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'
            task, _, write = self.fixture(root)
            task.pop('metric_reference')
            write()
            report = analyze_run(root, rescore=True)
            self.assertIsNone(report['summary']['formal_rule_average'])
            self.assertFalse(report['summary']['task_score_evidence_valid'])
            self.assertIsNone(report['tasks'][0]['navigation'])

    def test_false_predispatch_exclusion_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'
            self.fixture(root, excluded=True)
            report = analyze_run(root)
            self.assertFalse(report['summary']['task_score_evidence_valid'])
            self.assertTrue(any('exclusion fails' in item['error'] for item in report['issues']))

    def test_place_establishment_cannot_be_excluded_even_at_a_valid_arrival(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'
            task, _, write = self.fixture(root, excluded=True)
            task['state_before']['robot_pose']['position'][0] = 2.95
            task['repair']['runtime_ground_truth']['goals'][0]['goal_kind'] = 'establish_place'
            write()
            report = analyze_run(root)
            self.assertFalse(report['summary']['task_score_evidence_valid'])
            self.assertTrue(any('exclusion fails' in item['error'] for item in report['issues']))

    def test_nested_profile_success_rule_must_match_recorded_task(self):
        config = {'evaluation_profile': {'success_rule_version': 'contextual_v3_correctness_only_v1'}}
        with self.assertRaisesRegex(ValueError, 'versions disagree'):
            recorded_success_rule({'success_rule_version': SUCCESS_RULE_VERSION}, {}, config)
        self.assertEqual(recorded_success_rule({}, {}, config),
                         'contextual_v3_correctness_only_v1')

    def test_legacy_early_stop_is_reported_without_relabelling_it_as_a_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'
            task, result, write = self.fixture(root)
            old_rule = 'contextual_v3_correctness_only_v1'
            task['success_rule_version'] = result['success_rule_version'] = old_rule
            result.update(success=True, reason_code='success')
            result['evaluation'].update(version='progress_path_v3', score=100)
            result['goal_results'][0].update(success=True, pose_index=0)
            write()
            report = analyze_run(root, rescore=True)
            self.assertIsNone(report['summary']['formal_rule_average'])
            self.assertTrue(any('new run is required' in item['error'] for item in report['issues']))
            self.assertTrue(report['tasks'][0]['success'])
            self.assertEqual(report['tasks'][0]['success_rule_version'], old_rule)
            self.assertEqual(report['tasks'][0]['original_task_score'], 100)
            self.assertEqual(report['tasks'][0]['original_evaluation']['version'], 'progress_path_v3')

    def test_diagnostics_require_obstacle_evidence(self):
        truth = {'state_assumption': 'actual_predispatch_state',
                 'success_criteria': {'goal_order': ['g']},
                 'goals': [{'goal_id': 'g', 'goal_kind': 'visit', 'target': {'object_ids': ['o']}}]}
        with self.assertRaisesRegex(ValueError, 'arrival reference missing'):
            navigation_diagnostics(truth, {}, [], {})

    def test_reconstruction_preserves_unavailable_reason_without_inventing_a_file(self):
        from roboodyssey.evaluation.rescore import ReconstructedReference
        with tempfile.TemporaryDirectory() as tmp:
            reference = ReconstructedReference.__new__(ReconstructedReference)
            reference.scene_id = 'scene'
            reference.output = Path(tmp)
            reference.setup = {'doors': []}
            reference.cache = {'objects': {'o': {}}}
            reference.provenance = {'kind': 'reconstructed_from_verified_dataset_and_actual_task_state'}
            reference.environment = lambda before: object()
            with patch('roboodyssey.evaluation.rescore.prepare_reference', return_value=(
                    {'status': 'unavailable', 'reason': 'measured_collision_grid_mismatch'}, None)):
                record, grid = reference.prepare('scene', 'session',
                    {'goals': [{'goal_kind': 'visit'}]}, {'objects': {'o': {}}, 'door_open': []})
            self.assertEqual(record['reason'], 'measured_collision_grid_mismatch')
            self.assertEqual(record['status'], 'unavailable')
            self.assertNotIn('file', record)
            self.assertIsNone(grid)


if __name__ == '__main__':
    unittest.main()
