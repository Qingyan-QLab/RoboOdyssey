from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from benchmark.evaluation.task_diagnostics import analyze_run, navigation_diagnostics, task_score
from benchmark.reporting.task_diagnostics_report import write_diagnostics


def pose(x):
    return {'position': [x, 0, 1.5]}


def truth_at(*xs):
    goals = [{'goal_id': f'g{i}', 'goal_kind': 'visit', 'target': {'object_ids': [f'o{i}']},
              'expected_outcome': {'target_geometry': {'world_aabb_xy': {'min': [x, -1], 'max': [x+1, 1]}}},
              'success_criteria': {'position_tolerance_m': 1}} for i, x in enumerate(xs)]
    return {'task_id': 't', 'state_assumption': 'actual_predispatch_state', 'goals': goals,
            'success_criteria': {'goal_order': [g['goal_id'] for g in goals]}}


def action(step, before, after, name='move_forward', collision=False):
    return {'task_id': 't', 'task_step': step, 'action': {'action': name},
            'before': pose(before), 'after': pose(after),
            'result': {'collision': collision, **({'reset_applied': True} if name == 'reset' else {})}}


def outcome(*hits, success=False, steps=0):
    return {'task_id': 't', 'scored': True, 'dispatched': True, 'agent_status': 'ok',
            'success': success, 'reason_code': 'success' if success else 'step_limit', 'steps': steps,
            'goal_results': [{'goal_id': f'g{i}', 'success': hit is not None,
                              **({'pose_index': hit} if hit is not None else {})} for i, hit in enumerate(hits)]}


class NavigationDiagnosticsTests(unittest.TestCase):
    def test_final_and_best_progress_use_distance_to_allowed_region(self):
        actions = [action(0, 0, 5), action(1, 5, 3)]
        result = navigation_diagnostics(truth_at(10), {'robot_pose': pose(0)}, actions, outcome(None))
        goal = result['goals'][0]
        self.assertEqual((goal['start_distance_m'], goal['final_distance_m'], goal['closest_distance_m']), (10, 7, 5))
        self.assertAlmostEqual(result['final_progress_score'], 100/3)
        self.assertAlmostEqual(result['best_progress_score'], 500/9)
        self.assertEqual(result['motion']['walking_length_m'], 7)
        self.assertIsNone(result['walking_efficiency'])

    def test_out_of_order_pass_does_not_complete_later_goal(self):
        actions = [action(0, 0, 2), action(1, 2, 9), action(2, 9, 7)]
        result = navigation_diagnostics(truth_at(10, 2), {'robot_pose': pose(0)}, actions, outcome(3, None))
        self.assertEqual(result['goals'][1]['closest_distance_m'], 0)
        self.assertEqual(result['goals'][1]['active_closest_distance_m'], 4)
        self.assertEqual(result['goals_completed_in_order'], 1)
        self.assertAlmostEqual(result['final_progress_score'], 70)
        self.assertAlmostEqual(result['best_progress_score'], 70)
        self.assertIn('多目标只完成部分', result['categories'])

    def test_same_sample_cannot_complete_two_goals_with_zero_remaining(self):
        result = navigation_diagnostics(truth_at(10, 10), {'robot_pose': pose(9)},
                                        [action(0, 9, 15)], outcome(0, None))
        self.assertEqual(result['best_progress_score'], 50)
        self.assertEqual(result['final_progress_score'], 50)

    def test_reset_displacement_and_post_reset_recovery_are_separate(self):
        actions = [action(0, 0, 3, collision=True), action(1, 3, 8, 'reset'), action(2, 8, 9)]
        result = navigation_diagnostics(truth_at(10), {'robot_pose': pose(0)}, actions, outcome(5, success=True))
        motion = result['motion']
        self.assertEqual(motion['walking_length_m'], 4)
        self.assertEqual(motion['reset_displacement_m'], 5)
        self.assertEqual(motion['reset_count'], 1)
        self.assertEqual(motion['collision_rate'], .5)
        self.assertAlmostEqual(motion['collision_rate_all_actions'], 1/3)
        reset = result['reset_recoveries'][0]
        self.assertEqual([reset[k] for k in ('before_distance_m', 'after_distance_m', 'recovery_best_distance_m')], [7, 2, 1])
        self.assertEqual(reset['teleport_progress_m'], 5)
        self.assertEqual(reset['walking_progress_m'], 1)
        self.assertEqual(reset['category'], '复位后行走达标')
        self.assertIsNone(result['walking_efficiency'])

    def test_reset_arrival_is_not_counted_as_walking_recovery(self):
        actions = [action(0, 0, 3), action(1, 3, 9, 'reset')]
        result = navigation_diagnostics(truth_at(10), {'robot_pose': pose(0)}, actions, outcome(3, success=True))
        reset = result['reset_recoveries'][0]
        self.assertEqual(reset['category'], '复位直接进入达标范围')
        self.assertEqual(reset['walking_progress_m'], 0)
        self.assertEqual(reset['recovery_window']['action_count'], 0)

    def test_recovery_window_stops_before_next_reset(self):
        actions = [action(0, 0, 3), action(1, 3, 0, 'reset'), action(2, 0, 4), action(3, 4, 0, 'reset'), action(4, 0, 1)]
        result = navigation_diagnostics(truth_at(10), {'robot_pose': pose(0)}, actions, outcome(None))
        first, second = result['reset_recoveries']
        self.assertEqual(first['recovery_end_reason'], 'next_reset')
        self.assertEqual(first['recovery_best_distance_m'], 6)
        self.assertEqual(first['recovery_window']['walking_length_m'], 4)
        self.assertEqual(second['recovery_best_distance_m'], 9)
        self.assertTrue(first['new_best_from_walking'])
        self.assertFalse(second['new_best_from_walking'])

    def test_logged_geometry_and_order_must_agree_with_recorded_results(self):
        with self.assertRaisesRegex(ValueError, 'disagree'):
            navigation_diagnostics(truth_at(10), {'robot_pose': pose(0)}, [action(0, 0, 2)], outcome(1, success=True))
        a = action(0, 0, 2); a['state_changes'] = {'objects': [{'object_id': 'o0'}]}
        with self.assertRaisesRegex(ValueError, 'target moved'):
            navigation_diagnostics(truth_at(10), {'robot_pose': pose(0)}, [a], outcome(None))
        with self.assertRaisesRegex(ValueError, 'discontinuity'):
            navigation_diagnostics(truth_at(10), {'robot_pose': pose(0)}, [action(0, 0, 2), action(1, 5, 7)], outcome(None))

    def test_missing_collision_flag_is_unknown_not_zero(self):
        a = action(0, 0, 2); a['result'].pop('collision')
        result = navigation_diagnostics(truth_at(10), {'robot_pose': pose(0)}, [a], outcome(None))
        self.assertIsNone(result['motion']['collision_rate'])
        self.assertIsNone(result['motion']['collision_count'])

    def test_efficiency_uses_runtime_allowed_region_not_offline_path(self):
        truth = truth_at(10); truth['goals'][0]['reference_trajectory'] = {'length_m': 1234}
        result = navigation_diagnostics(truth, {'robot_pose': pose(0)}, [action(0, 0, -1), action(1, -1, 9)], outcome(3, success=True))
        self.assertAlmostEqual(result['walking_efficiency'], 9/11)


class TaskScoreReportTests(unittest.TestCase):
    def test_excluded_na_and_unsupported_zero(self):
        unsupported = {**outcome(None), 'agent_status': 'unsupported'}
        self.assertEqual(task_score(unsupported), 0)
        for reason in ('already_satisfied', 'unavailable'):
            self.assertIsNone(task_score({'scored': False, 'dispatched': False, 'reason_code': reason, 'failure_causes': [{'task_id': 'source'}]}))
        with self.assertRaisesRegex(ValueError, 'causality'):
            task_score({'scored': False, 'dispatched': False, 'reason_code': 'unavailable'})

    def test_report_uses_recorded_runtime_gt_and_preserves_source_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'; session = root/'scenes/s/sessions/session_000'; session.mkdir(parents=True)
            runtime = truth_at(10); original = truth_at(100); original['state_assumption'] = 'planned_success_assumption'
            public = {'task_id': 't', 'instruction': '<script>alert(1)</script>', 'issued_at': '2026-11-12'}
            result = outcome(1, success=True, steps=1)
            task = {'original': public, 'original_ground_truth': original,
                    'state_before': {'robot_pose': pose(0)}, 'repair': {'status': 'ready', 'public_task': public,
                    'runtime_ground_truth': runtime, 'repair_record': {'changed': True}}}
            def lines(name, rows):
                (session/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (root/'run_config.json').write_text(json.dumps({'official': False}))
            lines('runtime_tasks.jsonl', [task]); lines('request_outcomes.jsonl', [result])
            lines('action_changelog.jsonl', [action(0, 0, 9)])
            lines('agent_io.jsonl', [{'direction': 'request', 'request_type': 'step', 'task_id': 't', 'payload': public}])
            (session/'metrics.json').write_text(json.dumps({'run_status': 'completed', 'evaluation_valid': True, 'task_results': [result]}))
            before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            report = analyze_run(root); out = Path(tmp)/'analysis'; write_diagnostics(report, out)
            self.assertEqual(report['issues'], [])
            self.assertEqual(report['summary']['formal_rule_average'], 100)
            self.assertEqual(report['tasks'][0]['navigation']['goals'][0]['start_distance_m'], 10)
            for p, content in before.items(): self.assertEqual(p.read_bytes(), content)
            page = (out/'report.html').read_text()
            self.assertNotIn('<script>alert(1)', page)
            self.assertIn('&lt;script&gt;alert(1)', page)
            with (out/'tasks.csv').open(encoding='utf-8-sig') as f:
                row = next(csv.DictReader(f)); self.assertEqual(row['formal_task_score'], '100')
            # Missing action evidence is exposed and never silently scored as a
            # zero-length, collision-free trajectory.
            (session/'action_changelog.jsonl').unlink()
            report = analyze_run(root)
            self.assertIsNone(report['tasks'][0]['navigation'])
            self.assertEqual(report['issues'][0]['scope'], 'navigation')


if __name__ == '__main__':
    unittest.main()
