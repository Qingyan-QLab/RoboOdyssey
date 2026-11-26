"""Comparisons must retain failed attempts and reject changes in conditions."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from roboodyssey.artifacts.layout import read_json, write_json
from roboodyssey.evaluation.requests import request_totals
from roboodyssey.evaluation.task_metrics import METRIC_POLICY, METRIC_VERSION
from roboodyssey.reporting.compare import compare_runs, _profile_hash


def make_run(root, *, sequence='q', score=80, success=True):
    root.mkdir(parents=True)
    identity = {'scene_id': 's', 'sequence_id': sequence,
                'scene_reference': {'revision': 'r_1', 'manifest_sha256': 'a'*64}}
    runtime = {'name': 'contextual_requests', 'version': 3}
    write_json(root/'manifest.json', {'artifact_schema': 'benchmark_results', 'schema_version': 2,
        'kind': 'run', 'sequence_identity': identity, 'runtime_contract': runtime,
        'sequences': [{'scene_id': 's', 'session_ids': ['session_000'], 'sessions_root': 'sessions'}]})
    profile = {'dataset_manifest_sha256': 'b'*64, 'dataset_tree_sha256': 'c'*64,
        'sequence_identity': identity, 'scene_ids': ['s'], 'max_steps_per_task': 2000,
        'agent_timeout_seconds': 120, 'camera_resolution': [128, 128],
        'observation_policy': {'fps': 2}, 'metric_policy': METRIC_POLICY,
        'protocol_version': 3, 'robot_body': {'selection': 'cylinder'},
        'horizontal_fov_degrees': 100, 'robot_radius_m': .1, 'camera_height_m': 1.5,
        'success_rule_version': 'contextual_v3_correctness_only_v1', 'runtime_contract': runtime,
        'reference_models': {'navigation': 'grid_distance_v1', 'transfer': 1},
        'interaction_model': {'version': 'avatar_fixed_offset_v1'},
        'lighting': {'mode': 'roboodyssey', 'dome_intensity': 3000.0, 'fill_intensity': 800.0}}
    config = {**deepcopy(profile), 'official': True, 'evaluation_profile': profile,
        'evaluation_profile_sha256': _profile_hash(profile),
        'motion': {'move_step_m': .25, 'interaction_model': profile['interaction_model']},
        'camera': {'resolution': [128, 128]}, 'recording': None, 'holoagent_sensors': None,
        'step_delay_seconds': 0}
    write_json(root/'run_config.json', config)
    task = {'task_id': 't', 'task_type': 'long_horizon', 'capabilities': ['navigation', 'memory'],
            'scored': True, 'success': success,
            'evaluation': {'version': METRIC_VERSION, 'score': score, 'score_max': 100, 'status': 'ready'}}
    summary = {**request_totals([task]), 'sequence_identity': identity,
               'sessions': [{'scene_id': 's', 'session_id': 'session_000', 'task_results': [task]}]}
    write_json(root/'summary.json', summary)
    return root


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.a = make_run(self.root/'a', score=70, success=False)
        self.b = make_run(self.root/'b', score=90)

    def compare(self, **kwargs):
        return compare_runs({'model_a': [self.a], 'model_b': [self.b]}, **kwargs)

    def test_paired_sequence_and_overlapping_task_groups(self):
        report = self.compare()
        self.assertTrue(report['comparable'])
        comparison = report['comparisons'][0]
        self.assertEqual(comparison['pairs'][0]['score_delta'], 20)
        self.assertEqual(comparison['pairs'][0]['task_success_rate_delta'], 1)
        self.assertEqual(comparison['conditions'][0]['by_capability']['memory']['mean_score_delta'], 20)
        self.assertIsNone(report['method']['uncertainty_estimate'])

    def test_changed_profile_or_action_budget_does_not_mix(self):
        for field, value in (('motion', {'move_step_m': .5}), ('official', False),
                             ('recording', {'enabled': True})):
            with self.subTest(field=field):
                original = read_json(self.b/'run_config.json')
                modified = deepcopy(original); modified[field] = value
                write_json(self.b/'run_config.json', modified)
                self.assertFalse(self.compare()['comparable'])
                write_json(self.b/'run_config.json', original)

    def test_old_cylinder_and_stretch_official_results_do_not_mix(self):
        from roboodyssey.scenes.robot_body import default_robot_identity, OFFICIAL_EMBODIMENT_VERSION
        config = read_json(self.b/'run_config.json')
        profile = config['evaluation_profile']
        profile['robot_body'] = default_robot_identity()
        profile['robot_radius_m'] = profile['robot_body']['robot_radius_m']
        profile['embodiment_version'] = OFFICIAL_EMBODIMENT_VERSION
        config['evaluation_profile_sha256'] = _profile_hash(profile)
        write_json(self.b/'run_config.json', config)
        self.assertFalse(self.compare()['comparable'])

    def test_changed_data_content_rejected_even_with_same_ids(self):
        config = read_json(self.b/'run_config.json')
        config['dataset_tree_sha256'] = config['evaluation_profile']['dataset_tree_sha256'] = 'd'*64
        config['evaluation_profile_sha256'] = _profile_hash(config['evaluation_profile'])
        write_json(self.b/'run_config.json', config)
        report = self.compare()
        self.assertFalse(report['comparable'])
        self.assertIn('data_or_evaluation_profile_mismatch', report['comparisons'][0]['pairs'][0]['issues'])

    def test_missing_sequence_prevents_survivor_only_aggregate(self):
        other = make_run(self.root/'other', sequence='missing')
        report = compare_runs({'a': [self.a, other], 'b': [self.b]})
        self.assertFalse(report['comparable'])
        self.assertEqual(report['comparisons'][0]['planned_pairs'], 2)
        self.assertEqual(report['comparisons'][0]['comparable_pairs'], 1)
        self.assertIsNone(report['comparisons'][0]['conditions'][0]['mean_score_delta'])

    def test_failed_attempt_kept_and_no_best_retry_selection(self):
        (self.b/'summary.json').unlink()
        report = self.compare()
        self.assertFalse(report['comparable'])
        self.assertEqual(report['comparisons'][0]['planned_pairs'], 1)
        self.assertTrue(report['comparisons'][0]['pairs'][0]['candidate']['issues'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            compare_runs({'a': [self.a, self.a], 'b': [self.b]})

    def test_seeds_require_explicit_pairing(self):
        report = compare_runs({'a': [{'run': self.a, 'replicate': 'seed1'}],
                               'b': [{'run': self.b, 'replicate': 'seed2'}]})
        self.assertFalse(report['comparable'])
        self.assertEqual(report['comparisons'][0]['planned_pairs'], 2)

    def test_stale_summary_and_missing_session_are_not_comparable(self):
        original = read_json(self.b/'summary.json')
        for change in ({'score_100': 100}, {'sessions': []}):
            write_json(self.b/'summary.json', {**original, **change})
            self.assertFalse(self.compare()['comparable'])

    def test_different_runtime_conditions_are_grouped_separately(self):
        a2, b2 = make_run(self.root/'a2', sequence='q2'), make_run(self.root/'b2', sequence='q2')
        for root in (a2, b2):
            config = read_json(root/'run_config.json'); config['motion']['move_step_m'] = .3
            write_json(root/'run_config.json', config)
        report = compare_runs({'a': [self.a, a2], 'b': [self.b, b2]})
        self.assertTrue(report['comparable'])
        self.assertEqual(len(report['comparisons'][0]['conditions']), 2)

    def test_saved_bindings_provide_exact_history_gap_without_inference(self):
        import json
        for root in (self.a, self.b):
            summary = read_json(root/'summary.json')
            task = summary['sessions'][0]['task_results'][0]
            tasks = [{**deepcopy(task), 'task_id': f't{i}'} for i in range(3)]
            summary['sessions'][0]['task_results'] = tasks
            summary.update(request_totals(tasks))
            write_json(root/'summary.json', summary)
            path = root/'sessions/session_000/traces/runtime_tasks.jsonl'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'original': {'task_id': 't2'},
                'source_context': [{'session_id': 'session_000', 'task_id': 't0'}]})+'\n')
        report = self.compare()
        self.assertTrue(report['comparable'])
        groups = report['comparisons'][0]['conditions'][0]['by_history_intervening_requests']
        self.assertEqual(groups['1']['paired_tasks'], 1)
        self.assertEqual(groups['unknown']['paired_tasks'], 2)

    def test_older_profile_missing_rule_identity_is_not_guessed(self):
        config = read_json(self.b/'run_config.json')
        config['evaluation_profile'].pop('interaction_model')
        config['evaluation_profile_sha256'] = _profile_hash(config['evaluation_profile'])
        write_json(self.b/'run_config.json', config)
        self.assertFalse(self.compare()['comparable'])

    def test_malformed_json_structure_retains_bad_run_without_aborting_report(self):
        original = read_json(self.b/'run_config.json')
        for config in ([], {**original, 'motion': []}):
            write_json(self.b/'run_config.json', config)
            report = self.compare()
            self.assertFalse(report['comparable'])
            self.assertTrue(report['comparisons'][0]['pairs'][0]['candidate']['issues'])

    def test_reordered_sessions_cannot_change_history_gap(self):
        for root in (self.a, self.b):
            layout = read_json(root/'manifest.json')
            layout['sequences'][0]['session_ids'].append('session_001')
            write_json(root/'manifest.json', layout)
            summary = read_json(root/'summary.json')
            summary['sessions'].append({'scene_id': 's', 'session_id': 'session_001', 'task_results': []})
            if root == self.b:
                summary['sessions'].reverse()
            write_json(root/'summary.json', summary)
        self.assertFalse(self.compare()['comparable'])


if __name__ == '__main__':
    unittest.main()
