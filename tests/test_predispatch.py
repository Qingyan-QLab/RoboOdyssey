"""Task-start evidence, dispatch and denominator regressions (no simulator)."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from test_runtime_repair import Agent, Environment, fixture, transfer, question
from test_task_repair import return_transfer
from benchmark.evaluation.success import already_satisfied, evaluate_embodied_task
from benchmark.evaluation.report import task_totals
from benchmark.evaluation.report import write_summary
from benchmark.runtime.session_runner import SessionRunner
from benchmark.scenes.dataset_loader import SessionRecord
from benchmark.artifacts.layout import write_json as write
from benchmark.legacy.task_repair import instantiate_task
from benchmark.contracts.legacy_templates import freeze_template


class PredispatchTests(unittest.TestCase):
    def setUp(self):
        self.cache, self.state = fixture()
        self.pose = {'position': [0, 0, 1.5]}

    def prepare(self, state=None):
        template, task, gt = transfer()
        return instantiate_task(template, self.cache, state or self.state, [], task=task, ground_truth=gt)

    def test_transfer_requires_destination_released_and_floor_evidence(self):
        task = transfer()[1]
        gt = self.prepare()['runtime_ground_truth']
        obj = self.state['objects']['book']
        self.assertIsNone(already_satisfied(task, gt, self.state, self.pose))
        obj.update(room_id='kitchen', held=False)
        self.assertIsNone(already_satisfied(task, gt, self.state, self.pose))  # Surface unknown.
        obj['placement_surface'] = 'table'
        self.assertIsNone(already_satisfied(task, gt, self.state, self.pose))
        obj['placement_surface'] = 'floor'
        self.assertIsNotNone(already_satisfied(task, gt, self.state, self.pose))
        # Pre-dispatch exemption does not weaken the post-dispatch action requirement.
        self.assertFalse(evaluate_embodied_task(task, gt, [], self.state)['success'])
        for change in ({'held': True}, {'support_object_id': 'table'}, {'room_id': 'study'}):
            changed = deepcopy(self.state); changed['objects']['book'].update(change)
            self.assertIsNone(already_satisfied(task, gt, changed, self.pose))
        self.state['held_object'] = 'book'
        self.assertIsNone(already_satisfied(task, gt, self.state, self.pose))

    def test_navigation_uses_current_geometry_but_qa_and_ordered_visits_require_dispatch(self):
        task = {'task_type': 'navigation'}
        gt = {'target': {'object_ids': ['book']}, 'expected_outcome': {},
              'success_criteria': {'target_distance_type': 'aabb_2d', 'position_tolerance_m': .5}}
        self.assertIsNotNone(already_satisfied(task, gt, self.state, self.pose))
        self.state['objects']['book']['pose'] = {'position': [5, 0, 0]}
        self.assertIsNone(already_satisfied(task, gt, self.state, self.pose))
        del self.state['objects']['book']['pose']
        gt['success_criteria']['required_site_order'] = ['book', 'cup']
        self.assertIsNone(already_satisfied(task, gt, self.state, self.pose))
        self.assertIsNone(already_satisfied({'task_type': 'qa'}, gt, self.state, self.pose))

    def test_naming_place_still_dispatches_at_landmark(self):
        proposal = {'task_id': 'task_000', 'task_type': 'navigation', 'target_object_id': 'cup',
                    'named_place': {'name': 'Maple Point', 'mode': 'establish',
                                    'paired_task': {'session_id': 'session_003', 'task_id': 'task_000'}}}
        task = {'task_id': 'task_000', 'task_type': 'navigation',
                'instruction': 'Go to the cup and remember this place as Maple Point.'}
        gt = {'target': {'object_ids': ['cup']}, 'expected_outcome': {},
              'success_criteria': {'target_distance_type': 'aabb_2d'}}
        template = freeze_template(proposal, task, gt)
        def forbidden(_):
            self.fail('naming creates an experience and must not be skipped based on position')
        result = instantiate_task(template, self.cache, self.state, [], task=task, ground_truth=gt,
                                  completion_check=forbidden)
        self.assertEqual(result['status'], 'ready')

    def run_specs(self, root, specs, agent, environment=None):
        write(root / 'cache.json', self.cache)
        records = []
        for index, spec in enumerate(specs):
            template, task, gt = spec
            sid = f'session_{index:03d}'
            folder = root / sid
            write(folder / 'gt.json', {'tasks': [gt | {'task_id': task['task_id']}]})
            write(folder / 'templates.json', {'schema_version': 1, 'session_id': sid, 'tasks': [template]})
            records.append(SessionRecord('test', folder, {'session_id': sid, 'tasks': [task]}, {
                'ground_truth': folder / 'gt.json', 'task_templates': folder / 'templates.json',
                'runtime_scene_cache': root / 'cache.json'}))
        output = root / 'run'
        SessionRunner(output, agent, max_steps=8).run_scene('test', records, self.state,
                                                         lambda *_: (environment or Environment)(self.state))
        dataset = SimpleNamespace(manifest={'dataset_id': 'fixture'}, scene_ids=lambda: ['test'],
                                  sessions=lambda _: records)
        summary = write_summary(dataset, output)
        return output, summary

    def test_skipped_transfer_does_not_dispatch_or_create_memory_and_all_skipped_has_no_score(self):
        self.state['objects']['book'].update(room_id='kitchen', placement_surface='floor', held=False)
        # Identity is now ambiguous; physical completion can still be proven privately.
        self.cache['objects']['other']['identity_attributes'] = {}
        self.state['objects']['other']['room_id'] = 'kitchen'
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent()
            output, summary = self.run_specs(Path(tmp), [transfer()], agent)
            self.assertEqual([r['type'] for r in agent.requests], ['session_start'])
            self.assertEqual(summary['tasks_total'], 1)
            self.assertEqual(summary['tasks_evaluated'], 0)
            self.assertEqual(summary['tasks_excluded'], 1)
            self.assertIsNone(summary['task_success_rate'])
            self.assertIsNone(summary['session_success_rate'])
            folder = output / 'scenes/test/sessions/session_000'
            record = json.loads((folder / 'runtime_tasks.jsonl').read_text())
            self.assertEqual(record['status'], 'already_satisfied')
            self.assertIsNone(record['rendered_text'])
            history = json.loads((folder / 'runtime_history.jsonl').read_text())
            self.assertFalse(history['completed_action'])
            self.assertFalse(history['dispatched'])
            metrics = json.loads((folder / 'metrics.json').read_text())
            self.assertIsNone(metrics['task_score'])
            self.assertIsNone(metrics['reset_aware_score'])
            # HTML diagnostics must neither crash on null scores nor re-count exclusions.
            from benchmark.reporting.legacy import collect
            analysis = collect(output)
            self.assertIsNone(analysis['sessions'][0]['score'])
            self.assertEqual(analysis['families'], {})

    def test_mixed_summary_and_checkpoint_exclude_skips_without_hiding_missing_results(self):
        self.state['objects']['cup'].update(room_id='kitchen', placement_surface='floor', held=False)
        skipped = transfer('cup', 'Pick up the cup and place it on the floor in the kitchen.')
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent()
            output, summary = self.run_specs(Path(tmp), [skipped, question(), transfer()], agent)
            self.assertEqual(summary['tasks_total'], 3)
            self.assertEqual(summary['tasks_evaluated'], 2)
            self.assertEqual(summary['tasks_excluded'], 1)
            self.assertEqual(summary['task_success_rate'], .5)
            self.assertEqual(summary['task_type_success_rate'], {'qa': 1., 'manipulation': 0.})
            self.assertEqual(summary['capability_success_rate']['manipulation'], 0.)
            runner = SessionRunner(output, agent)
            runner._current_output = output
            runner._current_session_id = 'session_000'
            runner._current_task_results = summary['session_metrics'][0]['task_results']
            runner._current_task = {'task_id': 'missing', 'task_type': 'qa'}
            runner._write_checkpoint(status='interrupted')
            value = json.loads((output / 'metrics.json').read_text())
            self.assertEqual(value['tasks_excluded'], 1)
            self.assertEqual(value['tasks_evaluated'], 1)
            self.assertEqual(value['task_score'], 0.)
            self.assertEqual(task_totals([{'success': False}])['tasks_evaluated'], 1)

    def test_failed_transfer_repairs_history_questions_and_keeps_them_scored(self):
        move = transfer()
        move[0]['issued_at'] = move[1]['issued_at'] = '2026-11-05T09:10:00+08:00'
        current = question(history=True)
        current[1]['capabilities'].append('memory')
        recalled = question('room_after', history=True)
        recalled[1]['capabilities'].append('memory')
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent()
            output, summary = self.run_specs(Path(tmp), [move, current, recalled], agent)
            self.assertEqual(summary['tasks_evaluated'], 3)
            self.assertEqual(summary['tasks_excluded'], 0)
            self.assertEqual(summary['task_success_rate'], 2/3)
            self.assertEqual(summary['exclusions_by_reason'], {})
            self.assertEqual(summary['capability_success_rate']['memory'], 1.)
            self.assertEqual(summary['memory_success_rate_by_horizon'], {'1': 1., '2': 1.})
            self.assertEqual([r['session_id'] for r in agent.requests if r['type'] == 'qa'], ['session_001', 'session_002'])
            self.assertFalse(summary['runtime_repair']['unavailable_tasks_in_score_denominator'])

    def test_missing_time_is_an_archive_problem_and_not_an_agent_failure(self):
        move = transfer()  # Legacy source has no task timestamp.
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent()
            _, summary = self.run_specs(Path(tmp), [move, question(history=True)], agent)
            self.assertEqual(summary['tasks_evaluated'], 1)
            self.assertEqual(summary['exclusions_by_reason'], {'history_time_mismatch': 1})
            blocked = summary['session_metrics'][1]['task_results'][0]
            self.assertIsNone(blocked['success'])
            self.assertEqual(blocked['reason_code'], 'history_time_mismatch')

    def test_skipped_source_does_not_turn_into_a_request_in_later_history(self):
        self.state['objects']['book'].update(room_id='kitchen', placement_surface='floor', held=False)
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent()
            _, summary = self.run_specs(Path(tmp), [transfer(), question(history=True)], agent)
            self.assertEqual(summary['tasks_evaluated'], 0)
            blocked = summary['session_metrics'][1]['task_results'][0]
            self.assertEqual(blocked['reason_code'], 'source_task_not_dispatched')
            self.assertEqual(blocked['dependency']['source_status'], 'already_satisfied')
            self.assertFalse(any(r['type'] == 'qa' for r in agent.requests))

    def test_failed_move_can_make_return_goal_already_satisfied(self):
        # The source task failed before picking up. Its true initial location
        # remains a valid historical destination and the floor goal holds now.
        self.state['objects']['book'].update(held=False, placement_surface='floor')
        move = transfer()
        move[0]['issued_at'] = move[1]['issued_at'] = '2026-11-05T09:10:00+08:00'
        with tempfile.TemporaryDirectory() as tmp:
            _, summary = self.run_specs(Path(tmp), [move, return_transfer()], Agent())
            self.assertEqual(summary['tasks_evaluated'], 1)
            self.assertEqual(summary['task_success_rate'], 0.)
            self.assertEqual(summary['exclusions_by_reason'], {'already_satisfied': 1})
            self.assertEqual(summary['runtime_repair']['unavailable_tasks'], 0)

    def test_executor_records_intermediate_actions_and_repairs_held_final_state(self):
        class ScriptedAgent(Agent):
            def request(self, payload, on_wait=None):
                self.requests.append(deepcopy(payload))
                if payload['type'] == 'session_start': return {}
                if payload['type'] == 'qa': return {'answer': 'kitchen'}
                index = payload['task_step']
                return {'action': {'action': ['pick', 'place', 'pick', 'move_forward'][index]},
                        'done': index == 3}

        class MovingEnvironment(Environment):
            def step(self, action):
                obj = self.state['objects']['book']
                name = action['action']
                if name == 'pick':
                    obj['held'] = True; self.state['held_object'] = 'book'
                elif name == 'place':
                    obj.update(held=False, room_id='study', placement_surface='floor')
                    self.state['held_object'] = None
                else:
                    obj['room_id'] = 'kitchen'
                return self.observe(), 0, False, False, {'success': True, 'object_id': 'book', 'room_id': obj['room_id']}

        move = transfer()
        move[0]['issued_at'] = move[1]['issued_at'] = '2026-11-05T09:10:00+08:00'
        with tempfile.TemporaryDirectory() as tmp:
            agent = ScriptedAgent()
            output, summary = self.run_specs(Path(tmp), [move, question('room_after', history=True)], agent, MovingEnvironment)
            self.assertEqual(summary['tasks_evaluated'], 2)
            self.assertEqual(summary['tasks_excluded'], 0)
            self.assertEqual(summary['task_success_rate'], .5)
            folder = output / 'scenes/test/sessions/session_000'
            history = [json.loads(row) for row in (folder / 'runtime_history.jsonl').read_text().splitlines()]
            self.assertEqual([e['event_scope'] for e in history], ['task', 'action', 'action', 'action'])
            self.assertEqual(history[0]['object_after']['room_id'], 'kitchen')
            self.assertTrue(history[0]['object_after']['held'])
            self.assertEqual(history[2]['object_before']['room_id'], 'bedroom')
            self.assertEqual(history[2]['object_after']['room_id'], 'study')
            self.assertFalse(history[2]['object_after']['held'])
            request = next(r for r in agent.requests if r['type'] == 'qa')
            self.assertIn('when that task ended', request['question'])
            self.assertNotIn('kitchen', request['question'])
            self.assertNotIn('object_after', json.dumps(request))

    def test_timeout_after_pick_preserves_partial_history_and_remains_a_failure(self):
        class TimeoutAgent(Agent):
            def request(self, payload, on_wait=None):
                if payload['type'] == 'session_start': return {}
                if payload['task_step'] == 0: return {'action': {'action': 'pick'}}
                raise TimeoutError('agent did not respond')

        class PickEnvironment(Environment):
            def step(self, action):
                self.state['held_object'] = 'book'
                self.state['objects']['book']['held'] = True
                return self.observe(), 0, False, False, {'success': True, 'object_id': 'book'}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(TimeoutError):
                self.run_specs(root, [transfer()], TimeoutAgent(), PickEnvironment)
            folder = root / 'run/scenes/test/sessions/session_000'
            history = [json.loads(row) for row in (folder / 'runtime_history.jsonl').read_text().splitlines()]
            self.assertEqual(history[0]['task_status'], 'benchmark_interrupted')
            self.assertFalse(history[0]['success'])
            self.assertTrue(history[0]['object_after']['held'])
            self.assertEqual(history[1]['kind'], 'pick')
            metrics = json.loads((folder / 'metrics.json').read_text())
            self.assertEqual(metrics['tasks_evaluated'], 1)
            self.assertEqual(metrics['tasks_excluded'], 0)
            self.assertEqual(metrics['task_score'], 0.)


if __name__ == '__main__': unittest.main()
