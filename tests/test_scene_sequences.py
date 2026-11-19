from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_contextual_requests import Env, Agent, fixture_v3, task, goal, select_scenario, write_dataset
from roboodyssey.scenes.dataset_loader import BenchmarkDataset
from roboodyssey.runtime.session_runner import SessionRunner
from roboodyssey.launch.sequences import aggregate_runs, child_arguments, run_collection
from roboodyssey.evaluation.report import write_summary
from tests.frozen_data import prepare_scene
from roboodyssey.data.frozen import select_sequences
from tests.frozen_data import shared_plan


class ResetAgent(Agent):
    def request(self, payload, on_wait=None):
        response = super().request(payload, on_wait)
        if payload['type'] == 'session_start':
            context = payload['sequence_context']
            if context['reset_memory']:
                response['sequence_reset_ack'] = context['run_id']
            response['memory_update'] = {'prior_session': payload['session_id']}
        return response


class SequenceIntegrationTests(unittest.TestCase):
    def collection(self, root):
        c, inputs, plan = fixture_v3([task('ask', goal('book_red', 'answer', query='current_room'))],
            second=[task('cup', goal('cup_blue', 'answer', query='current_room'))])
        version = prepare_scene(root, c, inputs, plan['names'])
        compiler, shared = version.inputs()
        for sid in ('sequence_A', 'sequence_B'):
            result = select_scenario(compiler, shared_plan(plan, version, sid))
            write_dataset(root / 'sequences' / sid, result, compiler, shared)
        return root

    def test_collection_selection_and_sequence_world_memory_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.collection(Path(tmp) / 'scene')
            with self.assertRaisesRegex(ValueError, 'select'):
                BenchmarkDataset(root)
            agent, states, run_ids = ResetAgent(), [], []
            for sid in ('sequence_A', 'sequence_B'):
                data = BenchmarkDataset(root, sequence_id=sid)
                output = Path(tmp) / sid
                def environment(scene, state, session):
                    states.append(deepcopy(state))
                    return Env(state)
                start = len(agent.requests)
                SessionRunner(output, agent, max_steps=1).run_scene('synthetic_demo', data.sessions('synthetic_demo'),
                    data.canonical_state('synthetic_demo'), environment)
                requests = [r for r in agent.requests[start:] if r['type'] == 'session_start']
                self.assertEqual(requests[0]['memory'], {})
                self.assertTrue(requests[0]['sequence_context']['reset_memory'])
                self.assertEqual(requests[1]['memory']['prior_session'], 'session_000')
                self.assertFalse(requests[1]['sequence_context']['reset_memory'])
                run_ids.append(requests[0]['sequence_context']['run_id'])
                report = write_summary(data, output)
                self.assertTrue(report['evaluation_valid'])
                self.assertEqual(report['task_score'], 1.0)
                self.assertEqual(report['sequence_identity']['sequence_id'], sid)
            self.assertEqual(states[0], states[1])
            self.assertNotEqual(*run_ids)

    def test_missing_agent_reset_ack_invalidates_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.collection(Path(tmp) / 'scene')
            data = BenchmarkDataset(root, sequence_id='sequence_A')
            output = Path(tmp) / 'run'
            class NoResetAgent(Agent):
                def request(self,payload,on_wait=None):
                    return {}
            with self.assertRaisesRegex(ValueError, 'acknowledge'):
                SessionRunner(output, NoResetAgent(), max_steps=1).run_scene('synthetic_demo', data.sessions('synthetic_demo'),
                    data.canonical_state('synthetic_demo'), lambda _, state, sid: Env(state))
            report = write_summary(data, output)
            self.assertFalse(report['evaluation_valid'])
            self.assertIsNone(report['task_score'])

    def test_batch_dry_run_does_not_start_child_or_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.collection(Path(tmp) / 'scene')
            with patch('roboodyssey.launch.sequences.subprocess.Popen') as process:
                self.assertEqual(run_collection(select_sequences(root), Path(tmp) / 'output', [], dry_run=True), 0)
                process.assert_not_called()

    def test_batch_launches_fresh_processes_and_keeps_each_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.collection(Path(tmp) / 'scene')
            launched = []
            def spawn(command, **kwargs):
                launched.append((command, kwargs))
                target = Path(command[command.index('--run-directory') + 1])
                target.mkdir(parents=True)
                (target / 'summary.json').write_text(json.dumps({'evaluation_valid': True, 'task_score': .5, 'task_success_rate': .5, 'metric_version': 'progress_path_v3', 'metrics_complete': True}))
                from types import SimpleNamespace
                return SimpleNamespace(wait=lambda: 0, poll=lambda: 0, pid=99999999)
            output = Path(tmp) / 'results'
            with patch('roboodyssey.launch.sequences.subprocess.Popen', side_effect=spawn), \
                 patch('roboodyssey.launch.sequences.os.killpg', side_effect=ProcessLookupError):
                self.assertEqual(run_collection(select_sequences(root), output,
                    ['--dataset', str(root), '--all-sequences', '--agent-cmd', 'python agent.py']), 0)
            self.assertEqual(len(launched), 2)
            self.assertTrue(all(options['start_new_session'] for _, options in launched))
            summary = json.loads(next(output.glob('*/summary.json')).read_text())
            self.assertEqual(summary['task_score'], .5)
            self.assertEqual({r['identity']['sequence_id'] for r in summary['sequences']}, {'sequence_A', 'sequence_B'})

    def test_child_arguments_and_invalid_result_cannot_be_omitted(self):
        args = child_arguments(['--dataset=scene', '--sequence', 'A', '--sequence=B', '--output', 'runs',
            '--max-steps', '10', '--agent-cmd', 'python my_agent.py'], Path('scene/sequences/A'), Path('output/A'))
        self.assertNotIn('--sequence', args)
        self.assertIn('python my_agent.py', args)
        self.assertEqual(args[-4:], ['--dataset', 'scene/sequences/A', '--output', 'output/A'])
        good = {'exit_code': 0, 'summary': {'evaluation_valid': True, 'task_score': .75, 'metric_version':'progress_path_v3', 'metrics_complete':True}}
        self.assertEqual(aggregate_runs([good])['task_score'], .75)
        bad = {'exit_code': 1, 'summary': {'evaluation_valid': True, 'task_score': 1}}
        self.assertFalse(aggregate_runs([good, bad])['evaluation_valid'])
        self.assertIsNone(aggregate_runs([good, bad])['task_score'])

    def test_simplenav_same_scene_new_sequence_clears_local_history(self):
        from tests.test_simplenav_v5 import SimpleNavProtocolTests
        fixture = SimpleNavProtocolTests()
        agent = fixture.agent()
        agent.definitions = ['old name']
        agent.requests = [{'time': 'old', 'instruction': 'Old sequence.'}]
        agent.scene_id = 'scene'
        agent.history.images.append('old frame')
        response = agent.handle({'type': 'session_start', 'scene_id': 'scene',
            'instruction': 'Begin this sequence.', 'sequence_context': {'sequence_id': 'B', 'run_id': 'run_B', 'reset_memory': True}})
        self.assertEqual(response['sequence_reset_ack'], 'run_B')
        self.assertEqual(agent.requests, [])
        self.assertEqual(agent.definitions, [])
        self.assertEqual(agent.history.images, [])


if __name__ == '__main__':
    unittest.main()
