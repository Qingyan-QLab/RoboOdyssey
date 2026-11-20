"""Single-upload response-chain behavior with synthetic frames and simulated API."""
import base64
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from roboodyssey.agents.openai_api import APIError, OpenAIAPIAgent, SYSTEM, load_config, post_json
from roboodyssey.agents.stream import serve

ROOT = Path(__file__).resolve().parents[1]


def observation(frame_id):
    return {'images': {'front': {'encoding': 'base64', 'dtype': 'uint8', 'shape': [2, 2, 3],
            'data': base64.b64encode(bytes([frame_id % 256] * 12)).decode()}},
            'timing': {'frame_id': frame_id, 'captured_monotonic_s': frame_id * .5},
            'pose': {'position': [1, 2, 0], 'orientation': [0, 0, 0, 1], 'yaw_degrees': 15},
            'holding_object': True, 'held_object_name': 'reading book',
            'action_feedback': {'success': True, 'collision': False, 'executed_distance': .25}}


def start(session='s1', run_id='run1', reset=True):
    return {'type': 'session_start', 'scene_id': 'scene', 'session_id': session, 'request_id': 'start-' + session,
            'instruction': 'Help in this interaction.', 'scene_briefing': ['Call this area the study.'],
            'time_context': {'kind': 'scenario_calendar', 'starts_at': '2026-11-05T09:00:00+08:00'},
            'sequence_context': {'run_id': run_id, 'sequence_id': 'seq', 'reset_memory': reset}}


def step(frame_id=1, **extra):
    return {'type': 'step', 'scene_id': 'scene', 'session_id': 's1', 'task_id': 't1',
            'request_id': str(frame_id), 'instruction': 'Go to the reading book.',
            'name_introductions': ['Call the red book the reading book.'],
            'current_time': '2026-11-05T09:01:00+08:00', 'capabilities': ['navigation'],
            'response_mode': 'actions', 'observation': observation(frame_id),
            'sequence_context': {'run_id': 'run1'}, **extra}


def output(action='no_op', *, answer=None, done=False, **args):
    return {'action': {'action': action, 'distance': None, 'angle_degrees': None, 'target': None, **args},
            'answer': answer, 'done': done}


def result(value=None, *, usage=True, response_id='response-1'):
    data = {'id': response_id, 'model': 'verified-test-snapshot', 'status': 'completed',
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(output() if value is None else value)}]}]}
    if usage:
        data['usage'] = {'input_tokens': 20, 'output_tokens': 8, 'total_tokens': 28,
                         'input_tokens_details': {'cached_tokens': 5}, 'output_tokens_details': {'reasoning_tokens': 2}}
    return data


def delta(body):
    content = body['input'][-1]['content']
    return json.loads(content[0]['text']) if content[0]['type'] == 'input_text' else {}


class APIAgentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        env = patch.dict(os.environ, {'TEST_AGENT_KEY': 'secret-test-key-do-not-log'})
        env.start()
        self.addCleanup(env.stop)
        self.requests = []

    def config(self, **extra):
        path = self.root / 'agent.json'
        path.write_text(json.dumps({'model': 'verified-test-model', 'api_key_env': 'TEST_AGENT_KEY',
                                   'log_dir': 'artifacts', **extra}))
        return load_config(path)

    def agent(self, transport=None, **extra):
        def fake(url, body, api_key, timeout):
            self.requests.append(deepcopy(body))
            return result(response_id='response-' + str(len(self.requests)))
        return OpenAIAPIAgent(self.config(**extra), transport=transport or fake)

    def calls(self, agent):
        return [json.loads(line) for line in (agent.log_dir / 'calls.jsonl').read_text().splitlines()]

    def test_config_is_responses_only_and_rejects_old_memory_history_and_retry_fields(self):
        with patch.dict(os.environ, {}, clear=True):
            config = self.config()
        self.assertFalse((self.root / 'artifacts').exists())
        self.assertEqual(config['log_dir'], str(self.root / 'artifacts'))
        for extra in ({'model': 'REPLACE_WITH_VERIFIED_MODEL_ID'}, {'api_key': 'secret'}, {'api': 'chat_completions'},
                      {'history_frames': 0}, {'memory_max_chars': 1000}, {'max_retries': 0},
                      {'request_budget_seconds': 120}, {'feedback': 'detailed'},
                      {'endpoint': 'https://key@api.example/v1'}, {'endpoint': 'https://api.example/v1?key=secret'},
                      {'endpoint': 'https://api.example:invalid/v1'}, {'endpoint': 'https://api.example/v1/responses'}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.config(**extra)

    def test_cli_validation_without_credentials_or_outputs(self):
        self.config()
        env = dict(os.environ)
        env.pop('TEST_AGENT_KEY', None)
        checked = subprocess.run([sys.executable, str(ROOT / 'examples/openai_api_agent.py'),
                                  '--config', str(self.root / 'agent.json'), '--check-config'],
                                 capture_output=True, text=True, env=env, timeout=10, cwd=self.root)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(json.loads(checked.stdout)['feedback'], 'minimal')
        self.assertFalse((self.root / 'artifacts').exists())

    def test_chain_uploads_prompt_initial_state_and_task_once_and_each_current_image_once(self):
        agent = self.agent(reasoning_effort='configured-effort')
        ack = agent.handle(start())
        self.assertEqual(ack['sequence_reset_ack'], 'run1')
        self.assertFalse(self.requests)
        self.assertNotIn('memory_update', ack)
        for i in (1, 2, 3):
            response = agent.handle(step(i, observation_history=[observation(0)]))
            self.assertNotIn('agent_status', response)
            self.assertNotIn('memory_update', response)
        first, second, third = self.requests
        self.assertEqual(first['input'][0]['role'], 'developer')
        self.assertEqual(first['input'][0]['content'][0]['text'], SYSTEM)
        self.assertNotIn('instructions', first)
        self.assertNotIn('previous_response_id', first)
        self.assertEqual(second['previous_response_id'], 'response-1')
        self.assertEqual(third['previous_response_id'], 'response-2')
        self.assertIn('initial_state', delta(first))
        self.assertIn('task', delta(first))
        self.assertIn('sessions', delta(first))
        self.assertEqual(delta(second), {})
        self.assertEqual(delta(third), {})
        images = [item['image_url'] for b in self.requests for m in b['input'] for item in m['content'] if item['type'] == 'input_image']
        self.assertEqual(len(images), 3)
        self.assertEqual(len(set(images)), 3)
        for body in self.requests:
            self.assertTrue(body['store'])
            self.assertEqual(body['truncation'], 'disabled')
            self.assertNotIn('temperature', body)
            self.assertEqual(body['reasoning'], {'effort': 'configured-effort'})
        self.assertEqual(agent.summary['system_prompt_uploads'], 1)
        self.assertEqual(agent.summary['initial_state_uploads'], 1)
        self.assertEqual(agent.summary['uploaded_images'], 3)

    def test_nested_input_boundary_and_no_pose_memory_context_or_ids_after_first_call(self):
        agent = self.agent()
        agent.handle(start())
        agent.handle(step())
        request = step(2)
        request['memory'] = request['ground_truth'] = request['recent_action'] = 'HIDDEN_SENTINEL'
        obs = request['observation']
        obs['pose'] = obs['held_object_name'] = obs['depth'] = obs['nearby_manipulable_objects'] = 'HIDDEN_SENTINEL'
        obs['images']['overview'] = 'HIDDEN_SENTINEL'
        obs['action_feedback']['reason'] = obs['action_feedback']['object_id'] = 'HIDDEN_SENTINEL'
        response = agent.handle(request)
        self.assertNotIn('agent_status', response)
        wire = json.dumps(self.requests[1])
        for absent in ('HIDDEN_SENTINEL', 'pose', 'holding_object', 'held_object_name', 'recent_action', 'memory', 'task_id', 'scene_id'):
            self.assertNotIn(absent, wire)

    def test_first_state_nested_whitelist_and_name_deduplication(self):
        agent = self.agent()
        request = step()
        request['observation']['pose']['room_id'] = 'PRIVATE_SENTINEL'
        agent.handle(start())
        agent.handle(request)
        self.assertNotIn('PRIVATE_SENTINEL', json.dumps(self.requests[0]))
        request = step(2, task_id='t2', instruction='Return to the reading book.')
        agent.handle(request)
        self.assertEqual(delta(self.requests[1])['task']['instruction'], 'Return to the reading book.')
        self.assertNotIn('name_introductions', delta(self.requests[1]))
        self.assertNotIn('initial_state', delta(self.requests[1]))

    def test_cross_session_preserves_chain_and_new_run_discards_it(self):
        agent = self.agent()
        agent.handle(start())
        agent.handle(step())
        agent.handle(start(session='s2', reset=False))
        agent.handle(step(2, session_id='s2', task_id='t2'))
        self.assertEqual(self.requests[1]['previous_response_id'], 'response-1')
        self.assertNotIn('scene_briefing', delta(self.requests[1])['sessions'][0])
        self.assertNotIn('initial_state', delta(self.requests[1]))
        agent.handle(start(run_id='run2'))
        agent.handle(step(1, sequence_context={'run_id': 'run2'}))
        self.assertNotIn('previous_response_id', self.requests[2])
        self.assertEqual(self.requests[2]['input'][0]['role'], 'developer')
        self.assertIn('initial_state', delta(self.requests[2]))
        self.assertEqual(agent.summary['run_ids'], ['run1', 'run2'])

    def test_missing_reset_stale_run_and_duplicate_frames_fail_closed(self):
        for request in (start(run_id='run2', reset=False), step(2, sequence_context={'run_id': 'run2'}), step(1)):
            with self.subTest(request=request['type']):
                agent = self.agent()
                agent.handle(start())
                agent.handle(step())
                before = len(self.requests)
                response = agent.handle(request)
                self.assertEqual(response['agent_status'], 'integration_error')
                self.assertEqual(len(self.requests), before)

    def test_minimal_feedback_only_collision_or_discrete_action_success(self):
        for action, expected in [('move_forward', {'collision': True}), ('pick', {'success': False}),
                                 ('place', {'success': False}), ('turn_left', {}), ('reset', {'success': True})]:
            with self.subTest(action=action):
                bodies = []
                def fake(url, body, key, timeout):
                    bodies.append(body)
                    return result(output(action, **({'target': 'reading book'} if action == 'pick' else {})), response_id='r' + str(len(bodies)))
                agent = self.agent(fake)
                agent.handle(start()); agent.handle(step())
                obs = observation(2)
                obs.update(reset_applied=True)
                obs['action_feedback'].update(success=False, collision=True, reason='private-extra', executed_distance=.17)
                response = agent.handle(step(2, observation=obs))
                self.assertNotIn('agent_status', response)
                self.assertEqual(delta(bodies[1]).get('feedback', {}), expected)
                self.assertNotIn('executed_distance', json.dumps(bodies[1]))
                self.assertNotIn('private-extra', json.dumps(bodies[1]))

    def test_feedback_can_be_disabled_and_missing_feedback_does_not_imply_success(self):
        for mode in ('none', 'minimal'):
            bodies = []
            def fake(url, body, key, timeout):
                bodies.append(body)
                return result(output('pick', target='reading book'), response_id='r' + str(len(bodies)))
            agent = self.agent(fake, feedback=mode)
            agent.handle(start()); agent.handle(step())
            obs = observation(2)
            obs['action_feedback'] = {} if mode == 'minimal' else {'success': True, 'collision': True}
            agent.handle(step(2, observation=obs))
            self.assertNotIn('feedback', delta(bodies[1]))

    def test_bad_model_output_never_repaired_retried_or_marked_unsupported(self):
        private = output('pick', target='reading book'); private['action']['object_id'] = 'private'
        old = output(); old['memory_update'] = {'facts': []}
        cases = [output('pick'), output('move_forward', distance=True), output('move_forward', distance=float('nan')),
                 output('move_forward', distance=11), private, old, {'action': 'no_op'}]
        for value in cases:
            with self.subTest(value=str(value)[:80]):
                calls = []
                def fake(*args):
                    calls.append(1)
                    return result(value)
                agent = self.agent(fake)
                agent.handle(start())
                response = agent.handle(step())
                self.assertEqual(response['agent_status'], 'integration_error')
                self.assertEqual(calls, [1])
                self.assertIsNone(agent.previous_response_id)

    def test_answer_schema_modes_and_forbidden_actions(self):
        cases = [(output(answer='study', done=True), step(response_mode='actions_and_answer', allowed_actions=['no_op']), True),
                 (output('pick', target='reading book'), step(allowed_actions=['no_op']), False),
                 (output(done=True), step(response_mode='actions_and_answer'), False)]
        for value, request, valid in cases:
            agent = self.agent(lambda *args, value=value: result(value), output_format='json_object')
            agent.handle(start())
            response = agent.handle(request)
            self.assertEqual('agent_status' not in response, valid)

    def test_incomplete_refusal_missing_response_id_or_bad_json_are_errors(self):
        truncated = result(); truncated['status'] = 'incomplete'
        refusal = result(); refusal['output'][0]['content'] = [{'type': 'refusal', 'refusal': 'no'}]
        malformed = result(); malformed['output'][0]['content'][0]['text'] = '```json\n{}\n```'
        missing = result(); missing.pop('id')
        for value in (truncated, refusal, malformed, missing):
            agent = self.agent(lambda *args, value=value: value)
            agent.handle(start())
            response = agent.handle(step())
            self.assertEqual(response['agent_status'], 'integration_error')
            self.assertEqual(len(self.calls(agent)), 1)

    def test_http_failure_is_not_replayed_and_secret_is_not_logged(self):
        for status in (401, 429, 503):
            error = HTTPError('https://api.example', status, 'secret-test-key-do-not-log', {}, BytesIO(b'secret-test-key-do-not-log'))
            agent = self.agent(post_json)
            agent.handle(start())
            with patch('roboodyssey.agents.openai_api.urlopen', side_effect=error) as send:
                response = agent.handle(step())
            self.assertEqual(send.call_count, 1)
            self.assertEqual(response['reason_code'], 'http_' + str(status))
            agent.close()
            self.assertNotIn('secret-test-key-do-not-log', '\n'.join(p.read_text() for p in agent.log_dir.iterdir()))

    def test_deadline_stalled_transport_late_result_cannot_change_chain(self):
        release, exited = threading.Event(), threading.Event()
        calls = []
        def fake(*args):
            calls.append(1)
            release.wait(2)
            exited.set()
            return result()
        agent = self.agent(fake, request_budget_seconds=.05)
        agent.handle(start())
        began = time.monotonic()
        try:
            response = agent.handle(step())
            self.assertEqual(response['reason_code'], 'api_deadline_exceeded')
            self.assertLess(time.monotonic() - began, 1)
            self.assertEqual(agent.handle(step(2))['reason_code'], 'agent_already_failed')
        finally:
            release.set()
        self.assertTrue(exited.wait(1))
        self.assertEqual(len(calls), 1)
        self.assertIsNone(agent.previous_response_id)
        self.assertEqual(len(self.calls(agent)), 1)

    def test_usage_reports_full_context_without_substituting_upload_size(self):
        bodies = []
        def fake(url, body, key, timeout):
            bodies.append(body)
            reply = result(response_id='r' + str(len(bodies)))
            reply['usage']['input_tokens'] = len(bodies) * 1000
            reply['usage']['input_tokens_details']['cached_tokens'] = (len(bodies) - 1) * 800
            return reply
        agent = self.agent(fake)
        agent.handle(start()); agent.handle(step()); agent.handle(step(2))
        self.assertEqual(agent.summary['usage_totals']['input_tokens'], 3000)
        self.assertEqual(agent.summary['usage_totals']['cached_input_tokens'], 800)
        self.assertEqual(agent.summary['system_prompt_uploads'], 1)
        self.assertGreater(self.calls(agent)[0]['uploaded_text_chars'], self.calls(agent)[1]['uploaded_text_chars'])
        self.assertIsNone(agent.summary['estimated_cost'])

    def test_missing_and_optional_usage_metadata(self):
        for present in (False, True):
            reply = result(usage=present)
            if present:
                reply['usage']['input_tokens_details'] = 'unexpected extension'
                reply['model'] = {'unexpected': 'metadata'}
            agent = self.agent(lambda *args: reply)
            agent.handle(start())
            response = agent.handle(step())
            self.assertNotIn('agent_status', response)
            self.assertEqual(agent.summary['usage_missing_attempts'], int(not present))
            if present:
                self.assertEqual(agent.summary['model_ids'], [])
                self.assertNotIn('cached_input_tokens', agent.summary['usage_totals'])

    def test_stream_reads_pending_frames_but_api_uploads_only_current_frame(self):
        entered, release, received = threading.Event(), threading.Event(), threading.Event()
        messages, replies = Queue(), Queue()
        frames_seen, bodies = [], []
        class Reader:
            def __iter__(self):
                while True:
                    value = messages.get()
                    if value is None:
                        return
                    yield json.dumps(value) + '\n'
        class Writer:
            def write(self, line):
                replies.put(json.loads(line))
            def flush(self):
                pass
        def fake(url, body, key, timeout):
            bodies.append(deepcopy(body))
            entered.set()
            if not release.wait(2):
                raise APIError('test_release_missing')
            return result(response_id='r' + str(len(bodies)))
        def observe(payload):
            i = payload['observation']['timing']['frame_id']
            frames_seen.append(i)
            if i == 3:
                received.set()
        agent = self.agent(fake)
        thread = threading.Thread(target=serve, args=(agent.handle,), kwargs={
            'history_frames': 0, 'on_observation': observe, 'input_stream': Reader(), 'output_stream': Writer()}, daemon=True)
        thread.start()
        try:
            messages.put({**start(), 'observation_stream': {'fps': 2.0, 'mode': 'push'}})
            self.assertTrue(replies.get(timeout=2)['observation_stream_ack'])
            messages.put(step())
            self.assertTrue(entered.wait(2))
            for i in (2, 3):
                messages.put({'type': 'observation', 'scene_id': 'scene', 'session_id': 's1',
                              'task_id': 't1', 'request_id': '1', 'observation': observation(i)})
            self.assertTrue(received.wait(1))
            self.assertEqual(frames_seen, [1, 2, 3])
            release.set()
            first = replies.get(timeout=2)
            self.assertEqual(first['observation_frame_id'], 1)
            messages.put(step(4))
            self.assertEqual(replies.get(timeout=2)['observation_frame_id'], 4)
            self.assertEqual([[f['frame_id'] for f in r['input_frames']] for r in self.calls(agent)], [[1], [4]])
        finally:
            release.set(); messages.put(None); thread.join(timeout=3)
        self.assertFalse(thread.is_alive())

    def test_transport_auth_and_instance_isolation(self):
        raw = BytesIO(json.dumps(result()).encode()); raw.headers = {'x-request-id': 'req-test'}
        with patch('roboodyssey.agents.openai_api.urlopen', return_value=raw) as send:
            value = post_json('https://api.example/v1/responses', {'model': 'test'}, 'secret', 3)
        self.assertEqual(value['_http_request_id'], 'req-test')
        self.assertEqual(send.call_args.args[0].headers['Authorization'], 'Bearer secret')
        self.assertEqual(send.call_args.kwargs['timeout'], 3)
        self.assertNotEqual(self.agent().log_dir, self.agent().log_dir)


if __name__ == '__main__':
    unittest.main()
