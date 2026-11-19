"""CPU-only timing/transport tests, with deterministic time except pipe polling."""
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import shlex
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

from roboodyssey.agents.agent_api import AgentProcess
from roboodyssey.runtime.observations import ObservationClock
from roboodyssey.runtime.session_runner import SessionRunner
from roboodyssey.scenes.isaac_env import IsaacVLNEnv


class ManualTime:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def sampler(self):
        return ObservationClock(clock=self, sleep=self.sleep)


class ObservationTests(unittest.TestCase):
    def test_fast_consumer_gets_two_frames_per_second_with_owned_buffers(self):
        timer = ManualTime()
        sampler = timer.sampler()
        rgb = np.zeros((2, 2, 3), np.uint8)
        sampler.set_source(lambda: ({'images': {'front': rgb}}, [{'visible_pixels': 3}]))
        first, evidence = sampler.take()
        rgb.fill(9)
        second, _ = sampler.take()
        third, _ = sampler.take()
        self.assertTrue(np.all(first['images']['front'] == 0))
        self.assertTrue(np.all(second['images']['front'] == 9))
        self.assertEqual(evidence, [{'visible_pixels': 3}])
        self.assertEqual([x['timing']['captured_monotonic_s'] for x in (first, second, third)],
                         [10.0, 10.5, 11.0])
        self.assertEqual(sampler.missed, 0)

    def test_slow_agent_receives_every_frame_while_action_response_is_pending(self):
        timer = ManualTime()
        source = {'position': 0}
        requests = []
        delivered = []

        class SlowAgent:
            def send_observation(self, payload):
                delivered.append(deepcopy(payload))

            def request(self, payload, on_wait):
                requests.append(deepcopy(payload))
                for moment in (10.5, 11.0, 11.2):
                    timer.now = moment
                    on_wait()
                return {'action': 'move_forward'}

        runner = SessionRunner('.', SlowAgent(), observation_clock=timer.sampler())
        runner._agent_io = StringIO()
        runner._begin_observations('a', lambda: ({'pose': dict(source)}, []))
        first, _ = runner.observations.take()
        runner._request({'type': 'step', 'task_id': 'a', 'observation': first})
        self.assertEqual(runner.observations.frame_id, 3)
        self.assertEqual([r['observation']['timing']['captured_monotonic_s'] for r in delivered], [10.5, 11.0])
        self.assertEqual([r['type'] for r in delivered], ['observation', 'observation'])
        self.assertFalse(hasattr(runner.observations, 'latest'))
        source['position'] = 1
        following, _ = runner.observations.take()
        self.assertEqual(following['pose']['position'], 1)
        self.assertEqual(following['timing']['captured_monotonic_s'], 11.5)
        self.assertEqual(len(requests), 1)  # Capture ticks never enqueue requests.
        self.assertEqual(requests[0]['observation']['pose']['position'], 0)
        rows = [json.loads(line) for line in runner._agent_io.getvalue().splitlines()]
        self.assertEqual([r['frame_id'] for r in rows if r.get('event') == 'sent'], [2, 3])
        self.assertFalse(any(r.get('event') == 'drop' for r in rows))
        self.assertIn('dispatched_monotonic_s', requests[0]['observation']['timing'])
        runner._end_observations()

    def test_blocked_main_thread_and_slow_render_skip_slots_without_bursts(self):
        timer = ManualTime()
        sampler = timer.sampler()
        def render():
            timer.sleep(.7)
            return {'images': {}}, []
        sampler.set_source(render)
        first, _ = sampler.take()
        self.assertEqual(first['timing']['missed_slots'], 1)
        self.assertIsNone(sampler.poll())
        timer.now = 12.2
        latest, _ = sampler.poll()
        self.assertEqual(latest['timing']['slot'], 4)
        self.assertEqual(latest['timing']['missed_slots'], 3)
        self.assertAlmostEqual(latest['timing']['lateness_seconds'], .2)
        sampler.poll()
        self.assertEqual(sampler.frame_id, 2)
        self.assertEqual(sampler.missed, 4)

    def test_task_switch_changes_evidence_and_stopping_disables_sampling(self):
        timer = ManualTime()
        sampler = timer.sampler()
        sampler.set_source(lambda: ({'images': {}}, [{'object_id': 'old'}]))
        sampler.poll()
        timer.now = 10.1
        sampler.set_source(lambda: ({'images': {}}, [{'object_id': 'new'}]))
        sample, evidence = sampler.take()
        self.assertEqual(sample['timing']['captured_monotonic_s'], 10.5)
        self.assertEqual(evidence, [{'object_id': 'new'}])
        sampler.set_source(None)
        timer.now = 15
        sampler.poll()
        self.assertEqual(sampler.frame_id, 2)
        with self.assertRaisesRegex(RuntimeError, 'not active'):
            sampler.take()

    def test_capture_error_is_propagated_instead_of_reusing_previous_frame(self):
        sampler = ManualTime().sampler()
        def broken():
            raise RuntimeError('camera disconnected')
        sampler.set_source(broken)
        with self.assertRaisesRegex(RuntimeError, 'camera disconnected'):
            sampler.take()

    def test_fresh_camera_sample_never_combines_cached_rgb_or_depth(self):
        fresh = np.ones((2, 2, 4), np.uint8)
        env = SimpleNamespace(
            _follow_held_object=lambda: None, render_once=lambda: 1,
            camera=SimpleNamespace(get_current_frame=lambda: {'rgba': fresh}),
            observe=lambda: {'images': {'front': np.zeros((2, 2, 3), np.uint8)}, 'depth': 'old'},
        )
        sample = IsaacVLNEnv.capture_observation(env)
        self.assertTrue(np.all(sample['images']['front'] == 1))
        self.assertIsNone(sample['depth'])
        env.render_once = lambda: None
        with self.assertRaisesRegex(RuntimeError, 'fresh observation render'):
            IsaacVLNEnv.capture_observation(env)

    def test_real_receiver_gets_every_frame_before_slow_inference_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / 'slow.py'
            script.write_text(
                'import json, sys, time\n'
                f'sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n'
                'from roboodyssey.agents.stream import serve\n'
                'received = []\n'
                'def observe(message):\n'
                '    received.append((message["observation"]["timing"]["frame_id"], time.monotonic()))\n'
                'def handle(request):\n'
                '    if request["type"] == "session_start": return {}\n'
                '    time.sleep(1.15)\n'
                '    return {"action": "no_op", "received": list(received),\n'
                '            "history": [o["timing"]["frame_id"] for o in request["observation_history"]]}\n'
                'serve(handle, on_observation=observe)\n')
            with AgentProcess(shlex.join([sys.executable, str(script)]), timeout=5) as agent:
                sampler = ObservationClock()
                runner = SessionRunner(tmp, agent, observation_clock=sampler)
                runner._request({'type': 'session_start'})
                runner._begin_observations('a', lambda: ({'images': {}}, []))
                initial, _ = sampler.take()
                response = runner._request({'type': 'step', 'task_id': 'a', 'observation': initial})
                self.assertEqual(response['action'], 'no_op')
                self.assertEqual([r[0] for r in response['received']], [1, 2, 3])
                gaps = [b[1]-a[1] for a, b in zip(response['received'], response['received'][1:])]
                self.assertTrue(all(.35 < gap < .65 for gap in gaps), gaps)
                self.assertEqual(response['observation_frame_id'], 1)
                following, _ = sampler.take()
                response = runner._request({'type': 'step', 'task_id': 'a', 'observation': following})
                self.assertEqual(response['history'], [1, 2, 3])
                self.assertEqual(response['observation_frame_id'], 4)
                self.assertEqual([r[0] for r in response['received']], [1, 2, 3, 4, 5, 6])
                runner._end_observations()

    def test_timeout_keeps_timing_audit_and_allows_sampler_cleanup(self):
        timer = ManualTime()
        class TimeoutAgent:
            def send_observation(self, payload):
                pass

            def request(self, payload, on_wait):
                timer.now += .5
                on_wait()
                raise TimeoutError('test timeout')
        runner = SessionRunner('.', TimeoutAgent(), observation_clock=timer.sampler())
        runner._agent_io = StringIO()
        runner._begin_observations('a', lambda: ({'images': {}}, []))
        initial, _ = runner.observations.take()
        with self.assertRaisesRegex(TimeoutError, 'test timeout'):
            runner._request({'type': 'step', 'observation': initial})
        runner._end_observations()
        self.assertFalse(hasattr(runner.observations, 'latest'))
        self.assertIsNone(runner.observations.source)
        rows = [json.loads(line) for line in runner._agent_io.getvalue().splitlines()]
        self.assertEqual(sum(r.get('event') == 'sample' for r in rows), 2)
        self.assertEqual(sum(r['direction'] == 'error' for r in rows), 1)

    def test_nonreading_agent_fails_delivery_instead_of_buffering_or_dropping(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp)/'blocked.py'
            script.write_text('import time\ntime.sleep(30)\n')
            with AgentProcess(shlex.join([sys.executable, str(script)])) as agent:
                agent._stream_ready = True
                started = time.monotonic()
                with self.assertRaisesRegex(TimeoutError, 'stdin blocked'):
                    agent.send_observation({'type': 'observation', 'observation': {'rgb': 'x' * 1000000}})
                self.assertLess(time.monotonic()-started, 1.5)

    def test_legacy_receiver_is_rejected_at_stream_handshake(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp)/'old.py'
            script.write_text('import json, sys\nfor line in sys.stdin:\n'
                              '    print(json.dumps({"request_id": json.loads(line)["request_id"]}), flush=True)\n')
            with AgentProcess(shlex.join([sys.executable, str(script)])) as agent:
                with self.assertRaisesRegex(ValueError, 'continuous observation'):
                    agent.request({'type': 'session_start'})

    def test_response_from_another_request_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp)/'wrong_id.py'
            script.write_text('import json, sys\nfor line in sys.stdin:\n'
                              '    print(json.dumps({"request_id": "wrong", "observation_stream_ack": True}), flush=True)\n')
            with AgentProcess(shlex.join([sys.executable, str(script)])) as agent:
                with self.assertRaisesRegex(ValueError, 'request_id'):
                    agent.request({'type': 'session_start'})

    def test_evidence_requires_a_frame_that_was_sent_in_this_task(self):
        timer = ManualTime()
        class StreamingAgent:
            def send_observation(self, payload):
                self.frame_id = payload['observation']['timing']['frame_id']
            def request(self, payload, on_wait):
                timer.sleep(.5)
                on_wait()
                return {'action': 'no_op', 'observation_frame_id': self.frame_id}
        agent = StreamingAgent()
        runner = SessionRunner('.', agent, observation_clock=timer.sampler())
        runner._begin_observations('a', lambda: ({'images': {}}, [{'visible_pixels': 10}]))
        initial, _ = runner.observations.take()
        runner._request({'type': 'step', 'task_id': 'a', 'observation': initial})
        self.assertEqual(runner._consumed_stream_evidence(), [{'visible_pixels': 10}])
        self.assertEqual(runner._consumed_stream_evidence(), [])
        runner._begin_observations('b', lambda: ({'images': {}}, []))
        initial, _ = runner.observations.take()
        agent.request = lambda *a, **kw: {'action': 'no_op', 'observation_frame_id': 2}
        with self.assertRaisesRegex(ValueError, 'not a sent frame'):
            runner._request({'type': 'step', 'task_id': 'b', 'observation': initial})


if __name__ == '__main__':
    unittest.main()
