from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from roboodyssey.data import load_sequence
from roboodyssey.artifacts.layout import session_dir, session_file, sha256_file
from roboodyssey.contracts.snapshots import StateBranch
from roboodyssey.runtime.policy import run_sequence
from tests.frozen_data import fixture_v3, goal, task, select_scenario, write_dataset
from tests.test_contextual_requests import Env


class PolicyRuntimeTests(unittest.TestCase):
    def test_private_policy_uses_real_success_and_capture_links(self):
        c, i, p = fixture_v3([task('ask', goal('book_red', 'answer', query='current_room'))])
        class SensorEnv(Env):
            def observe(self):
                return {'images': {'front': np.full((3, 4, 3), 17, np.uint8)},
                        'depth': np.ones((3, 4), np.float32), 'pose': self.get_agent_pose()}
        class Policy:
            resets = 0
            def reset(self):
                self.resets += 1
            def decide(self, task, observation, context):
                self.context = context
                # Private GT is available only via the explicit trusted context.
                assert 'ground_truth' not in observation and 'state' not in observation
                assert isinstance(observation['images']['front'], np.ndarray)
                return {'action': 'no_op', 'answer': 'study', 'done': True}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root/'data', select_scenario(c, p), c, i)
            policy = Policy()
            summary = run_sequence(load_sequence(root/'data'), policy, root/'run',
                                   env_factory=lambda _, state, sid: SensorEnv(state), max_steps=1)
            self.assertTrue(summary['evaluation_valid'])
            self.assertEqual(policy.resets, 1)
            frames = [json.loads(line) for line in (root/'run/observations/frames.jsonl').read_text().splitlines()]
            self.assertTrue(frames)
            frame = frames[0]
            sensor = frame['images']['front']
            self.assertEqual(sha256_file(root/'run'/sensor['file']), sensor['sha256'])
            self.assertEqual(np.load(root/'run'/sensor['file'], allow_pickle=False)[0,0,0], 17)
            folder = session_dir(root/'run', frame['scene_id'], frame['session_id'])
            action = json.loads(session_file(folder, 'action_changelog.jsonl').read_text().splitlines()[0])
            self.assertEqual(action['observation_frame_id'], frame['frame_id'])
            self.assertEqual(frame['state_source'], 'actual')

    def test_perception_policy_receives_only_raw_public_observation_and_public_context(self):
        from roboodyssey.runtime.policy import InProcessPolicyAgent
        class Perception:
            permission_profile = 'perception_policy'
            def decide(self, task, observation, context):
                self.context = context
                self.observation = observation
                observation['images']['front'][0, 0, 0] = 99
                return {'action': 'no_op', 'done': True}
        policy = Perception()
        adapter = InProcessPolicyAgent(policy)
        frame = np.ones((2, 3, 3), np.uint8)
        adapter.set_context({'scene_id': 's', 'session_id': 'session_000', 'task': {'task_id': 'ask'},
            'state': {'secret': 1}, 'ground_truth': {'secret': 2}, 'cache': {'secret': 3},
            'history': [{'secret': 4}], 'reference': object(), 'feedback': {'secret': 5},
            'raw_public_observation': {'images': {'front': frame}, 'timing': {'frame_id': 7},
                                       'action_feedback': {'success': True}}})
        response = adapter.request({'type': 'step', 'request_id': '1', 'observation': {}})
        self.assertEqual(policy.context, {'scene_id': 's', 'session_id': 'session_000', 'feedback': {'success': True}})
        self.assertEqual(frame[0, 0, 0], 1)
        self.assertEqual(response['observation_frame_id'], 7)
        self.assertEqual(response['action_source'], 'perception_policy')

    def test_state_branches_reject_cross_source_and_invalidate_cache(self):
        c, i, p = fixture_v3()
        from tests.frozen_data import runtime_cache
        cache = runtime_cache(i, p)
        state = {'objects': {o['object_id']: deepcopy(o) for o in i['metadata']['objects']}, 'held_object': None}
        branch = StateBranch(cache, state, source='planned')
        child = branch.fork()
        old = child.snapshot()
        changed = old.state
        changed['objects']['book_red']['room_id'] = 'kitchen'
        child.cached('map', lambda _: 'old')
        child.commit(changed, expected_version=old.version, event_id='move')
        self.assertEqual(child.cached('map', lambda _: 'new'), 'new')
        self.assertEqual(branch.snapshot().state['objects']['book_red']['room_id'], 'study')
        with self.assertRaisesRegex(ValueError, 'stale'):
            child.commit(changed, expected_version=old.version, event_id='stale')
        actual = StateBranch(cache, state, source='actual')
        with self.assertRaisesRegex(ValueError, 'readback'):
            actual.commit(changed, expected_version=actual.snapshot().version, event_id='fake')
        with self.assertRaisesRegex(ValueError, 'mixed'):
            actual.append_history({'success': True}, source='planned')


if __name__ == '__main__':
    unittest.main()
