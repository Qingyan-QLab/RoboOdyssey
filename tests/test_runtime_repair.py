from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from test_task_repair import fixture, transfer, question, completed_event
from benchmark.artifacts.layout import write_json as write
from benchmark.runtime.session_runner import SessionRunner
from benchmark.scenes.dataset_loader import SessionRecord
from benchmark.scenes.state_composer import compose_session_state
from benchmark.evaluation.success import evaluate_embodied_task
from benchmark.agents.protocol import validate_response


class Agent:
    def __init__(self, inspect=False):
        self.requests = []
        self.inspect = inspect

    def request(self, payload, on_wait=None):
        self.requests.append(deepcopy(payload))
        if payload['type'] == 'session_start':
            return {}
        if payload['type'] == 'qa':
            if self.inspect and payload.get('allow_navigation_actions') and not payload.get('previous_action'):
                return {'action': {'action': 'move_forward'}}
            return {'answer': 'bedroom'}
        return {'action': {'action': 'no_op'}, 'done': True}


class Environment:
    def __init__(self, state):
        self.state = deepcopy(state)
        self.pose = {'position': [0, 0, 1.5], 'orientation': [0, 0, 0, 1]}

    def apply_world_state(self, state, preserve_runtime=True): self.state = deepcopy(state)
    def get_agent_pose(self): return deepcopy(self.pose)
    def set_agent_pose(self, pose, update_reset_pose=True): self.pose = deepcopy(pose)
    def observe(self): return {'images': {}}
    def runtime_state(self): return deepcopy(self.state)
    def set_task(self, task): pass
    def state_changes(self, initial): return []
    def close(self): pass
    def step(self, action):
        if action['action'] == 'move_forward': self.pose['position'][0] += 1
        return self.observe(), 0, False, False, {'success': True}


class RuntimeIntegrationTests(unittest.TestCase):
    def test_failure_repairs_bound_history_and_current_qa_uses_actual_state(self):
        cache, state = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / 'cache.json', cache)
            records = []
            for i, spec in enumerate((transfer(), question('room_after', history=True), question())):
                template, task, gt = spec
                sid = f'session_{i:03d}'
                issued_at = '2026-11-05T09:10:00+08:00' if i == 0 else f'2026-11-{5 + i:02d}T09:00:00+08:00'
                task['issued_at'] = template['issued_at'] = issued_at
                task.setdefault('task_family', 'object_transfer')
                folder = root / sid; folder.mkdir()
                write(folder / 'gt.json', {'tasks': [gt | {'task_id': task['task_id']}]})
                write(folder / 'templates.json', {'schema_version': 1, 'session_id': sid, 'tasks': [template]})
                records.append(SessionRecord('test', folder, {'session_id': sid, 'tasks': [task],
                    'time_context': {'kind': 'scenario_calendar', 'starts_at': issued_at}}, {
                    'ground_truth': folder / 'gt.json', 'task_templates': folder / 'templates.json',
                    'runtime_scene_cache': root / 'cache.json'}))
            agent = Agent(inspect=True)
            output = root / 'run'
            SessionRunner(output, agent, max_steps=2).run_scene('test', records, state,
                                                                lambda *_: Environment(state))
            metrics = [json.loads((output / 'scenes/test/sessions' / r.session_id / 'metrics.json').read_text()) for r in records]
            self.assertFalse(metrics[0]['success'])
            self.assertEqual(metrics[1]['task_results'][0]['repair_status'], 'ready')
            self.assertEqual(metrics[1]['task_score'], 1)
            self.assertEqual(metrics[1]['tasks_excluded'], 0)
            self.assertTrue(metrics[2]['success'])
            recalled = [r for r in agent.requests if r.get('session_id') == 'session_001' and r['type'] == 'qa']
            self.assertEqual(len(recalled), 1)
            self.assertIn('when that task ended', recalled[0]['question'])
            for request in agent.requests:
                session_index = int(request['session_id'].split('_')[-1])
                expected_time = records[session_index].tasks[0]['issued_at']
                if request['type'] == 'session_start':
                    self.assertEqual(request['time_context'], {'kind': 'scenario_calendar', 'starts_at': expected_time})
                else:
                    self.assertEqual(request['current_time'], expected_time)
            questions = [r for r in agent.requests if r['type'] == 'qa']
            self.assertEqual(len(questions), 3)
            self.assertNotIn('allow_navigation_actions', questions[0])
            self.assertTrue(questions[1]['allow_navigation_actions'])
            for request in questions:
                text = json.dumps(request)
                self.assertNotIn('ground_truth', text)
                self.assertNotIn('nightstand', request['question'])
                self.assertNotIn('bedroom', request['question'])
            audit = output / 'scenes/test/sessions/session_002'
            self.assertTrue((audit / 'runtime_review.html').is_file())
            record = json.loads((audit / 'runtime_tasks.jsonl').read_text())
            self.assertEqual(record['runtime_ground_truth']['expected_outcome']['answer'], 'bedroom')
            self.assertEqual(record['state_before'], state)

    def test_held_object_in_destination_cannot_pass_without_place(self):
        _, task, _ = transfer()
        gt = {'expected_outcome': {'state_changes': [{'object_id': 'book',
              'relation_after': {'type': 'in_room', 'room_id': 'kitchen'}}]},
              'success_criteria': {'require_released': True, 'require_place_event': True}}
        state = {'held_object': 'book', 'objects': {'book': {'room_id': 'kitchen', 'held': True}}}
        event = {'action': {'action': 'place'}, 'result': {'success': True, 'object_id': 'book'}}
        self.assertFalse(evaluate_embodied_task(task, gt, [event], state)['success'])
        state['held_object'] = None; state['objects']['book']['held'] = False
        self.assertFalse(evaluate_embodied_task(task, gt, [], state)['success'])
        self.assertTrue(evaluate_embodied_task(task, gt, [event], state)['success'])

    def test_named_place_scoring_keeps_established_location(self):
        fixed = {'world_aabb': {'min': [0,0,0], 'max': [.2,.2,.2]}}
        task = {'task_type': 'navigation'}
        gt = {'target': {'object_ids': ['cup']}, 'fixed_target_geometry': fixed,
              'success_criteria': {'target_distance_type': 'aabb_2d', 'position_tolerance_m': .5},
              'expected_outcome': {}}
        state = {'objects': {'cup': {'geometry': {'world_aabb': {'min': [5,5,0], 'max': [6,6,1]}}}}}
        self.assertTrue(evaluate_embodied_task(task, gt, [{'after': {'position': [0,0,1]}}], state)['success'])

    def test_vlm_mutation_updates_geometry_pose_and_releases_object(self):
        _, state = fixture()
        state['held_object'] = 'book'
        state['objects']['book']['held'] = True
        moved = deepcopy(state['objects']['book']['geometry'])
        moved['world_matrix4x4'][3][0] = 9
        mutation = {'changes': [{'object_id': 'book', 'relation_after': {'room_id': 'kitchen'}}],
                    'geometry_after': moved}
        result = compose_session_state(state, mutation)
        self.assertEqual(result['objects']['book']['pose']['position'][0], 9)
        self.assertEqual(result['objects']['book']['room_id'], 'kitchen')
        self.assertFalse(result['objects']['book']['held'])
        self.assertIsNone(result['held_object'])
        self.assertTrue(state['objects']['book']['held'])

    def test_qa_protocol_allows_inspection_but_not_world_changes(self):
        validate_response('qa', {'action': {'action': 'move_forward'}})
        for name in ('pick', 'place', 'open', 'close'):
            with self.assertRaises(ValueError): validate_response('qa', {'action': {'action': name}})
        with self.assertRaises(ValueError): validate_response('qa', {})

    def test_navigation_occupancy_tracks_moved_and_held_objects(self):
        from benchmark.scenes.isaac_env import IsaacVLNEnv
        from benchmark.contracts.doors import POLICY
        from test_task_repair import entity
        env = IsaacVLNEnv.__new__(IsaacVLNEnv)
        env._np = np
        env._environment_setup = {'policy': POLICY, 'doors': []}
        env._metadata = {'rooms': [{'room_id': 'room', 'boundary_polygon': [[0,0],[5,0],[5,5],[0,5]]}]}
        book = entity('book', 'book', 'room', 1)
        book['geometry']['world_aabb'] = {'min': [1,1,0], 'max': [1.2,1.2,.2]}
        book['pose'] = {'position': [3,0,0]}
        env._objects = {'book': book}; env._held_object = None
        env._resolution = .5; env._navigation = {'robot_radius': 0}
        env._refresh_vlm_navigation()
        self.assertFalse(env._static_base[3,3])  # Original [1,1] footprint.
        self.assertTrue(env._static_base[3,7])   # Actual [3,1] footprint.
        env._held_object = 'book'; book['held'] = True
        env._refresh_vlm_navigation()
        self.assertFalse(env._static_base[3,7])


if __name__ == '__main__': unittest.main()
