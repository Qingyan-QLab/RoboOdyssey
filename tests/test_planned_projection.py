from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from roboodyssey.contracts.contextual import instantiate_request_v3
from roboodyssey.contracts.planning import PlannedProjection, project_sequence
from roboodyssey.data import load_sequence
from tests.frozen_data import SOURCE, fixture_v3, goal, task, select_scenario, write_dataset


class PlannedProjectionTests(unittest.TestCase):
    def fixture(self, tasks=None, second=None):
        c, inputs, plan = fixture_v3(tasks, second=second)
        result = select_scenario(c, plan)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)/'data'
        write_dataset(root, result, c, inputs)
        projection = PlannedProjection(load_sequence(root))
        before = projection.branch.snapshot()
        state = before.state
        state['robot_pose']['position'] = [2.5, 1.5, 0]
        projection.branch.commit(state, expected_version=before.version, event_id='fixture_start')
        return projection, result, root

    def project(self, projection, session):
        return [projection.task(session['public']['session_id'], *rows)
                for rows in zip(session['public']['tasks'], session['ground_truth'], session['templates'])]

    def test_current_answer_is_planned_and_never_location_evidence(self):
        _, _, root = self.fixture([task('ask', goal('book_red', 'answer', query='current_room'))])
        result = project_sequence(root)
        self.assertEqual(result['status'], 'projected')
        self.assertEqual(result['tasks'][0]['goals'][0]['expected_answer'], 'study')
        self.assertFalse(result['physics_execution_verified'])
        self.assertFalse(result['formal_score_reference'])
        event = result['planned_history'][0]
        self.assertTrue(event['assumed_dispatch'])
        self.assertFalse(event['assumed_location_experience'])
        self.assertNotIn('success', event)
        self.assertNotIn('dispatched', event)

    def test_transfer_history_uses_shared_predispatch_semantics_and_task_boundaries(self):
        projection, result, _ = self.fixture(second=[task('ask', goal('book_red', 'answer',
            query='room_after', history=SOURCE))])
        first = self.project(projection, result['sessions'][0])[0]
        second = self.project(projection, result['sessions'][1])[0]
        self.assertEqual(second['goals'][0]['expected_answer'], 'kitchen')
        event = projection.branch.history()[0]
        self.assertEqual(event['before'], first['before'])
        self.assertEqual(event['after'], first['after'])
        self.assertEqual(event['before']['objects']['book_red']['room_id'], 'study')
        self.assertEqual(event['after']['objects']['book_red']['room_id'], 'kitchen')
        self.assertEqual(first['planned_repair']['state_source'], 'planned')

    def test_already_satisfied_source_converts_history_question(self):
        projection, result, _ = self.fixture(second=[task('ask', goal('book_red', 'answer',
            query='room_after', history=SOURCE))])
        before = projection.branch.snapshot()
        state = before.state
        state['objects']['book_red'].update(room_id='kitchen', held=False,
            geometry={'world_aabb': {'min': [4.,1.,0.], 'max': [4.2,1.2,.2]}})
        projection.branch.commit(state, expected_version=before.version, event_id='already_in_room')
        first = self.project(projection, result['sessions'][0])[0]
        second = self.project(projection, result['sessions'][1])[0]
        self.assertEqual(first['status'], 'already_satisfied')
        self.assertFalse(projection.branch.history()[0]['assumed_dispatch'])
        self.assertEqual(second['goals'][0]['expected_answer'], 'kitchen')
        self.assertEqual(second['planned_repair']['conversions'][0]['query_after'], 'current_room')

    def test_indeterminate_projection_is_atomic_and_sources_are_not_interchangeable(self):
        projection, result, _ = self.fixture()
        session = result['sessions'][0]
        public, truth, template = session['public']['tasks'][0], session['ground_truth'][0], session['templates'][0]
        before = projection.branch.snapshot()
        # Fail after semantic repair, when route generation begins.
        projection._move = lambda *args: (_ for _ in ()).throw(ValueError('no feasible placement'))
        with self.assertRaisesRegex(ValueError, 'no feasible placement'):
            projection.task('session_000', public, truth, template)
        self.assertEqual(projection.branch.snapshot(), before)
        self.assertEqual(projection.branch.history(), [])
        self.assertEqual(projection.ledger, {})
        for event in ({'state_source': 'planned'}, {'assumed_success': True}):
            with self.assertRaisesRegex(ValueError, 'cannot be mixed'):
                instantiate_request_v3(public, truth, template, projection.cache, before.state, [event])
        with self.assertRaisesRegex(ValueError, 'cannot be mixed'):
            instantiate_request_v3(public, truth, template, projection.cache, before.state,
                                   [{'success': True}], state_source='planned')


if __name__ == '__main__':
    unittest.main()
