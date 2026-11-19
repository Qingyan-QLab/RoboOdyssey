

from copy import deepcopy
import unittest

from benchmark.contracts.state import build_runtime_cache
from benchmark.legacy.task_repair import instantiate_task, object_reference
from benchmark.legacy.task_history import execution_event, execution_events
from benchmark.contracts.legacy_templates import freeze_template
from benchmark.contracts.language import historical_reference


def entity(oid, category, room, x=0):
    return {'object_id': oid, 'category': category, 'room_id': room,
            'geometry': {'world_matrix4x4': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[x,0,0,1]],
                         'world_aabb': {'min': [x,0,0], 'max': [x+.2,.2,.2]}}}


def fixture():
    objects = [entity('book', 'book', 'bedroom'), entity('other', 'book', 'study', 3),
               entity('cup', 'cup', 'study', 4)]
    metadata = {'scene': {'scene_id': 'test'}, 'objects': objects,
                'rooms': [{'room_id': r, 'room_type': r} for r in ('bedroom', 'study', 'kitchen')]}
    understanding = {'objects': [
        {'object_id': 'book', 'identity_attributes': {'color': 'red'},
         'referring_expressions': [{'text': 'the red book on the nightstand', 'unique': True}]},
        {'object_id': 'other', 'identity_attributes': {'color': 'blue'}, 'referring_expressions': []},
        {'object_id': 'cup', 'referring_expressions': [{'text': 'the cup', 'unique': True}]}],
        'rooms': [{'room_id': r['room_id'], 'description': r['room_id']} for r in metadata['rooms']]}
    return build_runtime_cache(metadata, understanding), {'objects': {o['object_id']: o for o in objects}}


def transfer(oid='book', text='Pick up the red book on the nightstand and place it on the floor in the kitchen.'):
    proposal = {'task_id': 'task_001', 'task_type': 'manipulation', 'target_object_id': oid,
                'destination_room_id': 'kitchen', 'depends_on': ['task_000']}
    task = {'task_id': 'task_001', 'task_type': 'manipulation', 'task_family': 'object_transfer',
            'instruction': text, 'capabilities': ['navigation', 'manipulation']}
    truth = {'task_id': 'task_001', 'task_type': 'manipulation', 'target': {'object_ids': [oid]},
             'expected_outcome': {'placement_geometry': {'planned': True}}, 'success_criteria': {}}
    return freeze_template(proposal, task, truth), task, truth


def question(query='current_room', history=False, ordinal=1):
    proposal = {'task_id': 'task_002', 'task_type': 'qa', 'target_object_id': 'book', 'query': query}
    reference = {'object_name': 'book', 'requested_action': 'move', 'source_time': '2026-11-05T09:10:00+08:00',
                 'current_time': '2026-11-06T09:00:00+08:00', 'include_clock': ordinal > 1}
    subject = historical_reference(reference) if history else 'the red book'
    texts = {'current_room': f'Find {subject} and report which area currently contains it.',
             'room_before': f'Which area contained {subject} before that move?',
             'room_after': f'Which area contained {subject} immediately after that move?',
             'has_changed': f'Find {subject} and report whether its area differs from where you left it during that move.'}
    task = {'task_id': 'task_002', 'task_type': 'qa', 'task_family': 'spatial_relation_qa',
            'question': texts[query], 'answer_type': 'boolean' if query == 'has_changed' else 'free_text',
            'capabilities': ['qa'] + (['navigation'] if query in {'current_room', 'has_changed'} else [])}
    truth = {'task_id': 'task_002', 'target': {'object_ids': ['book']}, 'expected_outcome': {},
             'success_criteria': {}, 'query': query}
    if history:
        task['issued_at'] = reference['current_time']
        truth['memory_evidence'] = {'selector': {'kind': 'planned_transfer', 'ordinal': ordinal},
            'public_reference': reference,
            'event': {'session_id': 'session_000', 'task_id': 'task_001', 'object_id': 'book'}}
    return freeze_template(proposal, task, truth), task, truth


def completed_event(success=True):
    return {'event_scope': 'task', 'session_id': 'session_000', 'task_id': 'task_001', 'object_id': 'book', 'kind': 'transfer',
            'issued_at': '2026-11-05T09:10:00+08:00',
            'success': success, 'completed_action': success, 'agent_information': 'actual_task_experience',
            'relation_before': {'room_id': 'bedroom'}, 'relation_after': {'room_id': 'study' if success else 'bedroom'},
            'object_before': entity('book', 'book', 'bedroom'),
            'object_after': entity('book', 'book', 'study' if success else 'bedroom')}


def return_transfer(phase='before'):
    source = question(history=True)[2]['memory_evidence']
    proposal = {'task_id': 'task_003', 'task_type': 'manipulation', 'target_object_id': 'book',
                'memory_selector': source['selector'], 'destination_from_history': phase}
    subject = historical_reference(source['public_reference'])
    destination = ('the area where it was before that move' if phase == 'before'
                   else 'the area where you placed it during that move')
    task = {'task_id': 'task_003', 'task_type': 'manipulation', 'task_family': 'object_transfer',
            'capabilities': ['navigation', 'manipulation', 'memory'],
            'instruction': f'Pick up {subject} and place it on the floor in {destination}.',
            'issued_at': source['public_reference']['current_time']}
    truth = {'target': {'object_ids': ['book']}, 'expected_outcome': {}, 'success_criteria': {},
             'memory_evidence': source}
    return freeze_template(proposal, task, truth), task, truth


class RuntimeRepairTests(unittest.TestCase):
    def setUp(self):
        self.cache, self.state = fixture()

    def instantiate(self, spec, state=None, history=None):
        template, task, truth = spec
        return instantiate_task(template, self.cache, state or self.state, history or [], task=task, ground_truth=truth)

    def test_frozen_wording_and_valid_initial_phrase_are_preserved(self):
        spec = transfer()
        result = self.instantiate(spec)
        self.assertEqual(result['public_task'], spec[1])
        self.assertFalse(result['repair_record']['changed'])
        self.assertEqual(spec[0]['text'], 'Pick up {target_reference} and place it on the floor in {destination_area}.')

    def test_moved_duplicate_uses_cached_attributes_without_stale_location(self):
        self.state['objects']['book']['room_id'] = 'study'
        result = self.instantiate(transfer())
        self.assertEqual(result['status'], 'ready')
        self.assertIn('Pick up the red book and', result['public_task']['instruction'])
        self.assertNotIn('nightstand', result['public_task']['instruction'])
        self.assertEqual(result['runtime_ground_truth']['target']['object_ids'], ['book'])
        self.assertEqual(result['runtime_ground_truth']['preconditions'][0]['room_id'], 'study')
        self.assertNotIn('placement_geometry', result['runtime_ground_truth']['expected_outcome'])

    def test_unchanged_local_description_can_be_scoped_after_peer_moves(self):
        self.state['objects']['other']['room_id'] = 'kitchen'
        result = self.instantiate(transfer())
        self.assertIn('nightstand in the bedroom', result['public_task']['instruction'])

    def test_replay_float_tolerance_preserves_semantic_checks(self):
        from benchmark.contracts.legacy_templates import replay_equal
        self.assertTrue(replay_equal({'length_m': 1.8884776310850244}, {'length_m': 1.888477631085024}))
        self.assertFalse(replay_equal({'length_m': 1.8884776}, {'length_m': 1.889}))
        self.assertFalse(replay_equal({'object_id': 'book'}, {'object_id': 'cup'}))
        self.assertFalse(replay_equal({'success': True}, {'success': 1}))

    def test_unknown_peer_attribute_does_not_prove_uniqueness(self):
        self.cache['objects']['other']['identity_attributes'] = {}
        self.state['objects']['book']['room_id'] = 'study'
        result = self.instantiate(transfer())
        self.assertEqual(result['status'], 'reference_unavailable')
        self.assertIsNone(result['public_task'])

    def test_current_qa_uses_actual_answer_and_hides_location(self):
        self.state['objects']['book']['room_id'] = 'kitchen'
        result = self.instantiate(question())
        self.assertEqual(result['runtime_ground_truth']['expected_outcome']['answer'], 'kitchen')
        self.assertNotIn('kitchen', result['public_task']['question'])

    def test_current_qa_does_not_fall_back_to_room_scoped_identity(self):
        self.cache['objects']['book']['identity_attributes'] = {}
        result = self.instantiate(question())
        self.assertEqual(result['status'], 'reference_unavailable')

    def test_failed_history_never_retargets_to_later_success(self):
        later = completed_event(); later.update(task_id='task_004', object_id='other')
        result = self.instantiate(question('room_after', history=True), history=[completed_event(False), later])
        self.assertEqual(result['status'], 'ready')
        self.assertIn('when that task ended', result['public_task']['question'])
        self.assertEqual(result['runtime_ground_truth']['expected_outcome']['answer'], 'bedroom')
        self.assertEqual(result['runtime_ground_truth']['memory_evidence']['binding']['task_id'], 'task_001')

    def test_missing_or_ambiguous_history_is_not_claimed_as_a_failed_action(self):
        for history, code in (([], 'history_record_missing'),
                              ([completed_event(), completed_event()], 'history_event_ambiguous')):
            result = self.instantiate(question('room_after', history=True), history=history)
            self.assertEqual(result['repair_record']['reason_code'], code)

    def test_undispatched_source_is_not_an_actual_experience(self):
        template, task, _ = transfer()
        event = execution_event('session_000', template, task, self.state, self.state,
            {'success': None, 'dispatched': False, 'termination_reason': 'already_satisfied',
             'reason_code': 'already_satisfied'}, [])
        result = self.instantiate(question(history=True), history=[event])
        self.assertEqual(result['repair_record']['reason_code'], 'source_task_not_dispatched')
        self.assertEqual(result['repair_record']['dependency']['source_status'], 'already_satisfied')
        self.assertFalse(event['completed_action'])

    def test_failed_action_still_has_a_real_request_for_current_state_questions(self):
        result = self.instantiate(question(history=True), history=[completed_event(False)])
        self.assertEqual(result['status'], 'ready')
        self.assertIn('book I asked you to pick up yesterday', result['public_task']['question'])
        self.assertEqual(result['runtime_ground_truth']['expected_outcome']['answer'], 'bedroom')

    def test_duplicate_task_summaries_are_not_silently_merged(self):
        result = self.instantiate(question(history=True), history=[completed_event(), completed_event()])
        self.assertEqual(result['repair_record']['reason_code'], 'history_event_ambiguous')

    def test_failed_earlier_task_does_not_change_time_or_identity(self):
        earlier = completed_event(False); earlier.update(task_id='task_000', object_id='other')
        result = self.instantiate(question(history=True, ordinal=2), history=[earlier, completed_event()])
        self.assertIn('book I asked you to pick up yesterday at 09:10', result['public_task']['question'])
        self.assertEqual(result['runtime_ground_truth']['target']['object_ids'], ['book'])

    def test_hidden_mutation_is_never_public_history(self):
        event = completed_event(); event['agent_information'] = 'hidden'
        self.assertEqual(self.instantiate(question(history=True), history=[event])['status'], 'dependency_unsatisfied')

    def test_historical_answer_uses_actual_event_not_planned_destination(self):
        self.state['objects']['book']['room_id'] = 'kitchen'
        result = self.instantiate(question('room_after', history=True), history=[completed_event()])
        self.assertEqual(result['runtime_ground_truth']['expected_outcome']['answer'], 'study')
        self.assertNotIn('study', result['public_task']['question'])

    def test_has_changed_uses_actual_placement_as_baseline(self):
        self.state['objects']['book']['room_id'] = 'study'
        result = self.instantiate(question('has_changed', history=True), history=[completed_event()])
        self.assertIs(result['runtime_ground_truth']['expected_outcome']['answer'], False)

    def test_held_target_uses_place_branch_and_release_criterion(self):
        self.state['held_object'] = 'book'
        self.state['objects']['book']['held'] = True
        result = self.instantiate(transfer())
        self.assertEqual(result['public_task']['instruction'], 'Place the object you are holding on the floor in the kitchen.')
        self.assertTrue(result['runtime_ground_truth']['success_criteria']['require_released'])
        self.assertEqual(self.instantiate(question())['runtime_ground_truth']['expected_outcome']['answer'], 'bedroom')

    def test_failed_pick_place_retains_partial_actual_state(self):
        template, task, _ = transfer()
        after = deepcopy(self.state); after['objects']['book']['held'] = True
        event = execution_event('session_000', template, task, self.state, after, {'success': False}, [])
        self.assertTrue(event['object_after']['held'])
        self.assertFalse(event['completed_action'])

    def test_wrong_destination_is_still_an_actual_move(self):
        template, task, _ = transfer()
        template['issued_at'] = '2026-11-05T09:10:00+08:00'
        after = deepcopy(self.state); after['objects']['book']['room_id'] = 'study'
        actions = [{'action': {'action': 'place'}, 'result': {'success': True, 'object_id': 'book', 'room_id': 'study'}}]
        event = execution_event('session_000', template, task, self.state, after, {'success': False}, actions)
        self.assertFalse(event['success'])
        self.assertTrue(event['completed_action'])
        result = self.instantiate(question('room_after', history=True), history=[event])
        self.assertEqual(result['runtime_ground_truth']['expected_outcome']['answer'], 'study')

    def test_unintended_placements_count_in_order_without_changing_binding(self):
        template, task, _ = transfer()
        template['issued_at'] = '2026-11-05T09:10:00+08:00'
        actions = [{'action': {'action': 'place'}, 'result': {'success': True, 'object_id': oid, 'room_id': 'study'}}
                   for oid in ('other', 'book')]
        history = execution_events('session_000', template, task, self.state, self.state, {'success': False}, actions)
        self.assertEqual([e['object_id'] for e in history if e['event_scope'] == 'action'], ['other', 'book'])
        self.assertEqual(history[0]['event_scope'], 'task')
        self.assertEqual(history[0]['object_id'], 'book')
        result = self.instantiate(question(history=True), history=history)
        self.assertIn('book I asked you to pick up yesterday', result['public_task']['question'])
        self.assertNotIn('second', result['public_task']['question'])
        self.assertEqual(result['runtime_ground_truth']['target']['object_ids'], ['book'])

    def test_multiple_moves_use_task_end_state_not_last_action_guess(self):
        template, task, _ = transfer()
        template['issued_at'] = '2026-11-05T09:10:00+08:00'
        actions = [{'action': {'action': 'place'}, 'result': {'success': True, 'object_id': 'book', 'room_id': 'study'}}] * 2
        history = execution_events('session_000', template, task, self.state, self.state, {'success': True}, actions)
        result = self.instantiate(question('room_after', history=True), history=history)
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['runtime_ground_truth']['expected_outcome']['answer'], 'bedroom')
        self.assertEqual(history[0]['placement_count'], 2)
        self.assertNotIn('object_after', history[1])  # Never fabricate per-action snapshots.

    def test_absent_target_and_determinism(self):
        spec = transfer()
        baseline = deepcopy((spec, self.cache, self.state))
        self.assertEqual(self.instantiate(spec), self.instantiate(spec))
        self.assertEqual((spec, self.cache, self.state), baseline)
        del self.state['objects']['book']
        self.assertEqual(self.instantiate(spec)['status'], 'target_unavailable')

    def test_new_duplicate_invalidates_unique_category(self):
        self.state['objects']['cup2'] = entity('cup2', 'cup', 'study')
        with self.assertRaisesRegex(ValueError, 'uniquely'):
            object_reference(self.cache, self.state, 'cup')

    def test_named_place_requires_establishment_and_freezes_geometry(self):
        p = {'task_id': 'task_000', 'task_type': 'navigation', 'target_object_id': 'cup',
             'named_place': {'name': 'Maple Point', 'mode': 'revisit',
                             'paired_task': {'session_id': 'session_000', 'task_id': 'task_000'}}}
        t = {'task_id': 'task_000', 'task_type': 'navigation', 'instruction': 'Return to Maple Point.'}
        gt = {'target': {'object_ids': ['cup']}}
        spec = freeze_template(p, t, gt), t, gt
        self.assertEqual(self.instantiate(spec)['status'], 'dependency_unsatisfied')
        original = deepcopy(self.state['objects']['cup']['geometry'])
        event = {'session_id': 'session_000', 'task_id': 'task_000', 'named_place': 'Maple Point',
                 'agent_information': 'actual_task_experience',
                 'success': True, 'completed_action': True, 'target_geometry': original}
        self.state['objects']['cup']['room_id'] = 'kitchen'
        result = self.instantiate(spec, history=[event])
        self.assertEqual(result['runtime_ground_truth']['fixed_target_geometry'], original)

    def test_failed_task_snapshots_repair_all_temporal_intents(self):
        event = completed_event(False)
        self.state['objects']['book']['room_id'] = 'kitchen'  # A later change is not yesterday's outcome.
        for query, answer, clause in [('room_before', 'bedroom', 'when that task started'),
                                      ('room_after', 'bedroom', 'when that task ended'),
                                      ('has_changed', True, 'where it was when that task ended')]:
            with self.subTest(query=query):
                result = self.instantiate(question(query, history=True), history=[event])
                self.assertEqual(result['status'], 'ready')
                self.assertEqual(result['runtime_ground_truth']['expected_outcome']['answer'], answer)
                self.assertIn(clause, result['public_task']['question'])
                self.assertNotIn('bedroom', result['public_task']['question'])
                self.assertNotIn('that move', result['public_task']['question'])
                self.assertEqual(result['repair_record']['temporal_basis'], 'actual_task_boundaries')
        for phase in ('before', 'after'):
            with self.subTest(destination=phase):
                result = self.instantiate(return_transfer(phase), history=[event])
                self.assertEqual(result['status'], 'ready')
                self.assertEqual(result['runtime_ground_truth']['success_criteria']['room_id'], 'bedroom')
                self.assertIn('when that task ' + ('started' if phase == 'before' else 'ended'),
                              result['public_task']['instruction'])
                self.assertNotIn('bedroom', result['public_task']['instruction'])

    def test_held_at_task_end_still_has_a_real_area_without_inventing_placement(self):
        template, task, _ = transfer()
        template['issued_at'] = '2026-11-05T09:10:00+08:00'
        after = deepcopy(self.state)
        after['objects']['book']['room_id'] = 'study'
        after['held_object'] = 'book'
        event = execution_event('session_000', template, task, self.state, after, {'success': False}, [])
        result = self.instantiate(question('room_after', history=True), history=[event])
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['runtime_ground_truth']['expected_outcome']['answer'], 'study')
        self.assertIn('when that task ended', result['public_task']['question'])
        self.assertFalse(event['completed_action'])
        self.assertTrue(event['object_after']['held'])
        returned = self.instantiate(return_transfer('after'), state=after, history=[event])
        self.assertIn('when that task ended', returned['public_task']['instruction'])
        self.assertTrue(returned['public_task']['instruction'].startswith('Place '))

    def test_missing_snapshot_never_falls_back_to_planned_or_current_state(self):
        for edit, code in [('missing', 'history_snapshot_missing'), ('conflict', 'history_snapshot_mismatch'),
                           ('unknown_room', 'historical_area_unknown')]:
            event = completed_event(False)
            if edit == 'missing': event.pop('object_after')
            if edit == 'conflict': event['object_after']['room_id'] = 'study'
            if edit == 'unknown_room':
                event['object_after']['room_id'] = event['relation_after']['room_id'] = 'unknown'
            result = self.instantiate(question('room_after', history=True), history=[event])
            self.assertEqual(result['repair_record']['reason_code'], code)
            self.assertIsNone(result['public_task'])
            self.assertFalse(result['repair_record']['scored'])
            self.assertEqual(result['repair_record']['dependency']['task_id'], 'task_001')
        event = completed_event(False); event.pop('object_after')
        self.assertEqual(self.instantiate(question('room_before', history=True), history=[event])['status'], 'ready')
        self.assertEqual(self.instantiate(question(history=True), history=[event])['status'], 'ready')

    def test_past_question_can_be_answered_even_if_object_is_absent_now(self):
        del self.state['objects']['book']
        result = self.instantiate(question('room_after', history=True), history=[completed_event(False)])
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['runtime_ground_truth']['expected_outcome']['answer'], 'bedroom')
        self.assertEqual(self.instantiate(question(history=True), history=[completed_event(False)])['status'], 'target_unavailable')

    def test_action_rows_do_not_supply_a_task_boundary(self):
        event = completed_event(); event.update(event_scope='action', action_index=0)
        result = self.instantiate(question('room_after', history=True), history=[event])
        self.assertEqual(result['repair_record']['reason_code'], 'history_record_missing')
        # Legacy action rows cannot masquerade as full task snapshots either.
        event.pop('event_scope')
        result = self.instantiate(question('room_after', history=True), history=[event])
        self.assertEqual(result['repair_record']['reason_code'], 'history_snapshot_missing')

    def test_successful_place_followed_by_pick_uses_final_held_state(self):
        template, task, _ = transfer(); template['issued_at'] = '2026-11-05T09:10:00+08:00'
        after = deepcopy(self.state)
        after['objects']['book'].update(room_id='kitchen', held=True)
        actions = [
            {'action': {'action': 'place'}, 'result': {'success': True, 'object_id': 'book', 'room_id': 'study'},
             'object_before': {'room_id': 'study', 'held': True}, 'object_after': {'room_id': 'study', 'held': False}},
            {'action': {'action': 'pick'}, 'result': {'success': True, 'object_id': 'book'},
             'object_before': {'room_id': 'study', 'held': False}, 'object_after': {'room_id': 'study', 'held': True}}]
        history = execution_events('session_000', template, task, self.state, after, {'success': False}, actions)
        result = self.instantiate(question('room_after', history=True), history=history)
        self.assertEqual(result['runtime_ground_truth']['expected_outcome']['answer'], 'kitchen')
        self.assertEqual(history[1]['object_after'], {'room_id': 'study', 'held': False})
        self.assertTrue(history[2]['object_after']['held'])
        self.assertTrue(history[0]['object_after']['held'])

    def test_failed_named_place_is_not_replaced_with_a_scene_target(self):
        p = {'task_id': 'task_000', 'task_type': 'navigation', 'target_object_id': 'cup',
             'named_place': {'name': 'Maple Point', 'mode': 'revisit',
                             'paired_task': {'session_id': 'session_000', 'task_id': 'task_000'}}}
        task = {'task_id': 'task_000', 'task_type': 'navigation', 'instruction': 'Return to Maple Point.'}
        truth = {'target': {'object_ids': ['cup']}}
        event = {'session_id': 'session_000', 'task_id': 'task_000', 'event_scope': 'task',
                 'agent_information': 'actual_task_experience', 'named_place': 'Maple Point',
                 'success': False, 'completed_action': False, 'dispatched': True}
        result = self.instantiate((freeze_template(p, task, truth), task, truth), history=[event])
        self.assertEqual(result['repair_record']['reason_code'], 'named_place_not_established')
        self.assertIn('verify_named_place_experience', result['repair_record']['repair_steps'])


if __name__ == '__main__':
    unittest.main()
