"""Instantiate frozen tasks against actual world state; no model calls."""


from copy import deepcopy
from itertools import combinations
import re
from types import SimpleNamespace

from roboodyssey.contracts.language import validate_public_wording
from roboodyssey.contracts.references import _metadata_reference_holds
from roboodyssey.contracts.state import ATTRIBUTE_KEYS, digest, geometry, same_location
from roboodyssey.contracts.legacy_templates import validate_template
from roboodyssey.contracts.language import historical_reference


class RepairUnavailable(ValueError):
    def __init__(self, status, reason, *, code=None, dependency=None):
        super().__init__(reason)
        self.status = status
        self.code = code or status
        self.dependency = dependency


def _dependency_unavailable(binding, matches, *, named=False):
    """Explain missing experience without assuming that a missing log is failure."""
    dependency = {key: binding[key] for key in ('session_id', 'task_id', 'object_id') if key in binding}
    if not matches:
        code, reason = 'history_record_missing', 'no actual execution record for the bound source task'
    elif len(matches) > 1:
        code, reason = 'history_event_ambiguous', 'multiple actual events match the bound source task'
    else:
        event = matches[0]
        dependency.update(source_success=event.get('success'), source_status=event.get('task_status'),
                          source_reason=event.get('reason_code'))
        if event.get('dispatched') is False:
            code, reason = 'source_task_not_dispatched', 'source task was not dispatched; no requested experience occurred'
        elif named:
            code, reason = 'named_place_not_established', 'source task did not establish the named place'
        else:
            code, reason = 'source_experience_missing', 'source record does not establish the required experience'
    raise RepairUnavailable('dependency_unsatisfied', reason, code=code, dependency=dependency)


def area_name(cache, state, room_id):
    area = cache['areas'].get(room_id)
    if not area:
        raise RepairUnavailable('state_unavailable', 'target has no known area', code='area_unknown')
    for oid in area['anchor_object_ids']:
        if oid not in state['objects'] or not same_location(state['objects'][oid], cache['objects'][oid]['initial']):
            raise RepairUnavailable('reference_unavailable', 'area landmark changed: ' + oid, code='area_landmark_changed')
    return area['text']


def _with_article(text):
    return text if text.lower().startswith(('the ', 'a ', 'an ')) else 'the ' + text


def _appearance(cache, state, oid, peers):
    traits = cache['objects'][oid]['identity_attributes']
    if state['objects'][oid].get('identity_attributes') is not None:
        traits = state['objects'][oid]['identity_attributes']
    keys = [key for key in ATTRIBUTE_KEYS if traits.get(key)]
    for count in range(1, len(keys) + 1):
        for subset in combinations(keys, count):
            # An unknown trait does not prove that another object is different.
            def differs(other):
                other_traits = state['objects'][other].get('identity_attributes',
                    cache['objects'].get(other, {}).get('identity_attributes', {}))
                return any(other_traits.get(k) and other_traits[k] != traits[k] for k in subset)
            if all(differs(other) for other in peers if other != oid):
                return 'the ' + ' '.join(traits[k] for k in subset) + ' ' + cache['objects'][oid]['category_name']
    return None


def object_reference(cache, state, oid, *, hides_room=False, preferred=None):
    entities = state['objects']
    if oid not in entities:
        raise RepairUnavailable('target_unavailable', 'target object is absent', code='target_absent')
    if oid not in cache['objects']:
        raise RepairUnavailable('target_unavailable', 'target object is missing from scene cache', code='target_cache_missing')
    item, obj = cache['objects'][oid], entities[oid]
    peers = [other for other, entity in entities.items() if entity.get('category') == item['category']]
    if oid not in peers:
        raise RepairUnavailable('state_unavailable', 'target category changed', code='target_category_changed')
    initial_peers = {other for other, value in cache['objects'].items() if value['category'] == item['category']}
    normalized = {other: {**entity, 'geometry': geometry(entity)} for other, entity in entities.items()}
    context = SimpleNamespace(entities=normalized)
    for expression in sorted(item['expressions'], key=lambda e: e['text'] != preferred):
        if any(entities[other].get('identity_attributes') is not None and
               entities[other]['identity_attributes'] != cache['objects'].get(other, {}).get('identity_attributes', {})
               for other in peers):
            continue
        if hides_room and (expression['spatial'] or re.search(
                r'\b(bedroom|bathroom|kitchen|study|balcony|corridor|living room|passage)\b', expression['text'], re.I)
                or any(area['text'].lower() in expression['text'].lower() for area in cache['areas'].values())):
            continue
        binding = expression.get('binding')
        if binding:
            valid = _metadata_reference_holds(oid, binding, context)
        else:
            # Unstructured legacy claims are valid only in the same comparison
            # set and context. Never carry an initial spatial phrase across moves.
            dependencies = set(peers)
            if expression['spatial']:
                dependencies.update(expression['dependency_object_ids'])
            valid = set(peers) == initial_peers and all(
                other in entities and other in cache['objects'] and
                same_location(entities[other], cache['objects'][other]['initial'])
                for other in dependencies)
            valid = valid and not any(other != oid and any(e['text'].casefold() == expression['text'].casefold()
                for e in value['expressions']) for other, value in cache['objects'].items())
        if valid:
            return _with_article(expression['text'])
        if not hides_room and not binding and expression['spatial']:
            room = obj.get('room_id')
            local = {other for other in peers if entities[other].get('room_id') == room}
            original_local = {other for other in initial_peers if cache['objects'][other]['initial']['room_id'] == room}
            dependencies = local | {oid} | set(expression['dependency_object_ids'])
            if local <= original_local and all(other in entities and other in cache['objects'] and
                    same_location(entities[other], cache['objects'][other]['initial']) for other in dependencies):
                # The original unique visual description remains valid within
                # an unchanged subset. Add its scope after peers move elsewhere.
                name = area_name(cache, state, room)
                phrase = _with_article(expression['text'])
                return phrase if name.casefold() in phrase.casefold() else phrase + ' in the ' + name
    if len(peers) == 1:
        return 'the ' + item['category_name']
    phrase = _appearance(cache, state, oid, peers)
    if phrase:
        return phrase
    if hides_room:
        raise RepairUnavailable('reference_unavailable', 'no unique reference that hides the answer',
                                code='answer_safe_reference_unavailable')
    room = obj.get('room_id')
    local = [other for other in peers if entities[other].get('room_id') == room]
    name = area_name(cache, state, room)
    if len(local) == 1:
        return 'the ' + item['category_name'] + ' in the ' + name
    phrase = _appearance(cache, state, oid, local)
    if phrase:
        return phrase + ' in the ' + name
    # Re-evaluate only authored geometric qualifiers; do not invent a ranking
    # that was never validated during construction.
    for expression in item['expressions']:
        binding = deepcopy(expression.get('binding'))
        if not binding or binding['kind'] not in {'height_extreme', 'nearest_landmark'}:
            continue
        binding['room_id'] = room
        if not _metadata_reference_holds(oid, binding, context):
            continue
        if binding['kind'] == 'height_extreme':
            adjective = 'tallest' if binding['sign'] == 1 else 'shortest'
            return f'the {adjective} {item["category_name"]} in the {name}'
        anchor = cache['objects'][binding['anchor_object_id']]['category_name']
        return f'the {item["category_name"]} closest to the {anchor} in the {name}'
    raise RepairUnavailable('reference_unavailable', 'target cannot be uniquely described with valid cached evidence',
                            code='object_reference_ambiguous')


def _history_event(template, history):
    binding = template.get('history_binding')
    if not binding:
        return None, None
    matches = [event for event in history if event.get('session_id') == binding['session_id']
               and event.get('task_id') == binding['task_id'] and event.get('object_id') == binding['object_id']
               and event.get('agent_information') == 'actual_task_experience']
    # New action rows are evidence within a task, never alternative bindings.
    matches = [e for e in matches if e.get('event_scope') != 'action']
    snapshots = [e for e in matches if e.get('event_scope') == 'task']
    if snapshots:
        matches = snapshots
    if not matches:
        _dependency_unavailable(binding, matches)
    if len(matches) != 1:
        _dependency_unavailable(binding, matches)
    if any(e.get('dispatched') is False for e in matches):
        _dependency_unavailable(binding, [next(e for e in matches if e.get('dispatched') is False)])
    event = matches[0]
    kind = template['history_selector']['kind'].removeprefix('planned_')
    if any(e.get('kind') != kind for e in matches):
        raise RepairUnavailable('dependency_unsatisfied', 'bound historical action kind differs',
                                code='history_kind_mismatch', dependency=binding)
    reference = template['contract'].get('historical_reference', {})
    if not reference.get('source_time') or not template.get('issued_at'):
        raise RepairUnavailable('time_context_unavailable', 'archive requires explicit task times; migrate temporal references first',
                                code='archive_time_missing')
    if any(e.get('issued_at') != reference['source_time'] for e in matches):
        raise RepairUnavailable('dependency_unsatisfied', 'bound historical request time differs',
                                code='history_time_mismatch', dependency=binding)
    reference = {**reference, 'current_time': template['issued_at']}
    return event, historical_reference(reference)


def _historical_room(event, phase, cache, binding):
    """Resolve only the requested task boundary, never a later/current state."""
    snapshot = event.get('object_' + phase)
    if ('action_index' in event or not isinstance(snapshot, dict)
            or not snapshot.get('room_id')):
        raise RepairUnavailable('state_unavailable', 'source task has no usable ' + phase + ' snapshot',
                                code='history_snapshot_missing', dependency=binding)
    room = snapshot['room_id']
    relation = event.get('relation_' + phase, {})
    if relation.get('room_id') is not None and relation['room_id'] != room:
        raise RepairUnavailable('state_unavailable', 'source task snapshot and relation disagree',
                                code='history_snapshot_mismatch', dependency=binding)
    if room not in cache['areas']:
        raise RepairUnavailable('state_unavailable', 'source task snapshot has no known area',
                                code='historical_area_unknown', dependency=binding)
    return room


def _task_boundary_text(text, query):
    """Replace the authored move presupposition with a real task boundary.

    Preserve the frozen connective/target wording; these are bounded grammar
    substitutions, not free generation or a disclosure of the historical area.
    """
    replacements = {
        'room_before': ('before that move', 'when that task started'),
        'room_after': ('immediately after that move', 'when that task ended'),
        'has_changed': ('where you left it during that move', 'where it was when that task ended'),
    }
    if query not in replacements:
        return text
    old, new = replacements[query]
    text, count = re.subn(re.escape(old), new, text, flags=re.I)
    if count != 1:
        raise RepairUnavailable('template_unavailable', 'cannot safely replace the historical time clause',
                                code='history_wording_unsupported')
    return text


def instantiate_task(template, scene_cache, current_state, execution_history, *, task, ground_truth,
                     completion_check=None):
    """Resolve dependencies/GT, optionally check completion, then render wording.

    completion_check receives the resolved private GT and returns evidence only
    if the current state already satisfies the task. Only public_task is public.
    """
    validate_template(template)
    public, truth = deepcopy(task), deepcopy(ground_truth)
    record = {'schema_version': 1, 'task_id': task['task_id'], 'template_version': template['schema_version'],
              'cache_key': scene_cache['cache_key'], 'state_hash': digest(current_state),
              'history_hash': digest(execution_history), 'original_text': task[template['field']],
              'slots': {}, 'reasons': [], 'repair_steps': [], 'semantic_degradation': False}
    try:
        oid = template['target_object_id']
        contract = template['contract']
        query = contract.get('query')
        named = template.get('named_place')
        recalled = query in {'room_before', 'room_after'}
        if oid not in current_state['objects'] and not recalled and not (named and named['mode'] == 'revisit'):
            raise RepairUnavailable('target_unavailable', 'target object is absent', code='target_absent')
        obj = current_state['objects'].get(oid, {})
        held = current_state.get('held_object') == oid or bool(obj.get('held'))
        if template.get('issued_at') and task.get('issued_at') != template['issued_at']:
            raise RepairUnavailable('time_context_unavailable', 'task time differs from frozen template', code='task_time_mismatch')
        record['repair_steps'].append('resolve_original_request')
        event, historical_reference = _history_event(template, execution_history)
        slots = deepcopy(template['preview_slots'])
        if named and named['mode'] == 'revisit':
            record['repair_steps'].append('verify_named_place_experience')
            pair = named['paired_task']
            established = [e for e in execution_history if e.get('session_id') == pair['session_id']
                           and e.get('task_id') == pair['task_id']
                           and e.get('event_scope') != 'action'
                           and e.get('agent_information') == 'actual_task_experience']
            if (len(established) != 1 or not established[0].get('success')
                    or not established[0].get('completed_action') or established[0].get('named_place') != named['name']):
                _dependency_unavailable(pair, established, named=True)
            truth['fixed_target_geometry'] = established[0]['target_geometry']
        elif named:
            truth['fixed_target_geometry'] = geometry(obj)
        if task['task_type'] == 'manipulation':
            destination = contract.get('destination_room_id')
            if contract['destination_mode'] == 'history':
                record['repair_steps'].append('resolve_historical_task_destination')
                phase = contract['destination_phase']
                destination = _historical_room(event, phase, scene_cache, template['history_binding'])
                slots['destination_area'] = 'the area it was in when that task ' + ('started' if phase == 'before' else 'ended')
                record.update(temporal_basis='actual_task_boundaries')
                record['reasons'].append('historical_destination_uses_task_snapshot')
            truth['preconditions'] = [{'type': 'held'} if held else {'type': 'in_room', 'room_id': obj.get('room_id')}]
            outcome = truth['expected_outcome']
            outcome['state_changes'] = [{'object_id': oid,
                'relation_before': {'type': 'in_room', 'room_id': obj.get('room_id')},
                'relation_after': {'type': 'in_room', 'room_id': destination}}]
            outcome.pop('placement_geometry', None)
            truth.pop('placement_validation', None)
            truth['success_criteria'].update(room_id=destination, require_released=True,
                                             require_place_event=True, placement_surface='floor')
            truth['reference_actions'] = ([] if held else [{'action': 'pick', 'object_id': oid}]) + [
                {'action': 'place', 'object_id': oid, 'room_id': destination}]
            truth['target']['room_ids'] = [obj.get('room_id'), destination]
        elif task['task_type'] == 'qa':
            record['repair_steps'].append('resolve_answer_from_actual_state')
            if query in {'current_room', 'has_changed'} and obj.get('room_id') not in scene_cache['areas']:
                raise RepairUnavailable('state_unavailable', 'object has no known current area', code='area_unknown')
            if query in {'room_before', 'room_after'}:
                answer = _historical_room(event, query.removeprefix('room_'), scene_cache, template['history_binding'])
            elif query == 'has_changed':
                answer = obj.get('room_id') != _historical_room(event, 'after', scene_cache, template['history_binding'])
            else:
                answer = obj.get('room_id')
            if answer is None:
                raise RepairUnavailable('state_unavailable', 'question has no known answer', code='historical_area_unknown')
            accepted = [str(answer).lower()] if isinstance(answer, bool) else [area_name(scene_cache, current_state, answer)]
            truth['expected_outcome'].update(answer=answer, accepted_answers=accepted)
            truth['target']['room_ids'] = [obj.get('room_id')]
            if query in {'room_before', 'room_after', 'has_changed'}:
                record.update(temporal_basis='actual_task_boundaries')
                record['reasons'].append('historical_question_uses_task_snapshot')
        else:
            truth['target']['room_ids'] = [obj.get('room_id')]
        truth.pop('reference_trajectory', None)
        truth['state_assumption'] = 'actual_task_start_state'
        if event:
            truth['memory_evidence'] = {'event': deepcopy(event), 'binding': template['history_binding']}
            if record.get('temporal_basis'):
                truth['memory_evidence']['temporal_basis'] = record['temporal_basis']
        # Naming creates a memory even if the robot is already at the landmark.
        if completion_check and not (named and named['mode'] == 'establish'):
            record['repair_steps'].append('check_goal_already_satisfied')
            evidence = completion_check(truth)
            if evidence:
                record.update(status='already_satisfied', reason_code='already_satisfied',
                              reasons=[evidence['reason']], completion_evidence=evidence,
                              scored=False, dispatched=False, changed=False, rendered_text=None)
                return {'status': 'already_satisfied', 'public_task': None,
                        'runtime_ground_truth': truth, 'repair_record': record}
        if 'target_reference' in slots:
            record['repair_steps'].append('repair_object_reference')
            if historical_reference:
                slots['target_reference'] = historical_reference
            elif held and task['task_type'] == 'manipulation':
                slots['target_reference'] = 'the object you are holding'
            else:
                slots['target_reference'] = object_reference(scene_cache, current_state, oid,
                    hides_room=query in {'current_room', 'has_changed'}, preferred=slots['target_reference'])
        if task['task_type'] == 'manipulation' and contract['destination_mode'] != 'history':
            slots['destination_area'] = 'the ' + area_name(scene_cache, current_state, destination)
        text = template['held_text'] if held and task['task_type'] == 'manipulation' else template['text']
        record['repair_steps'].append('render_validated_wording')
        text = _task_boundary_text(text, query)
        public[template['field']] = text.format_map(slots)
        validate_public_wording(public[template['field']])
        record.update(slots=slots, rendered_text=public[template['field']], status='ready')
        for key, value in slots.items():
            if value != template['preview_slots'].get(key):
                record['reasons'].append(key + '_updated')
        if held and task['task_type'] == 'manipulation':
            record['reasons'].append('already_holding_target')
        record['changed'] = public[template['field']] != record['original_text']
        return {'status': 'ready', 'public_task': public, 'runtime_ground_truth': truth, 'repair_record': record}
    except RepairUnavailable as exc:
        record.update(status=exc.status, reason_code=exc.code, dependency=exc.dependency,
                      reasons=[str(exc)], scored=False, dispatched=False, changed=False, rendered_text=None)
        return {'status': exc.status, 'public_task': None, 'runtime_ground_truth': None, 'repair_record': record}
