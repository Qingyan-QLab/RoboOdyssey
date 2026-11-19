"""Actual task boundaries and action evidence owned by the benchmark runner."""
from ..agents.protocol import action_name
from copy import deepcopy

from roboodyssey.contracts.state import geometry


def _object_snapshot(state, oid):
    obj = deepcopy(state.get('objects', {}).get(oid))
    if obj is not None:
        obj['held'] = bool(obj.get('held') or state.get('held_object') == oid)
    return obj


def execution_event(session_id, template, task, before, after, result, actions):
    """One authoritative record for the original target and whole task.

    Completion of a place action is evidence, not task success. A failed task
    still has real start/end states; an undispatched task creates no request.
    """
    oid = template['target_object_id']
    kind = {'navigation': 'visit', 'manipulation': 'transfer'}.get(task['task_type'], 'query')
    placements = [a for a in actions if action_name(a) == 'place'
                  and a.get('result', {}).get('success') and a['result'].get('object_id') == oid]
    dispatched = result.get('dispatched', True)
    completed = bool(placements) if kind == 'transfer' else bool(result.get('success'))
    objects = {phase: _object_snapshot(state, oid) for phase, state in (('before', before), ('after', after))}
    event = {'schema_version': 2, 'event_scope': 'task',
             'session_id': session_id, 'task_id': task['task_id'], 'object_id': oid, 'kind': kind,
             'success': result.get('success'), 'completed_action': dispatched and completed,
             'placement_count': len(placements), 'dispatched': dispatched,
             'task_status': result.get('termination_reason') or ('succeeded' if result.get('success') else 'failed'),
             'reason_code': result.get('reason_code'), 'agent_information': 'actual_task_experience',
             'object_before': objects['before'], 'object_after': objects['after'],
             'relation_before': {'type': 'in_room', 'room_id': (objects['before'] or {}).get('room_id')},
             'relation_after': {'type': 'in_room', 'room_id': (objects['after'] or {}).get('room_id')},
             'actions': deepcopy(actions)}
    if template.get('issued_at'):
        event['issued_at'] = template['issued_at']
    named = template.get('named_place')
    if named and named['mode'] == 'establish':
        event.update(named_place=named['name'], target_geometry=geometry(objects['before'] or {}),
                     named_place_established=bool(dispatched and result.get('success')))
    return event


def execution_events(session_id, template, task, before, after, result, actions):
    """Keep the task snapshot separate from individual (possibly wrong) actions.

    Per-action snapshots come only from the executor. Never reconstruct them
    using the final task state, which may follow several picks and placements.
    """
    primary = execution_event(session_id, template, task, before, after, result, actions)
    events = [primary]
    for index, action in enumerate(actions):
        if action_name(action) not in {'pick', 'place', 'open', 'close'}:
            continue
        info = action.get('result', {})
        event = {'schema_version': 2, 'event_scope': 'action', 'session_id': session_id,
                 'task_id': task['task_id'], 'object_id': info.get('object_id'),
                 'action_index': index, 'kind': action_name(action), 'success': info.get('success'),
                 'agent_information': 'actual_task_experience', 'action': deepcopy(action)}
        for key in ('object_before', 'object_after'):
            if key in action:
                event[key] = deepcopy(action[key])
        events.append(event)
    return events
