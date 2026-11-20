"""Frozen-data runtime helpers. Adapted from MIT-licensed sources; see LICENSE."""
from __future__ import annotations

from copy import deepcopy

import math

from .language import render_request, render

from .language import shorten_areas

from .references import RepairUnavailable, object_reference, area_name

from .state import geometry, location_signature

class RuntimeContractError(ValueError):
    """Invalid or insufficient authoritative evidence; invalidate the affected run."""

def normalize_state(cache, state):
    if not isinstance(state, dict): raise RuntimeContractError('authoritative state must be an object')
    state = deepcopy(state)
    if not isinstance(state.get('objects'), dict): raise RuntimeContractError('authoritative objects missing')
    for oid, cached in cache['objects'].items():
        obj = state['objects'].get(oid)
        if not isinstance(obj, dict): raise RuntimeContractError('authoritative object missing: ' + oid)
        if obj.get('object_id', oid) != oid or obj.get('category', cached['category']) != cached['category']:
            raise RuntimeContractError('object identity/category differs from scene cache')
        obj.setdefault('object_id', oid)
        obj.setdefault('category', cached['category'])
        if 'held' not in obj:
            if 'held_object' not in state: raise RuntimeContractError('authoritative held state missing')
            obj['held'] = state['held_object'] == oid
        unassigned_fixture = obj.get('room_id') is None and cached['category'] in {'door', 'window'}
        if type(obj['held']) is not bool or (obj.get('room_id') not in cache['areas'] and not unassigned_fixture):
            raise RuntimeContractError('invalid authoritative held/room state: ' + oid)
        if 'held_object' in state and obj['held'] != (state['held_object'] == oid):
            raise RuntimeContractError('authoritative held flag conflicts with held_object')
        try: obj['geometry'] = geometry(obj)
        except (TypeError, KeyError, ValueError) as exc: raise RuntimeContractError('invalid authoritative geometry: ' + oid) from exc
        box = obj['geometry'].get('world_aabb', {})
        if not isinstance(box, dict) or set(box) != {'min', 'max'} or any(not isinstance(box[k], (list,tuple)) or len(box[k]) != 3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in box[k]) for k in box):
            raise RuntimeContractError('invalid authoritative geometry: ' + oid)
        if any(a >= b for a,b in zip(box['min'],box['max'])):
            raise RuntimeContractError('empty authoritative geometry')
    if 'robot_pose' not in state and state.get('agent_pose'):
        state['robot_pose'] = deepcopy(state['agent_pose'])
    if 'held_object' in state and state['held_object'] is not None and state['held_object'] not in state['objects']:
        raise RuntimeContractError('unknown authoritative held_object')
    if 'robot_pose' in state:
        pose = state['robot_pose'].get('position') if isinstance(state['robot_pose'], dict) else None
        if not isinstance(pose, (list,tuple)) or len(pose) != 3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in pose):
            raise RuntimeContractError('invalid authoritative robot pose')
    return state

def distance_to_box(pose, box):
    return sum(max(box['min'][i]-pose[i], 0, pose[i]-box['max'][i])**2 for i in (0,1))**.5

def completed(goals, state):
    if len(goals) != 1: return None  # Ordered visits require post-dispatch order evidence.
    goal = goals[0]
    kind, oid = goal['goal_kind'], goal['target']['object_ids'][0]
    if kind == 'transfer':
        obj = state['objects'][oid]
        room = goal['success_criteria']['room_id']
        floor = state.get('floor_heights', {}).get(room, 0)
        if obj['room_id'] == room and not obj['held'] and abs(geometry(obj)['world_aabb']['min'][2]-floor) <= .04:
            return {'goal_id':goal['goal_id'], 'released':True, 'on_floor':True, 'room_id':room}
    if kind in {'visit', 'revisit_place'}:
        pose = state.get('robot_pose', {}).get('position')
        if pose is None: raise RuntimeContractError('robot pose required for navigation completion')
        box = (goal['fixed_target_geometry'] if kind == 'revisit_place' else geometry(state['objects'][oid]))['world_aabb']
        distance = distance_to_box(pose, box)
        if distance <= goal['success_criteria'].get('position_tolerance_m', .8):
            return {'goal_id':goal['goal_id'], 'distance_m':distance}
    return None

def bound_record(binding, oid, history):
    matches = [e for e in history if e.get('object_id')==oid and all(e.get(k)==binding[k] for k in ('session_id','task_id','goal_id'))]
    if len(matches) != 1: raise RuntimeContractError('missing or ambiguous original request record')
    return matches[0]

def snapshot_room(event, phase, oid, cache):
    room = event.get(phase, {}).get('objects', {}).get(oid, {}).get('room_id')
    if room not in cache['areas']: raise RuntimeContractError('necessary actual snapshot/area missing: ' + phase)
    return room

def introduction_reference(cache, state, oid, *, hides_room, original):
    """Validate the frozen first introduction using the published area names.

    Compilation shortens scene descriptions after reference validation. Runtime
    must compare the same vocabulary and prefer that selected description, not
    the object's short name or another cached expression. Known object names
    never enter this path, regardless of the outcome of their first request.
    """
    def public_description(text):
        return shorten_areas(text, cache['names'], cache.get('original_area_descriptions', {})).removeprefix('the ')

    references = {**cache, 'objects': {
        key: {**item, 'expressions': [
            {**expression, 'text': public_description(expression['text'])}
            for expression in item['expressions']]}
        for key, item in cache['objects'].items()}}
    preferred = original['slots']['description'] if original else None
    preferred_text = public_description(preferred) if preferred else None
    scoped_base = None
    room = state['objects'][oid].get('room_id')
    room_name = cache['names'].get('areas', {}).get(room, {}).get('en')
    if preferred_text and not hides_room and room_name:
        suffix = ' in the ' + room_name
        if preferred_text.casefold().endswith(suffix.casefold()):
            scoped_base = preferred_text[:-len(suffix)]
    # Compilation may have added a room scope after another object moved.
    # Retain that still-true scope even if actual history makes it redundant.
    expressions = references['objects'][oid]['expressions']
    candidates = [preferred_text, scoped_base] if scoped_base else [preferred_text]
    selected = next((e['text'] for candidate in candidates if candidate
                     for e in expressions if e['text'].casefold() == candidate.casefold()), preferred_text)
    description = object_reference(references, state, oid, hides_room=hides_room,
                                   preferred=selected)
    # Preserve reviewed spelling/capitalization when the validated relation is
    # unchanged. A different description must come from reference revalidation.
    if preferred and description.casefold() == preferred.casefold():
        return preferred
    if scoped_base and public_description(description).casefold() == scoped_base.casefold():
        area_name(cache, state, room)  # Validate the retained room scope too.
        return preferred
    return description

def failed_causes(history, object_ids, binding=None):
    causes = []
    for event in history:
        bound = binding and all(event.get(k)==binding[k] for k in ('session_id','task_id','goal_id'))
        if event.get('dispatched') is True and event.get('success') is False:
            changes = [oid for oid in object_ids if location_signature(event.get('before',{}).get('objects',{}).get(oid,{}))
                       != location_signature(event.get('after',{}).get('objects',{}).get(oid,{}))]
            if bound or changes:
                causes.append({'session_id':event['session_id'], 'task_id':event['task_id'],
                               'goal_id':event['goal_id'], 'affected_objects':changes, 'source_failed':bool(bound)})
        elif bound and event.get('reason_code')=='unavailable' and event.get('failure_causes'):
            causes.extend(deepcopy(event['failure_causes']))
    return causes

def prepare_mutation_v3(initial, mutation, cache, state, history):
    from .requests import prepare_mutation
    state = normalize_state(cache, state)
    changes = mutation.get('changes')
    if not isinstance(changes, list) or len(changes) != 1:
        raise RuntimeContractError('one object transition required per mutation')
    change = changes[0]
    oid = change.get('object_id')
    if oid not in state['objects'] or change.get('relation_after', {}).get('room_id') not in cache['areas']:
        raise RuntimeContractError('invalid mutation object/destination')
    if not state.get('robot_pose', {}).get('position'):
        raise RuntimeContractError('actual robot pose required for mutation validation')
    if not mutation.get('inspection_binding'):
        raise RuntimeContractError('mutation inspection binding missing')
    event = bound_record(mutation['inspection_binding'], oid, history)
    if event.get('dispatched') is True:
        for phase in ('before', 'after'): snapshot_room(event, phase, oid, cache)
        if event.get('location_experience') is not True:
            return {'status':'skipped','object_id':oid,'reason':'bound request did not establish actual location experience'}
    elif event.get('reason_code') not in {'already_satisfied', 'unavailable'}:
        raise RuntimeContractError('mutation source dispatch decision missing')
    return prepare_mutation(initial, mutation, state, history)

def instantiate_request_v3(public, truth, template, cache, state, history, introduced=None):
    introduced = deepcopy(introduced or {})
    if (template.get('schema_version') != 3 or template.get('task_id') != public['task_id']
            or truth.get('task_id') != public['task_id'] or render_request(template) != public['instruction']
            or len(truth['goals']) != len(template['goals']) or not truth['goals']):
        raise RuntimeContractError('public request, frozen template and GT mismatch')
    state = normalize_state(cache, state)
    runtime, frozen = deepcopy(truth), deepcopy(template)
    runtime.update(state_assumption='actual_predispatch_state')
    runtime.pop('reference_steps',None)
    slots = deepcopy(template['preview_slots'])
    record = {'task_id':public['task_id'], 'changed':False, 'scored':True, 'dispatched':False,
              'steps':[], 'reason_code':None, 'conversions':[], 'failure_causes':[]}
    intros = []
    names = cache.get('names',{}).get('objects',{})
    current_binding, relevant = None, set()
    try:
        for index, (goal, spec) in enumerate(zip(runtime['goals'], template['goals'])):
            oid = spec['target_object_id']
            if goal['goal_id']!=spec['goal_id'] or goal['target']['object_ids'] != [oid]:
                raise RuntimeContractError('frozen goal identity mismatch')
            relevant.add(oid)
            relevant.update(dep for exp in cache['objects'][oid]['expressions'] for dep in exp['dependency_object_ids'])
            obj = state['objects'][oid]
            semantic, rule = spec['semantic'], spec['contract']
            if (goal['goal_kind'] != spec['goal_kind'] or goal['goal_kind'] != semantic['kind']
                    or goal.get('query') != semantic.get('query') or semantic['object_id'] != oid
                    or template['goal_order'] != [g['goal_id'] for g in truth['goals']]
                    or template['main']['kind'] != truth['goals'][0]['goal_kind']
                    or template['main']['query'] != truth['goals'][0].get('query')):
                raise RuntimeContractError('frozen semantic intent/order differs from GT')
            binding = spec.get('history_binding') or spec.get('place_binding')
            current_binding = binding
            event = bound_record(binding, oid, history) if binding else None
            converted, destination = False, None
            if event and event.get('dispatched') is False:
                if event.get('reason_code') == 'already_satisfied':
                    if goal['goal_kind'] == 'revisit_place':
                        raise RuntimeContractError('a place establishment cannot be already satisfied')
                    original_query = goal.get('query')
                    if goal['goal_kind'] == 'answer':
                        goal['query'] = 'current_room'
                        frozen['main'] = deepcopy(template['fallbacks']['current_room'])
                    elif goal['goal_kind'] == 'transfer':
                        if rule.get('destination_mode') == 'history':
                            destination = snapshot_room(event, 'decision_state', oid, cache)
                        frozen['main'] = deepcopy(template['fallbacks']['transfer'])
                    elif goal['goal_kind'] == 'visit':
                        if len(runtime['goals']) == 1:
                            frozen['main'] = deepcopy(template['fallbacks']['visit'])
                    else:
                        raise RuntimeContractError('unsupported undispatched-source conversion')
                    record['conversions'].append({'binding':deepcopy(binding), 'reason':'source_already_satisfied',
                        'query_before':original_query,'query_after':goal.get('query'),
                        'template_before':template['main']['template_id'], 'template_after':frozen['main']['template_id'],
                        'destination_room_id':destination})
                    goal.pop('memory_evidence',None)
                    goal['capabilities'] = [c for c in goal['capabilities'] if c not in {'memory','change_detection'}]
                    converted = True
                elif failed_causes(history, relevant, binding):
                    raise RepairUnavailable('unavailable', 'bound experience unavailable after earlier failure', code='source_experience_unavailable')
                else:
                    raise RuntimeContractError('undispatched source lacks a supported decision reason')
            elif event and event.get('dispatched') is not True:
                raise RuntimeContractError('source dispatch state missing')
            if (event and not converted and goal['goal_kind'] != 'revisit_place'
                    and (goal.get('query') in {'room_before','room_after','has_changed'} or rule.get('destination_mode') == 'history')
                    and event.get('location_experience') is not True):
                if event.get('success') is False:
                    raise RepairUnavailable('unavailable', 'failed source established no actual location experience', code='source_experience_unavailable')
                raise RuntimeContractError('location source lacks actual encounter evidence')
            hide = goal.get('query') in {'current_room','has_changed'}
            name = names.get(oid,{}).get('en')
            if not name: raise RuntimeContractError('v3 target lacks an approved name')
            phrase = 'the '+name
            key = 'objects/'+oid
            if index == 0 and 'topic' in frozen['main']['requires'] and key not in introduced:
                frozen['main'] = deepcopy(template['fallbacks'][goal['goal_kind']])
                record['steps'].append('replace_topic_expression_without_actual_prior_mention')
            if key not in introduced:
                record['steps'].append('try_current_answer_safe_identity')
                original = next((i for i in template['introductions'] if i['key']==key),None)
                description = introduction_reference(cache, state, oid, hides_room=hide, original=original)
                chosen = original['snapshot'] if original else template['introduction_fallback']
                intros.append({'key':key,'snapshot':deepcopy(chosen),'slots':{'description':description,'name':name}})
            if not event or converted:
                if index == 0: slots['target']=phrase
                slots[f'target_{index}']=phrase
                if index==0:slots['first_target']=phrase
                if index==len(runtime['goals'])-1:slots['second_target']=phrase
            elif goal.get('memory_evidence'):
                goal['memory_evidence']['event'] = deepcopy(event)
            if goal['goal_kind'] == 'transfer':
                if destination is None:
                    destination = (snapshot_room(event,rule['destination_phase'],oid,cache)
                                   if rule.get('destination_mode')=='history' else rule['destination_room_id'])
                if destination not in cache['areas']: raise RuntimeContractError('unknown destination')
                slots['destination'] = 'the '+area_name(cache,state,destination)
                goal['success_criteria']['room_id']=destination
                goal['target']['room_ids']=[obj['room_id'],destination]
                goal['expected_outcome']={'state_changes':[{'object_id':oid,
                    'relation_before':{'type':'in_room','room_id':obj['room_id']},
                    'relation_after':{'type':'in_room','room_id':destination}}]}
            elif goal['goal_kind']=='answer':
                query=goal['query']
                if query=='current_room':answer=obj['room_id']
                elif query in {'room_before','room_after'}:answer=snapshot_room(event,query.removeprefix('room_'),oid,cache)
                elif query=='has_changed':answer=obj['room_id']!=snapshot_room(event,'after',oid,cache)
                else:raise RuntimeContractError('unknown runtime query')
                if query=='has_changed' and frozen['main']['answer_mapping']=='not_has_changed':answer=not answer
                goal['expected_outcome']={'answer':answer,'answer_type':'boolean' if isinstance(answer,bool) else 'free_text',
                    'accepted_answers':[str(answer).lower()] if isinstance(answer,bool) else [area_name(cache,state,answer)]}
                if query in {'current_room','has_changed'}:
                    goal['inspection_contract']={'object_id':oid,**deepcopy(template['inspection_policy']),
                        'require_current_task_observation':True,'world_changes_allowed':False,'answer_mapping':frozen['main']['answer_mapping']}
                    goal['success_criteria']={'comparison':'accepted_answer','require_valid_observation':True}
                    goal['capabilities']=sorted(set(goal['capabilities'])|{'navigation','qa'})
            elif goal['goal_kind']=='revisit_place':
                if not event.get('place_established'):
                    if event.get('success') is False:
                        raise RepairUnavailable('unavailable','failed establishment did not create the named place',code='named_place_not_established')
                    raise RuntimeContractError('successful establishment lacks actual place evidence')
                if not event.get('place_geometry',{}).get('world_aabb'):raise RuntimeContractError('actual place geometry missing')
                goal['fixed_target_geometry']=deepcopy(event['place_geometry'])
                goal['named_place']['geometry']=deepcopy(event['place_geometry'])
            if goal['goal_kind'] in {'visit','revisit_place','establish_place'}:
                geom=goal['fixed_target_geometry'] if goal['goal_kind']=='revisit_place' else geometry(obj)
                if goal['goal_kind']=='establish_place':goal['named_place']['geometry']=deepcopy(geom)
                goal['expected_outcome']={'reached_target':True,'target_geometry':{'type':'aabb_2d',
                    'world_aabb_xy':{k:geom['world_aabb'][k][:2] for k in ('min','max')}}}
            goal['state_assumption']='actual_predispatch_state'
            for stale in ('reference_trajectory','reference_stages','reference_actions','placement_validation'):
                goal.pop(stale,None)
        text=render_request(frozen,slots,intros)
        record['steps'].append('render_frozen_semantics_and_refresh_gt')
        completion=completed(runtime['goals'],state)
        if completion is not None:
            record.update(scored=False,reason_code='already_satisfied',completion_evidence=completion)
            return {'status':'already_satisfied','public_task':None,'runtime_ground_truth':runtime,'repair_record':record,'name_ledger':introduced}
        task=deepcopy(public)
        task['instruction']=text
        task['capabilities']=sorted({c for g in runtime['goals'] for c in g['capabilities']})
        task['response_mode']='actions_and_answer' if 'qa' in task['capabilities'] else 'actions'
        task['name_introductions']=[render(i['snapshot']['expression'],i['slots']) for i in intros]
        for intro in intros:introduced[intro['key']]={'task_id':public['task_id'],'issued_at':public['issued_at']}
        record.update(changed=text!=public['instruction'] or runtime!=truth)
        return {'status':'ready','public_task':task,'runtime_ground_truth':runtime,'repair_record':record,'name_ledger':introduced}
    except RepairUnavailable as exc:
        causes=failed_causes(history,relevant,current_binding)
        if not causes:raise RuntimeContractError('repair failed without an evidenced prior failure: '+str(exc)) from exc
        record.update(scored=False,reason_code=exc.code,reason=str(exc),failure_causes=causes,
                      steps=[*record['steps'],'no_legal_identity_or_original_experience_after_repair'])
        return {'status':'unavailable','public_task':None,'runtime_ground_truth':None,'repair_record':record,'name_ledger':introduced}

def execution_events_v3(session_id, public, truth, before, after, *, dispatched, success,
                        goal_evidence=None, reason_code=None, repair_record=None):
    if type(dispatched) is not bool or (dispatched and type(success) is not bool) or (not dispatched and success is not None):
        raise RuntimeContractError('explicit actual dispatch/result required')
    if not dispatched and reason_code not in {'already_satisfied','unavailable'}:
        raise RuntimeContractError('undispatched decision requires its actual reason')
    records=[]
    for goal in truth['goals']:
        oid=goal['target']['object_ids'][0]
        evidence=(goal_evidence or {}).get(goal['goal_id'],{})
        event={'session_id':session_id,'task_id':public['task_id'],'goal_id':goal['goal_id'],
               'object_id':oid,'issued_at':public['issued_at'],'goal_kind':goal['goal_kind'],
               'dispatched':dispatched,'success':success,'reason_code':reason_code,'place_established':False,'location_experience':False,
               'failure_causes':deepcopy((repair_record or {}).get('failure_causes',[]))}
        if dispatched:
            if oid not in before.get('objects',{}) or oid not in after.get('objects',{}):
                raise RuntimeContractError('actual request boundary snapshots missing')
            # Original request boundaries, independent of individual goal timing.
            event.update(before=deepcopy(before),after=deepcopy(after),actions=deepcopy(evidence.get('actions',[])),
                         observation_evidence=deepcopy(evidence.get('observations',[])))
            if goal['goal_kind'] in {'visit','establish_place','revisit_place'}:
                event['location_experience'] = evidence.get('success') is True
            elif goal['goal_kind'] == 'answer' and goal.get('query') in {'current_room','has_changed'}:
                event['location_experience'] = evidence.get('observation_valid') is True
            elif goal['goal_kind'] == 'transfer':
                for action in evidence.get('actions', []):
                    name = action.get('action')
                    name = name if isinstance(name, str) else (name or {}).get('action')
                    result = action.get('result', {})
                    if name in {'pick','place'} and result.get('success') is True and result.get('object_id') == oid:
                        event['location_experience'] = True
            if evidence.get('place_established'):
                if goal['goal_kind']!='establish_place' or not evidence.get('place_geometry',{}).get('world_aabb'):
                    raise RuntimeContractError('invalid actual place establishment evidence')
                event.update(place_established=True,place_geometry=deepcopy(evidence['place_geometry']))
        else:
            event['decision_state']=deepcopy(before)
        records.append(event)
    return records
