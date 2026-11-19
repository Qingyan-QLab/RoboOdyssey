"""Frozen-data runtime helpers. Adapted from MIT-licensed sources; see LICENSE."""
from __future__ import annotations

from copy import deepcopy

import re

from .references import RepairUnavailable, object_reference, area_name

from .state import geometry

from .language import validate_public_wording

from .language import historical_reference

def execution_events(session_id, public, truth, before, after, *, dispatched, success, goal_evidence=None):
    """Package actual runner snapshots; never substitute offline expected history.

    Single-object tasks use request snapshots. A runner can supply per-goal
    snapshots and place evidence keyed by goal_id for multi-target requests.
    The caller determines success using executed evidence, not this helper.
    """
    if type(dispatched) is not bool or (success is not None and type(success) is not bool):
        raise ValueError('actual dispatch and success flags must be explicit')
    if not dispatched and success is not None:
        raise ValueError('an undispatched request has no execution result')
    records = []
    for goal in truth['goals']:
        evidence = (goal_evidence or {}).get(goal['goal_id'], {})
        first = evidence.get('before', before)
        last = evidence.get('after', after)
        oid = goal['target']['object_ids'][0]
        if dispatched and any(oid not in state.get('objects', {}) for state in (first, last)):
            raise ValueError('dispatched history needs actual before/after target snapshots')
        event = {'session_id': session_id, 'task_id': public['task_id'], 'goal_id': goal['goal_id'],
                 'object_id': oid, 'issued_at': public['issued_at'], 'goal_kind': goal['goal_kind'],
                 'dispatched': dispatched, 'success': success, 'before': deepcopy(first), 'after': deepcopy(last),
                 'actions': deepcopy(evidence.get('actions', [])), 'place_established': False}
        if evidence.get('place_established'):
            if not dispatched or goal['goal_kind'] != 'establish_place' or not evidence.get('place_geometry'):
                raise ValueError('place establishment needs dispatched establishment evidence and actual geometry')
            event.update(place_established=True, place_geometry=deepcopy(evidence['place_geometry']))
        records.append(event)
    return records

def history_event(binding, oid, history):
    matches = [e for e in history if all(e.get(k) == binding[k] for k in ('session_id','task_id','goal_id'))
               and e.get('object_id') == oid]
    if len(matches) != 1:
        raise RepairUnavailable('unavailable', 'bound original request evidence is missing or ambiguous',
                                code='history_record_missing' if not matches else 'history_record_ambiguous', dependency=binding)
    event = matches[0]
    if event.get('dispatched') is not True:
        raise RepairUnavailable('unavailable', 'bound request was not dispatched', code='source_not_dispatched',dependency=binding)
    for key in ('before','after'):
        if oid not in event.get(key, {}).get('objects', {}):
            raise RepairUnavailable('unavailable', 'bound actual request snapshot is missing', code='history_snapshot_missing',dependency=binding)
    return event

def completed_before_dispatch(goals, state):
    evidence = []
    for goal in goals:
        kind = goal['goal_kind']; oid = goal['target']['object_ids'][0]
        obj = state['objects'].get(oid, {})
        if kind == 'transfer':
            target = goal['success_criteria']['room_id']
            box = geometry(obj).get('world_aabb')
            floor = state.get('floor_heights', {}).get(target, 0)
            on_floor = bool(box and abs(box['min'][2] - floor) <= .04)
            if obj.get('room_id') != target or obj.get('held') or not on_floor:
                return None
            evidence.append({'goal_id':goal['goal_id'],'room_id':target,'released':True,'on_floor':True})
        elif kind == 'visit':
            robot = state.get('robot_pose', {}).get('position')
            box = geometry(obj).get('world_aabb')
            if not robot or not box:
                return None
            distance = sum(max(box['min'][i]-robot[i],0,robot[i]-box['max'][i])**2 for i in (0,1))**.5
            if distance > goal['success_criteria'].get('position_tolerance_m',.8):
                return None
            evidence.append({'goal_id':goal['goal_id'],'distance_m':distance})
        else:
            # An answer or memory-establishment request needs new experience.
            return None
    return evidence

def instantiate_request(public, truth, template, cache, state, history, introduced=None):
    """Return ready/already_satisfied/unavailable; no partial request is dispatched."""
    if template.get('schema_version') == 3:
        from .contextual import instantiate_request_v3
        return instantiate_request_v3(public, truth, template, cache, state, history, introduced)
    introduced = deepcopy(introduced or {})
    record = {'task_id':public['task_id'],'changed':False,'scored':True,'dispatched':False,
              'steps':[],'reason_code':None,'dependency':None}
    try:
        if (template.get('schema_version') != 2 or template.get('task_id') != public['task_id']
                or truth.get('task_id') != public['task_id'] or not truth.get('goals')
                or len(truth['goals']) != len(template.get('goals', []))):
            raise RepairUnavailable('unavailable', 'request/template identity or goal count mismatch', code='invalid_runtime_contract')
        runtime = deepcopy(truth)
        runtime['state_assumption'] = 'actual_predispatch_state'
        runtime.pop('reference_steps',None)
        slots = deepcopy(template['preview_slots'])
        intros = []
        for index,(goal, contract) in enumerate(zip(runtime['goals'],template['goals'])):
            oid = contract['target_object_id']
            if goal['goal_id'] != contract['goal_id'] or goal['target']['object_ids'] != [oid]:
                raise RepairUnavailable('unavailable', 'goal identity differs from frozen contract', code='invalid_runtime_contract')
            if oid not in state.get('objects',{}):
                raise RepairUnavailable('unavailable','bound target is absent',code='target_absent')
            obj = state['objects'][oid]; rule = contract['contract']
            binding = contract.get('history_binding')
            event = history_event(binding,oid,history) if binding else None
            goal['state_assumption'] = 'actual_predispatch_state'
            if event:
                goal['memory_evidence']['event'] = deepcopy(event)
            key='objects/'+oid
            name=cache.get('names',{}).get('objects',{}).get(oid,{}).get('en')
            hide=goal.get('query') in {'current_room','has_changed'}
            if name and key not in introduced:
                desc=object_reference(cache,state,oid,hides_room=hide,preferred=contract['source_reference'])
                from .language import shorten_areas
                desc=shorten_areas(desc,cache['names'],cache.get('original_area_descriptions',{}))
                intros.append({'key':key,'text':f'Call {desc} the {name}'})
            if event:
                # Keep the original time and identity even if its requested action failed.
                reference=deepcopy(rule['historical_reference'])
                if name: reference['object_name']=name
                phrase=historical_reference(reference)
                record['steps'].append('resolve_original_request_actual_snapshots')
            elif name:
                phrase='the '+name
            else:
                phrase=object_reference(cache,state,oid,hides_room=hide,preferred=contract['source_reference'])
            if f'goal_{index}_target' in slots:
                slots[f'goal_{index}_target']=phrase
            if goal['goal_kind']=='transfer':
                destination=rule.get('destination_room_id')
                if rule.get('destination_mode')=='history':
                    destination=event[rule['destination_phase']]['objects'][oid].get('room_id')
                if destination not in cache['areas']:
                    raise RepairUnavailable('unavailable','destination is not in current scene',code='destination_unknown')
                goal['target']['room_ids']=[obj.get('room_id'),destination]
                goal['preconditions']=[{'type':'in_room','room_id':obj.get('room_id')}]
                goal['expected_outcome']={'state_changes':[{'object_id':oid,
                    'relation_before':{'type':'in_room','room_id':obj.get('room_id')},
                    'relation_after':{'type':'in_room','room_id':destination}}]}
                goal['success_criteria']['room_id']=destination
                if f'goal_{index}_destination' in slots:
                    slots[f'goal_{index}_destination']='the '+area_name(cache,state,destination)
            elif goal['goal_kind']=='answer':
                query=goal['query']
                current=obj.get('room_id')
                if query=='current_room': answer=current
                elif query in {'room_before','room_after'}:answer=event[query.removeprefix('room_')]['objects'][oid].get('room_id')
                else:answer=current!=event['after']['objects'][oid].get('room_id')
                accepted=[str(answer).lower()] if isinstance(answer,bool) else [area_name(cache,state,answer)]
                goal['expected_outcome']={'answer':answer,'accepted_answers':accepted,
                    'answer_type':'boolean' if isinstance(answer,bool) else 'free_text'}
                goal['target']['room_ids']=[current]
            elif goal['goal_kind']=='revisit_place':
                place_binding=contract.get('place_binding')
                place=history_event(place_binding,oid,history)
                if not place.get('place_established'):
                    raise RepairUnavailable('unavailable','named place was not established',code='named_place_not_established',dependency=place_binding)
                goal['fixed_target_geometry']=deepcopy(place['place_geometry'])
                goal['named_place']['geometry']=deepcopy(place['place_geometry'])
                box=geometry({'geometry': place['place_geometry']})['world_aabb']
                goal['expected_outcome']={'reached_target':True,'target_geometry':{'type':'aabb_2d',
                    'world_aabb_xy':{k:box[k][:2] for k in ('min','max')}}}
            elif goal['goal_kind']=='visit':
                goal['target']['room_ids']=[obj.get('room_id')]
                goal['preconditions']=[{'type':'in_room','room_id':obj.get('room_id')}]
                goal['expected_outcome']={'reached_target':True,'target_geometry':{'type':'aabb_2d','world_aabb_xy':
                    {k:geometry(obj)['world_aabb'][k][:2] for k in ('min','max')}}}
            elif goal['goal_kind']=='establish_place':
                goal['target']['room_ids']=[obj.get('room_id')]
                goal['named_place']['geometry']=deepcopy(geometry(obj))
                goal['expected_outcome']={'reached_target':True,'target_geometry':{'type':'aabb_2d','world_aabb_xy':
                    {k:geometry(obj)['world_aabb'][k][:2] for k in ('min','max')}}}
            for stale in ('reference_trajectory','reference_stages','reference_actions','placement_validation'):
                goal.pop(stale,None)
            record['steps'].append('refresh_'+goal['goal_id'])
        original_intros=template.get('introductions',[])
        pending={v['key']:v['text'] for v in intros}
        for index,intro in enumerate(original_intros):
            slots[f'introduction_{index}']=pending.pop(intro['key'],'')
        frozen = template['text']
        for index in range(len(original_intros)):
            key = f'introduction_{index}'
            if not slots[key]:
                frozen = re.sub(r'(?:please\s+)?\{' + key + r'\}\s*(?:[,;:.]\s*)?(?:(?:and\s+)?then\s+|and\s+)?', '', frozen, flags=re.I)
        text=frozen.format_map(slots)
        # Definitions lost through an undispatched earlier task are reintroduced.
        if pending:text='. '.join(pending.values())+'. '+text
        text=re.sub(r'^\s*(?:[,;.]+\s*|then\s+|and\s+)+','',text,flags=re.I)
        text=re.sub(r'[,;]\s*(?:then\s*)?(?=[,;])','',text,flags=re.I)
        validate_public_wording(text)
        record['steps'].append('validate_and_render_current_wording')
        completion=completed_before_dispatch(runtime['goals'],state)
        if completion is not None:
            record.update(scored=False,reason_code='already_satisfied',completion_evidence=completion)
            return {'status':'already_satisfied','public_task':None,'runtime_ground_truth':runtime,'repair_record':record,'name_ledger':introduced}
        task=deepcopy(public);task['instruction']=text;task['name_introductions']=[v['text'] for v in intros]
        for intro in intros:introduced[intro['key']]={'task_id':public['task_id'],'issued_at':public['issued_at']}
        record['changed']=text!=public['instruction'] or runtime!=truth
        return {'status':'ready','public_task':task,'runtime_ground_truth':runtime,'repair_record':record,'name_ledger':introduced}
    except RepairUnavailable as exc:
        record.update(scored=False,reason_code=exc.code,dependency=exc.dependency,reason=str(exc))
        return {'status':'unavailable','public_task':None,'runtime_ground_truth':None,'repair_record':record,'name_ledger':introduced}

def prepare_mutation(initial, mutation, state, history):
    """Revalidate a planned move in real state. Caller applies returned geometry.

    Never reuse a stale planned placement. Held/absent targets or an unavailable
    check are explicitly skipped; the following QA is repaired against real state.
    """
    from .placement import floor_placement
    from .raster import derive_navigation
    from .navigation import validate_connectivity
    change=mutation['changes'][0];oid=change['object_id'];destination=change['relation_after']['room_id']
    try:
        obj=deepcopy(state['objects'].get(oid))
        if oid not in initial.engine.pick_ids:
            raise ValueError('mutation target is not an approved movable object')
        if obj is None or obj.get('held') or obj.get('room_id')==destination:
            raise ValueError('target absent, held, or already in the planned destination')
        if not any(e.get('object_id')==oid and e.get('dispatched') is True and e.get('after',{}).get('objects',{}).get(oid) for e in history):
            raise ValueError('no actual earlier request experience for mutation target')
        if mutation.get('inspection_binding'):
            history_event(mutation['inspection_binding'], oid, history)
        entities=[]
        for source in state['objects'].values():
            entity=deepcopy(source);entity['geometry']=geometry(source);entities.append(entity)
        obj=next(e for e in entities if e['object_id']==oid)
        room=next(r for r in initial.engine.rooms if r['room_id']==destination)
        nav=derive_navigation(initial.engine.rooms,entities,resolution=initial.engine.navigation['resolution'],
            robot_radius=initial.engine.navigation['robot_radius'],environment_setup=initial.engine.navigation.get('environment_setup'))
        nav['scene_id']=initial.metadata['scene']['scene_id']
        pose=state.get('robot_pose',{}).get('position')
        if not pose:raise ValueError('actual robot pose required for a safe mutation approach')
        start=tuple(int((pose[i]-nav['origin_xy'][i])/nav['resolution']) for i in (0,1))
        placement=floor_placement(obj,room,entities,initial.engine.rooms,nav,start=start)
        final_nav = placement['navigation']
        x, y = start
        if (not 0 <= y < len(final_nav['static_occupancy']) or not 0 <= x < len(final_nav['static_occupancy'][y])
                or final_nav['static_occupancy'][y][x] or final_nav['dynamic_occupancy'][y][x]):
            raise ValueError('mutation would obstruct the actual robot pose')
        from types import SimpleNamespace
        validate_connectivity(SimpleNamespace(rooms=initial.engine.rooms, navigation=final_nav))
        return {'status':'ready','object_id':oid,'destination_room_id':destination,
                'geometry':placement['entity']['geometry'],'validation':placement['validation']}
    except (ValueError,KeyError,StopIteration) as exc:
        return {'status':'skipped','object_id':oid,'reason':str(exc)}
