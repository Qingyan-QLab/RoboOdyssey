"""Frozen-data runtime helpers. Adapted from MIT-licensed sources; see LICENSE."""
from __future__ import annotations

from copy import deepcopy

from .references import RepairUnavailable

from .state import geometry



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
        if initial.engine.navigation.get('robot_profile') is not None:
            nav['robot_profile'] = deepcopy(initial.engine.navigation['robot_profile'])
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
