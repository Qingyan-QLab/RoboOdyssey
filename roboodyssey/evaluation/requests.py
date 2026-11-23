"""Complete-request scoring from actual goal, action and observation evidence."""
from ..interaction.protocol import action_name
from copy import deepcopy
from .metrics.qa import contextual_answer_correct
from .task_metrics import metric_totals

from roboodyssey.contracts.contextual import distance_to_box, RuntimeContractError
from roboodyssey.contracts.state import geometry




def evaluate_request(truth, state, poses, actions, observations, answer):
    results=[]
    previous_visit=-1
    for goal in truth['goals']:
        oid=goal['target']['object_ids'][0]
        kind=goal['goal_kind']
        evidence={'goal_id':goal['goal_id'],'success':False}
        obj=state['objects'][oid]
        if kind in {'visit','establish_place','revisit_place'}:
            box=(goal['fixed_target_geometry'] if kind=='revisit_place' else geometry(obj))['world_aabb']
            tolerance=goal['success_criteria'].get('position_tolerance_m',.8)
            reached=next((i for i,p in enumerate(poses) if i>previous_visit and distance_to_box(p['position'],box)<=tolerance),None)
            evidence['success']=reached is not None
            if reached is not None:
                previous_visit=reached
                evidence['pose_index']=reached
                if kind=='establish_place':
                    # The named spot is the actual reached position, not the
                    # offline landmark box. Subsequent landmark movement is irrelevant.
                    p=poses[reached]['position']
                    evidence.update(place_established=True,place_geometry={'world_aabb':{
                        'min':[p[0]-.05,p[1]-.05,p[2]-.05],'max':[p[0]+.05,p[1]+.05,p[2]+.05]}})
            else:previous_visit=len(poses)
        elif kind=='transfer':
            room=goal['success_criteria']['room_id']
            box=geometry(obj)['world_aabb']
            released=not obj.get('held') and state.get('held_object')!=oid
            floor=state.get('floor_heights',{}).get(room,0)
            placed=any(action_name(a)=='place' and a.get('result',{}).get('success') is True
                       and a['result'].get('object_id')==oid for a in actions)
            evidence.update(success=obj['room_id']==room and released and abs(box['min'][2]-floor)<=.04 and placed,
                            released=released,place_event=placed)
        elif kind=='answer':
            correct=contextual_answer_correct(answer,goal['expected_outcome'])
            evidence.update(success=correct,answer_correct=correct)
        else:raise RuntimeContractError('unknown goal kind')
        evidence['observations']=deepcopy([e for e in observations if e['object_id']==oid])
        evidence['actions']=deepcopy(actions)
        results.append(evidence)
    return {'success':all(e['success'] for e in results), 'goals':results}


def request_totals(results, *, valid=True):
    excluded={'already_satisfied':0,'unavailable':0}
    for result in results:
        if result.get('scored',True):continue
        status=result.get('reason_code')
        if status not in excluded or (status=='unavailable' and not result.get('failure_causes')):
            raise RuntimeContractError('unproved scoring exclusion')
        excluded[status]+=1
    scored=[r for r in results if r.get('scored',True)]
    success_rate=sum(r['success'] is True for r in scored)/len(scored) if valid and scored else None
    totals={'tasks_total':len(results),'tasks_evaluated':len(scored),'tasks_excluded':sum(excluded.values()),
            'exclusions':excluded,'evaluation_valid':valid,
            'task_score':success_rate, 'task_success_rate':success_rate,
            'success':all(r['success'] is True for r in scored) if valid and scored else None}
    totals.update(metric_totals(results, valid=valid))
    totals['task_score'] = totals['score_100']/100 if totals['score_100'] is not None else None
    return totals
