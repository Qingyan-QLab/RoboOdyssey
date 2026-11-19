#!/usr/bin/env python3
"""Explicit private scripted Isaac diagnostics, never a model baseline.

--prepare-only validates and selects an existing frozen sequence on CPU. Real runs use the same v5 runner,
actual USD pick/place, observation evidence, recorder and repair machinery.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def prepare(source, sequence=None):
    from roboodyssey.scenes.dataset_loader import BenchmarkDataset
    data = BenchmarkDataset(source, sequence_id=sequence)
    scene = data.scene_ids()[0]
    sessions = data.sessions(scene)
    # Keep the original, complete sequence and its dependencies unchanged.
    candidates = [(session.session_id, truth['task_id'], truth['goals'][0])
        for session in sessions[:1] for truth in session.ground_truth()
        if len(truth['goals']) == 1 and truth['goals'][0]['goal_kind'] == 'transfer'
        and truth['goals'][0].get('expected_outcome', {}).get('placement_geometry')]
    if not candidates:
        raise ValueError('diagnostic requires a frozen single-target transfer in the first Session')
    return data, candidates[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sequence')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--case',choices=['held_failure','wrong_room','already_satisfied'],action='append')
    args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    ds, (source_session, source_task, truth) = prepare(args.source, args.sequence)
    archive = ds.root
    if args.prepare_only:
        (root/'diagnostic_selection.json').write_text(json.dumps({'dataset':str(archive),
            'sequence':ds.sequence_identity(),'source_session':source_session,'source_task':source_task,
            'complete_sequence':True,'not_model_baseline':True},indent=2)+'\n')
        print('CPU diagnostic selection:',archive);return 0
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':False,'active_gpu':args.gpu,'physics_gpu':args.gpu,'multi_gpu':False})
    error=None
    try:
        from roboodyssey.scenes.dataset_loader import BenchmarkDataset
        from roboodyssey.scenes.isaac_backend import make_env_factory
        from roboodyssey.viewer.viewport import _sync_viewer
        from roboodyssey.scenes.isaac_env import EnvConfig
        from roboodyssey.runtime.session_runner import SessionRunner
        from roboodyssey.runtime.request_runner import actual_state
        from roboodyssey.evaluation.report import write_summary
        from roboodyssey.viewer.viewer_overlay import ViewerOverlay
        scene=ds.scene_ids()[0];sessions=ds.sessions(scene)
        oid=truth['target']['object_ids'][0]
        cache=json.loads(sessions[0].private_paths['runtime_scene_cache'].read_text())
        name=cache['names']['objects'][oid]['en']
        results=[]
        for mode in args.case or ['held_failure','wrong_room','already_satisfied']:
            output=root/mode;output.mkdir(exist_ok=False)
            options={'layout':'front_overview','fps':4,'event_seconds':.5}
            overlay=ViewerOverlay();holder={};live={}
            factory=make_env_factory(ds,app,viewer=True,overview_holder=holder,video_options=options,camera_config=EnvConfig())
            def initialize(*a):
                env=factory(*a);live['env']=env
                initial=actual_state(env,cache)
                if mode=='already_satisfied':
                    env.apply_mutation_v5({'object_id':oid,'destination_room_id':truth['success_criteria']['room_id'],
                        'geometry':truth['expected_outcome']['placement_geometry']})
                box=actual_state(env,cache)['objects'][oid]['geometry']['world_aabb']
                center=[(x+y)/2 for x,y in zip(box['min'],box['max'])]
                # Controlled initial pose, never an injected model action.
                candidate=env._np.array([center[0]-.9,center[1],env._floor_height+env.config.camera_height])
                position=env._nearest_walkable_position(candidate)
                yaw=math.atan2(center[1]-position[1],center[0]-position[0])
                env.set_agent_pose({'position':position.tolist(),'orientation':[0,0,math.sin(yaw/2),math.cos(yaw/2)]})
                env._pitch=math.atan2(center[2]-position[2],math.dist(center[:2],position[:2]));env._author_camera_pose()
                env.render_once();app.update()
                from roboodyssey.runtime.request_events import state_changes
                before_inspection=actual_state(env,cache)
                _,first_evidence=env.observe_with_evidence([{'object_id':oid,'max_distance_m':2,'min_visible_pixels':8}])
                _,second_evidence=env.observe_with_evidence([{'object_id':oid,'max_distance_m':2,'min_visible_pixels':8}])
                changes=state_changes(before_inspection,actual_state(env,cache))
                (output/'inspection_invariance.json').write_text(json.dumps({'state_changes':changes,
                    'first':first_evidence,'second':second_evidence},indent=2))
                if changes['objects']:
                    raise RuntimeError('inspection changed the physical scene')
                (output/'diagnostic_setup.json').write_text(json.dumps({'private':True,'scripted':True,'not_model_baseline':True,
                    'case':mode,'object_id':oid,'initial_state':initial,'state_after_setup':actual_state(env,cache)},indent=2))
                return env
            class ScriptedAgent:
                def __init__(self):self.steps=0
                def send_observation(self,payload):pass
                def request(self,payload,on_wait=None):
                    if payload['type']=='session_start':
                        context=payload.get('sequence_context',{})
                        return {'sequence_reset_ack':context.get('run_id')} if context.get('reset_memory') else {}
                    if payload['session_id']==source_session and payload['task_id']==source_task:
                        self.steps+=1
                        action={'action':'pick','target':name} if self.steps==1 else {'action':'place'}
                        return {'action':action,'done':mode=='held_failure' or self.steps==2,'action_source':'scripted_diagnostic'}
                    return {'agent_status':'unsupported','reason_code':'unsupported_capability',
                        'done':True,'action_source':'scripted_diagnostic'}
            def show(**event):overlay.update(**event);app.update()
            try:
                runner=SessionRunner(output,ScriptedAgent(),max_steps=2,video_options=options,viewer_callback=show,
                    viewer_sync_callback=lambda env:_sync_viewer(holder.get('front_viewport'),holder.get('window'),
                        env.config.camera_prim_path,holder.get('overview_camera_path'),app))
                runner.run_scene(scene,sessions,ds.canonical_state(scene),initialize)
                summary=write_summary(ds,output)
                audit=json.loads((output/'request_audit.json').read_text())
                first=next(t for t in audit['tasks'] if t['session_id']==source_session and t['task_id']==source_task)
                outcome=first.get('outcome') or {}
                checks={'run_valid':summary['evaluation_valid']}
                if mode=='already_satisfied':
                    checks['source_not_dispatched']=first['repair_status']=='already_satisfied'
                else:
                    checks['source_failed']=outcome.get('success') is False
                    changes=outcome.get('state_changes',{})
                    checks['actual_objects_changed']=bool(changes.get('objects'))
                    checks['holding_matches_case']=(changes.get('held_object',{}).get('after')==oid if mode=='held_failure'
                        else changes.get('held_object',{}).get('after') is None)
                result={'case':mode,'checks':checks,'passed':all(checks.values()),'not_model_baseline':True}
                results.append(result);print('DIAGNOSTIC',json.dumps(result),flush=True)
                (root/'diagnostic_results.json').write_text(json.dumps(results,indent=2)+'\n')
            finally:
                overlay.close()
                if holder.get('window'):holder['window'].destroy()
        if not all(r['passed'] for r in results):raise RuntimeError('one or more diagnostic checks failed')
    except BaseException as exc:
        import traceback
        traceback.print_exc();error=exc
        (root/'diagnostic_error.json').write_text(json.dumps({'type':type(exc).__name__,'error':str(exc)}))
    finally:
        app.close(exit_code=1 if error else 0)
    return 1 if error else 0


if __name__=='__main__':raise SystemExit(main())
