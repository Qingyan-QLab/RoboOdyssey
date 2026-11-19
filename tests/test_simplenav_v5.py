from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from benchmark.agents.simplenav import SimpleNavAgent
from benchmark.runtime.recording import SessionRecorder, RecordingError
from benchmark.launch.simplenav import load_config, commands, preflight
from benchmark.runtime.request_runner import apply_boundary
from benchmark.runtime.session_runner import SessionRunner
from benchmark.reporting.request_audit import write_request_audit, lines
from benchmark.evaluation.report import write_summary
from tests.test_contextual_requests import Env, Agent, fixture_v3, task, goal, SOURCE, select_scenario, write_dataset, BenchmarkDataset


class SimpleNavProtocolTests(unittest.TestCase):
    def test_continuous_frames_enter_model_history_with_real_capture_times(self):
        agent = self.agent()
        captured = []
        agent.adapter.build_example = lambda **kw: captured.append(kw) or {'instruction': kw['instruction']}
        history = []
        for frame_id in (1, 2, 3):
            history.append({'images': {'front': np.full((4,4,3), frame_id, np.uint8)},
                'timing': {'frame_id': frame_id, 'captured_monotonic_s': 10 + .5 * frame_id}})
        current = self.request(observation_history=history)
        current['observation']['timing'] = {'frame_id': 4, 'captured_monotonic_s': 12.0}
        agent.handle(current)
        model_history = captured[0]['history'].observations
        self.assertEqual([o['navvla_eval']['frame_index'] for o in model_history], [1, 2, 3])
        self.assertEqual([o['navvla_eval']['timestamp'] for o in model_history], [10.5, 11.0, 11.5])
        self.assertEqual(captured[0]['observation']['navvla_eval']['frame_index'], 4)

    def agent(self):
        self.calls=[]
        policy=SimpleNamespace(predict_action=lambda q:self.calls.append(q) or {'normalized_actions':[[.2,0,0,0]]})
        codec=SimpleNamespace(decode=lambda q:SimpleNamespace(raw_actions=q['normalized_actions']))
        adapter=SimpleNamespace(prepare_observation_for_step=lambda **kw:kw['observation'],
            build_example=lambda **kw:{'instruction':kw['instruction']},update_history=lambda **kw:None)
        return SimpleNavAgent(policy,codec,adapter,lambda:SimpleNamespace(images=[],observations=[],instructions=[]),collision_reset_threshold=1)

    def request(self, **changes):
        return {'type':'step','scene_id':'scene','session_id':'session_000','task_id':'task_000',
                'task_type':'long_horizon','capabilities':['navigation'], 'instruction':'Go to the lamp.',
                'current_time':'2026-11-05T09:00:00+08:00',
                'observation':{'images':{'front':np.zeros((10,10,3),dtype=np.uint8)}},**changes}

    def test_v5_calls_model_with_public_memory_across_sessions(self):
        agent=self.agent()
        agent.handle({'type':'session_start','scene_id':'scene','instruction':'Help me.', 'scene_briefing':['Call this area the study.']})
        result=agent.handle(self.request(name_introductions=['Call the tall light the lamp.']))
        self.assertEqual(result['action']['action'],'move_forward')
        agent.handle({'type':'session_start','scene_id':'scene','instruction':'Continue.'})
        agent.handle(self.request(session_id='session_001'))
        text=self.calls[-1]['examples'][0]['instruction']
        self.assertIn('study',text);self.assertIn('tall light',text);self.assertIn('2026-11-05',text)
        self.assertIn('not evidence of success',text)
        self.assertEqual(len(agent.requests),2)
        agent.handle({'type':'session_start','scene_id':'other','instruction':'New scene.'})
        self.assertFalse(agent.definitions);self.assertFalse(agent.requests)

    def test_unsupported_is_explicit_and_does_not_call_model(self):
        agent=self.agent()
        for caps in (['navigation','qa'],['navigation','manipulation']):
            result=agent.handle(self.request(capabilities=caps))
            self.assertEqual(result['agent_status'],'unsupported');self.assertTrue(result['done'])
        self.assertEqual(self.calls,[])
        with self.assertRaisesRegex(ValueError,'capabilities'):agent.handle(self.request(capabilities=[]))

    def test_long_public_memory_cannot_push_current_request_out_of_model_prefix(self):
        agent=self.agent()
        agent.definitions=['A previous object has a long public description. '*400]
        agent.requests=[{'time':'old','instruction':'Old request. '*200} for _ in range(32)]
        request=self.request(instruction='Go to the new lamp beside the sofa.')
        agent.handle(request)
        text=self.calls[-1]['examples'][0]['instruction']
        # The backend truncates from the right. Even a conservative short
        # retained prefix must include the current instruction and its time.
        self.assertIn(request['instruction'],text[:1024])
        self.assertIn(request['current_time'],text[:1024])

    def test_service_error_and_bad_waypoint_are_not_task_failures(self):
        agent=self.agent();agent.policy.predict_action=lambda q:{'ok':False,'error':'offline'}
        with self.assertRaisesRegex(RuntimeError,'offline'):agent.handle(self.request())
        agent.policy.predict_action=lambda q:{'normalized_actions':[[float('nan'),0,0,0]]}
        with self.assertRaisesRegex(ValueError,'non-finite'):agent.handle(self.request())

    def test_recovery_respects_allowed_actions_and_legacy_navigation_works(self):
        agent=self.agent();request=self.request(allowed_actions=['move_forward'])
        request['observation']['action_feedback']={'collision':True}
        self.assertEqual(agent.handle(request)['action']['action'],'move_forward')
        request.pop('allowed_actions')
        self.assertEqual(agent.handle(request)['action']['action'],'reset')
        self.assertEqual(agent.handle(self.request(task_type='navigation'))['action']['action'],'move_forward')


class RecordingTests(unittest.TestCase):
    def event(self, n=1):
        return {'event_id':f'session_000:{n:06d}','event':'task_start','scene_id':'scene','session_id':'session_000',
                'task_id':'task','utc':'2026-11-10T00:00:00+00:00','monotonic_ns':n,'instruction':'A request — with punctuation.'}

    def test_real_cpu_encoder_finalizes_interrupted_file_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rec=SessionRecorder(root,fps=4,event_seconds=.5)
            rec.capture(self.event(),{'front':np.full((40,60,3),(255,0,0),np.uint8),
                                      'overview':np.full((40,60,3),(0,255,0),np.uint8)})
            rec.capture(self.event(2));rec.close('interrupted');rec.close('interrupted')
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                'stream=width,height,nb_frames','-of','json',str(root/'review.mp4')]))['streams'][0]
            self.assertEqual((probe['width'],probe['height'],int(probe['nb_frames'])),(1280,960,4))
            index=lines(root/'video_index.jsonl');self.assertEqual(index[1]['video_seconds'],.5)
            self.assertFalse(index[1]['new_camera_sample'])
            manifest=json.loads((root/'video_manifest.json').read_text())
            self.assertEqual(manifest['run_status'],'interrupted');self.assertEqual(manifest['status'],'finalized')
            # Decode pixels to verify both panels were actually encoded.
            frame=subprocess.check_output(['ffmpeg','-v','error','-i',str(root/'review.mp4'),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
            array=np.frombuffer(frame,np.uint8).reshape(960,1280,3)
            self.assertGreater(int(array[200,200,0]),200);self.assertGreater(int(array[200,800,1]),200)

    def test_missing_overview_is_an_error_not_a_fake_blank_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec=SessionRecorder(tmp)
            with self.assertRaisesRegex(RecordingError,'missing'):rec.capture(self.event(),{'front':np.zeros((4,4,3),np.uint8)})
            rec.close('interrupted')
            self.assertEqual(json.loads((Path(tmp)/'video_manifest.json').read_text())['status'],'empty')

    def test_encoder_failure_is_reported_and_streams_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec=SessionRecorder(tmp,layout='front')
            rec.capture(self.event(),{'front':np.zeros((20,20,3),np.uint8)})
            rec.process.kill();rec.process.wait()
            with self.assertRaises(RecordingError):rec.close('interrupted')
            self.assertTrue(rec.index.closed);self.assertTrue(rec.error_log.closed)
            self.assertEqual(json.loads((Path(tmp)/'video_manifest.json').read_text())['status'],'error')


class CapturingEnv(Env):
    def capture_review_frames(self,layout):
        return {'front':np.zeros((24,32,3),np.uint8),'overview':np.full((24,32,3),90,np.uint8)}


class V5TraceTests(unittest.TestCase):
    def test_unchanged_usd_aabb_keeps_reference_despite_different_geometry_encoding(self):
        from benchmark.contracts.references import same_location
        before={'room_id':'study','geometry':{'world_aabb':{'min':[0,1,2],'max':[1,2,3]},
            'world_matrix4x4':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}}
        actual=deepcopy(before);actual['geometry'].pop('world_matrix4x4');actual['held']=False
        actual['geometry']['world_aabb']['min'][0]+=1e-6
        self.assertTrue(same_location(actual,before))
        actual['geometry']['world_aabb']['min'][0]+=.01
        self.assertFalse(same_location(actual,before))

    def run_case(self, tmp, result, compiler, inputs, agent, env_class=CapturingEnv, **kwargs):
        archive=Path(tmp)/'archive';output=Path(tmp)/'run';write_dataset(archive,result,compiler,inputs)
        ds=BenchmarkDataset(archive);env=env_class(ds.canonical_state('synthetic_demo'))
        shown=[]
        runner=SessionRunner(output,agent,max_steps=2,viewer_callback=lambda **v:shown.append(v),**kwargs)
        runner.run_scene('synthetic_demo',ds.sessions('synthetic_demo'),ds.canonical_state('synthetic_demo'),lambda *a:env)
        return ds,output,env,shown

    def test_failed_pick_changes_state_and_later_gt_is_repaired(self):
        c,i,p=fixture_v3(second=[task('ask',goal('book_red','answer',query='room_after',history=SOURCE))]);r=select_scenario(c,p)
        class Picker(Agent):
            def request(self,payload,on_wait=None):
                response=super().request(payload,on_wait)
                return {'action':{'action':'pick','target':'reading book'},'done':True} if payload.get('task_id')=='move' else response
        class Picked(CapturingEnv):
            def step(self,action):
                if isinstance(action,dict) and action.get('action')=='pick':
                    self.state['held_object']='book_red';self.state['objects']['book_red']['held']=True
                    return self.observe(),0,False,False,{'success':True,'object_id':'book_red'}
                return super().step(action)
        with tempfile.TemporaryDirectory() as tmp:
            agent=Picker();ds,out,env,events=self.run_case(tmp,r,c,i,agent,Picked)
            report=json.loads((out/'request_audit.json').read_text());first,second=report['tasks']
            self.assertFalse(first['outcome']['success'])
            self.assertEqual(first['outcome']['state_changes']['held_object']['after'],'book_red')
            self.assertEqual(second['original_ground_truth']['goals'][0]['expected_outcome']['answer'],'kitchen')
            self.assertEqual(second['runtime_ground_truth']['goals'][0]['expected_outcome']['answer'],'study')
            self.assertFalse(second['source_context'][0]['success'])
            self.assertTrue(second['source_context'][0]['actual_state_changes']['objects'])
            summary=write_summary(ds,out)
            self.assertIsNone(summary['task_score'])  # No transfer reference in this history-only fixture.
            self.assertEqual(summary['task_success_rate'],.5)
            from benchmark.evaluation.task_diagnostics import analyze_run
            (out/'run_config.json').write_text(json.dumps({'official':False}))
            diagnostic=analyze_run(out)
            self.assertEqual(diagnostic['issues'],[])
            self.assertIsNone(diagnostic['summary']['formal_rule_average'])
            self.assertIn('task_repair',[e['event'] for e in events]);self.assertTrue(env.closed)
            steps=[q for q in agent.requests if q['type']=='step']
            self.assertIn('capabilities',steps[0]);self.assertIn('name_introductions',steps[0])
            self.assertNotIn('ground_truth',json.dumps(steps))

    def test_satisfied_source_conversion_has_video_and_exact_dispatch_text(self):
        c,i,p=fixture_v3(second=[task('ask',goal('book_red','answer',query='room_after',history=SOURCE))]);r=select_scenario(c,p)
        class Satisfied(CapturingEnv):
            def __init__(self,state):
                super().__init__(state)
                obj=self.state['objects']['book_red'];obj['room_id']='kitchen'
                obj['geometry']={'world_aabb':{'min':[4,1,0],'max':[4.2,1.2,.2]}}
        class Kitchen(Agent):
            def request(self,payload,on_wait=None):
                response=super().request(payload,on_wait)
                if payload['type']=='step':response['answer']='kitchen'
                return response
        with tempfile.TemporaryDirectory() as tmp:
            agent=Kitchen();ds,out,env,events=self.run_case(tmp,r,c,i,agent,Satisfied,
                video_options={'fps':2,'event_seconds':.5})
            report=json.loads((out/'request_audit.json').read_text());first,second=report['tasks']
            self.assertEqual(first['repair_status'],'already_satisfied')
            self.assertEqual(second['runtime_ground_truth']['goals'][0]['query'],'current_room')
            self.assertTrue(second['repair_record']['conversions'])
            self.assertNotIn('asked you',second['issued_instruction'])
            self.assertEqual(second['issued_instruction'],[q for q in agent.requests if q['type']=='step'][0]['instruction'])
            self.assertTrue((out/second['video']['path']).is_file())
            self.assertEqual(write_summary(ds,out)['tasks_excluded'],1)

    def test_backend_error_invalidates_run_but_keeps_finalized_video(self):
        c,i,p=fixture_v3([task('ask',goal('book_red','answer',query='current_room'))]);r=select_scenario(c,p)
        class Broken(Agent):
            def request(self,payload,on_wait=None):
                if payload['type']=='session_start':return {}
                return {'action':'no_op','agent_status':'integration_error','reason_code':'server_unreachable','error':'offline'}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'offline'):
                self.run_case(tmp,r,c,i,Broken(),video_options={'layout':'front','fps':2,'event_seconds':.5})
            out=Path(tmp)/'run';session=out/'scenes/synthetic_demo/sessions/session_000'
            self.assertIsNone(json.loads((session/'metrics.json').read_text())['task_score'])
            self.assertEqual(json.loads((session/'video_manifest.json').read_text())['run_status'],'interrupted')
            audit=json.loads((out/'request_audit.json').read_text())
            self.assertTrue(audit['integration_errors']);self.assertEqual(audit['incomplete_tasks'],1)
            io=[e for e in lines(session/'agent_io.jsonl') if e['request_type']=='step']
            self.assertEqual([e['direction'] for e in io],['request','response','error'])
            self.assertEqual(io[0]['payload']['instruction'],audit['tasks'][0]['issued_instruction'])

    def test_failed_place_establishment_excludes_revisit_with_proven_source(self):
        source={'session_id':'session_000','task_id':'name','goal_id':'establish_place_cabinet_dark'}
        c,i,p=fixture_v3([task('name',goal('cabinet_dark','establish_place',place_name='Cabinet Spot'))],
            second=[task('back',goal('cabinet_dark','revisit_place',place_name='Cabinet Spot',place_source=source))])
        r=select_scenario(c,p)
        with tempfile.TemporaryDirectory() as tmp:
            agent=Agent();ds,out,_,_=self.run_case(tmp,r,c,i,agent)
            first,second=json.loads((out/'request_audit.json').read_text())['tasks']
            self.assertFalse(first['outcome']['success']);self.assertEqual(second['repair_status'],'unavailable')
            self.assertTrue(second['repair_record']['failure_causes'])
            self.assertEqual(second['source_context'][0]['task_id'],'name')
            self.assertFalse(any(q.get('task_id')=='back' for q in agent.requests))
            self.assertEqual(write_summary(ds,out)['tasks_excluded'],1)

    def test_wrong_room_placement_keeps_actual_geometry_and_repairs_recall(self):
        c,i,p=fixture_v3(second=[task('ask',goal('book_red','answer',query='room_after',history=SOURCE))])
        r=select_scenario(c,p)
        class Placer(Agent):
            def request(self,payload,on_wait=None):
                super().request(payload,on_wait)
                if payload['type']=='session_start':return {}
                if payload.get('task_id')=='move':
                    step=sum(q.get('task_id')=='move' for q in self.requests)
                    return {'action':{'action':'pick' if step==1 else 'place','target':'reading book'},'done':step==2}
                return {'action':'no_op','answer':'study','done':True}
        class WrongRoom(CapturingEnv):
            def step(self,action):
                name=action if isinstance(action,str) else action['action']
                obj=self.state['objects']['book_red']
                if name in ('pick','place'):
                    obj['held']=name=='pick';self.state['held_object']='book_red' if obj['held'] else None
                    if name=='place':obj['geometry']={'world_aabb':{'min':[1.4,1.4,0],'max':[1.6,1.6,.2]}}
                    return self.observe(),0,False,False,{'success':True,'object_id':'book_red'}
                return super().step(action)
        with tempfile.TemporaryDirectory() as tmp:
            _,out,_,_=self.run_case(tmp,r,c,i,Placer(),WrongRoom)
            first,second=json.loads((out/'request_audit.json').read_text())['tasks']
            self.assertFalse(first['outcome']['success'])
            self.assertIn('geometry',first['outcome']['state_changes']['objects'][0]['changes'])
            self.assertEqual(second['runtime_ground_truth']['goals'][0]['expected_outcome']['answer'],'study')
            self.assertTrue(second['outcome']['success'])

    def test_interrupt_and_recorder_error_close_environment_and_invalidate_score(self):
        c,i,p=fixture_v3([task('ask',goal('book_red','answer',query='current_room'))]);r=select_scenario(c,p)
        class Interrupted(Agent):
            def request(self,payload,on_wait=None):
                if payload['type']=='step':raise KeyboardInterrupt()
                return {}
        class BadCamera(CapturingEnv):
            def capture_review_frames(self,layout):raise RecordingError('camera disconnected')
        for agent,env_class,error in ((Interrupted(),CapturingEnv,KeyboardInterrupt),(Agent(),BadCamera,RecordingError)):
            with self.subTest(error=error),tempfile.TemporaryDirectory() as tmp,patch.object(env_class,'close') as close:
                with self.assertRaises(error):
                    self.run_case(tmp,r,c,i,agent,env_class,video_options={'layout':'front','fps':2,'event_seconds':.5})
                close.assert_called_once()
                session=Path(tmp)/'run/scenes/synthetic_demo/sessions/session_000'
                self.assertFalse(json.loads((session/'metrics.json').read_text())['evaluation_valid'])
                manifest=json.loads((session/'video_manifest.json').read_text())
                self.assertEqual(manifest['run_status'],'interrupted')

    def test_unsupported_counts_as_task_failure_not_exclusion(self):
        c,i,p=fixture_v3();r=select_scenario(c,p)
        class Unsupported(Agent):
            def request(self,payload,on_wait=None):
                if payload['type']=='session_start':return {}
                return {'action':'no_op','done':True,'agent_status':'unsupported','reason_code':'simplenav_unsupported_manipulation'}
        with tempfile.TemporaryDirectory() as tmp:
            ds,out,_,_=self.run_case(tmp,r,c,i,Unsupported())
            report=write_summary(ds,out)
            self.assertEqual(report['tasks_evaluated'],1);self.assertEqual(report['tasks_excluded'],0)
            row=json.loads((out/'request_audit.json').read_text())['tasks'][0]
            self.assertEqual(row['outcome']['agent_status'],'unsupported');self.assertFalse(row['outcome']['success'])

    def test_blocked_question_action_does_not_override_correct_answer(self):
        c,i,p=fixture_v3([task('ask',goal('book_red','answer',query='current_room'))]);r=select_scenario(c,p)
        class Forbidden(Agent):
            def request(self,payload,on_wait=None):
                response=super().request(payload,on_wait)
                if payload['type']=='step':response['action']={'action':'pick','target':'reading book'}
                return response
        with tempfile.TemporaryDirectory() as tmp:
            ds,out,env,_=self.run_case(tmp,r,c,i,Forbidden())
            result=json.loads((out/'request_audit.json').read_text())['tasks'][0]['outcome']
            self.assertTrue(result['success']);self.assertEqual(result['reason_code'],'success')
            self.assertIsNone(env.state['held_object']);self.assertEqual(write_summary(ds,out)['task_score'],1)
            event=lines(next(out.glob('scenes/*/sessions/*'))/'action_changelog.jsonl')[0]
            self.assertEqual(event['result']['reason'],'question_action_not_allowed')

    def test_mutation_readback_mismatch_is_logged(self):
        c,i,p=fixture_v3();r=select_scenario(c,p)
        from tests.frozen_data import runtime_cache
        env=CapturingEnv({'objects':{o['object_id']:deepcopy(o) for o in i['metadata']['objects']}})
        env.apply_mutation_v5=lambda prepared:None
        prepared={'status':'ready','object_id':'book_red','destination_room_id':'kitchen',
                  'geometry':{'world_aabb':{'min':[4,1,0],'max':[4.2,1.2,.2]}}}
        with tempfile.TemporaryDirectory() as tmp,patch('benchmark.runtime.request_runner.prepare_mutation_v3',return_value=prepared):
            apply_boundary(env,c,[{'changes':[]}],runtime_cache(i,p),[],Path(tmp),3)
            log=lines(Path(tmp)/'runtime_mutations.jsonl')[0]
            self.assertEqual(log['result']['status'],'not_applied');self.assertFalse(log['result']['confirmed'])
            self.assertEqual(log['state_changes']['objects'],[])


class PreflightTests(unittest.TestCase):
    def test_positive_camera_pitch_points_up_and_negative_pitch_down(self):
        from benchmark.scenes.isaac_env import _camera_quaternion
        for pitch in (.4,-.4):
            w,x,y,z=_camera_quaternion(0,pitch)
            forward_z=2*(x*z-w*y)
            self.assertAlmostEqual(forward_z,float(np.sin(pitch)))

    def test_inspection_uses_completed_renders_when_simulation_time_is_paused(self):
        from benchmark.scenes.request_evidence import observe
        frame={'rendering_frame':{'referenceTimeNumerator':0},
            'instance_id_segmentation':{'data':np.ones((4,4),dtype=np.uint32),'info':{'idToLabels':{'1':'/World/target'}}},
            'rgb':np.zeros((4,4,3),np.uint8)}
        env=SimpleNamespace(_np=np,_prim_paths={'target':'/World/target'},
            camera=SimpleNamespace(add_instance_id_segmentation_to_frame=lambda:None,get_current_frame=lambda:frame),
            observe=lambda:{'images':{}},get_agent_pose=lambda:{'position':[0,0,0]})
        count=[0]
        def render():count[0]+=1;return count[0]
        env.render_once=render
        state={'objects':{'target':{'geometry':{'world_aabb':{'min':[0,0,0],'max':[1,1,1]}}}}}
        with patch('benchmark.scenes.request_evidence.runtime_state',return_value=state):
            _,first=observe(env,[{'object_id':'target'}])
            frame['instance_id_segmentation']['data']=np.zeros((4,4),np.uint32)
            _,second=observe(env,[{'object_id':'target'}])
            self.assertEqual(first[0]['visible_pixels'],16);self.assertEqual(second[0]['visible_pixels'],0)
            self.assertEqual(first[0]['rendering_frame'],second[0]['rendering_frame'])
            self.assertGreater(second[0]['completed_render_sequence'],first[0]['completed_render_sequence'])
            env.render_once=lambda:None
            with self.assertRaisesRegex(RuntimeError,'fresh rendered'):observe(env,[{'object_id':'target'}])

    def test_viewer_status_survives_session_task_reset_and_result_events(self):
        from benchmark.viewer.viewer_overlay import ViewerOverlay
        overlay=ViewerOverlay.__new__(ViewerOverlay)
        overlay._operation_count=overlay._movement_count=overlay._reset_count=0
        for name in ('_title','_body','_session','_status','_reset'):
            setattr(overlay,name,SimpleNamespace(text='',enabled=True))
        overlay.update(event='session_start',instruction='assist',runtime_contract={'name':'contextual_requests'})
        self.assertFalse(overlay._reset.enabled)
        overlay.update(event='task_start',instruction='Go — there.')
        overlay.update(event='reset',reset_count=1)
        overlay.update(event='agent_request',instruction='Go — there.')
        overlay.update(event='task_complete',success=False,instruction='Go — there.')
        self.assertIn('resets 1',overlay._status.text)
        self.assertIn('FAILED',overlay._title.text)

    def test_default_preparation_never_executes_even_when_checks_pass(self):
        from benchmark.cli import main
        from contextlib import redirect_stdout
        from io import StringIO
        with patch('benchmark.launch.simplenav.load_config',return_value={}), \
             patch('benchmark.launch.simplenav.preflight',return_value={'ok':True}), \
             patch('benchmark.launch.experiment.execute') as execute,redirect_stdout(StringIO()):
            self.assertEqual(main(['preflight','--config','unused.json']),0)
            execute.assert_not_called()

    def test_resource_cleanup_uses_camera_destroy_and_attempts_all_resources(self):
        from benchmark.scenes.isaac_env import IsaacVLNEnv
        env=IsaacVLNEnv.__new__(IsaacVLNEnv)
        released=[]
        def broken():released.append('overview');raise RuntimeError('release error')
        env._review_overview_sensor=SimpleNamespace(destroy=broken)
        env._holoagent_sensor_bridge=SimpleNamespace(close=lambda:released.append('bridge'))
        env.camera=SimpleNamespace(destroy=lambda:released.append('front'))
        with self.assertRaisesRegex(RuntimeError,'release error'):env.close()
        self.assertEqual(released,['overview','bridge','front'])
        env.close()

    def test_cpu_preflight_does_not_launch_isaac_or_connect_model(self):
        from benchmark.launch.simplenav import ROOT
        example=ROOT/'configs/simplenav/review.example.json'
        config=load_config(example)
        cmd=commands(config)
        self.assertIn('--record-video',cmd['benchmark_argv']);self.assertIn('--viewer',cmd['benchmark_argv'])
        self.assertIn('--simplenav-root',cmd['agent_argv'])
        invoked=[]
        def subprocess_run(argv,**kwargs):
            invoked.append(argv)
            if argv[0]==config['isaac_python']:raise AssertionError('must not execute Isaac')
            if '-encoders' in argv:return SimpleNamespace(stdout='libx264')
            self.assertNotIn('import torch',argv[-1]);self.assertNotIn('import isaacsim',argv[-1])
            return SimpleNamespace(stdout=json.dumps({'numpy':True,'PIL':True}))
        with patch('benchmark.launch.simplenav.subprocess.run',side_effect=subprocess_run), \
             patch('benchmark.scenes.dataset_loader.BenchmarkDataset',side_effect=ValueError('fixture avoids full archive')):
            report=preflight(config)
        self.assertFalse(report['ok'])
        self.assertIn('model_connection',report['not_executed'])
        self.assertTrue(all('benchmark.run' not in argv for argv in invoked))

    def test_report_escapes_html_and_detects_truncated_logs(self):
        c,i,p=fixture_v3();r=select_scenario(c,p)
        with tempfile.TemporaryDirectory() as tmp:
            _,out,_,_=V5TraceTests().run_case(tmp,r,c,i,Agent())
            session=out/'scenes/synthetic_demo/sessions/session_000'
            with (session/'runtime_events.jsonl').open('a') as stream:
                stream.write(json.dumps({'event':'integration_error','error':'<script>evil()</script>'})+'\n{"truncated":')
            report=write_request_audit(out)
            self.assertEqual(len(report['integration_errors']),2)
            self.assertNotIn('<script>',(out/'request_audit.html').read_text())
            self.assertIn('&lt;script&gt;',(out/'request_audit.html').read_text())
            for name in ('request_outcomes.jsonl','video_index.jsonl','runtime_mutations.jsonl'):
                with (session/name).open('a') as stream:stream.write('[]\n')
            self.assertEqual(len(write_request_audit(out)['integration_errors']),5)


if __name__=='__main__':unittest.main()
