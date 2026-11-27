from roboodyssey.artifacts.layout import session_file
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from tests.frozen_data import fixture_v3, goal, task, SOURCE
from tests.frozen_data import select_scenario
from tests.frozen_data import write_dataset
from roboodyssey.scenes.dataset_loader import BenchmarkDataset
from roboodyssey.runtime.session_runner import SessionRunner
from roboodyssey.evaluation.requests import evaluate_request,request_totals
from roboodyssey.evaluation.report import write_summary
from roboodyssey.runtime.request_runner import public_observation,resolve_action


class Env:
    def __init__(self,state,*,visible=True):
        self.state=deepcopy(state);self.state['held_object']=None
        self.pose={'position':[1.1,1.1,0]};self.visible=visible;self.observation_frame_id=0
        self.closed=False
    def runtime_state_v5(self):return deepcopy(self.state)
    def get_agent_pose(self):return deepcopy(self.pose)
    def set_task(self,t):self.task=t
    def observe(self):return {'images':{}}
    def evaluation_navigation(self):
        import numpy as np
        # This protocol fixture explicitly represents an unobstructed world.
        return {'blocked':np.zeros((40,60),bool),'origin':[0,0],'resolution':.1,'radius':0}
    def observe_with_evidence(self,requirements):
        self.observation_frame_id+=1
        return self.observe(),[{'object_id':r['object_id'],'frame_id':self.observation_frame_id,'current_frame':True,
            'visible_pixels':20 if self.visible else 0,'distance_m':.1,'pose':deepcopy(self.pose)} for r in requirements]
    def step(self,action):return self.observe(),0,False,False,{'success':True}
    def close(self):self.closed=True


class Agent:
    def __init__(self):self.requests=[]
    def send_observation(self,payload):self.requests.append(deepcopy(payload))
    def request(self,payload,on_wait=None):
        self.requests.append(deepcopy(payload))
        if payload['type']=='session_start':
            context=payload.get('sequence_context',{})
            return {'sequence_reset_ack':context['run_id']} if context.get('reset_memory') else {}
        return {'action':'no_op','answer':'study','done':True}


class RequestRunnerTests(unittest.TestCase):
    def test_navigation_partial_metric_is_saved_and_recomputed_from_actual_runtime_gt(self):
        import numpy as np
        from roboodyssey.evaluation.task_diagnostics import analyze_run
        c,i,p=fixture_v3([task('visit',goal('book_red'))])
        compiled=select_scenario(c,p)
        class WalkingEnv(Env):
            def __init__(self,state):
                super().__init__(state);self.pose={'position':[4.1,1.1,0]}
            def evaluation_navigation(self):
                return {'blocked':np.zeros((40,60),bool),'origin':[0,0],'resolution':.1,'radius':0}
            def step(self,action):
                self.pose['position'][0]-=1
                return self.observe(),0,False,False,{'success':True,'collision':False}
        class Walker(Agent):
            def request(self,payload,on_wait=None):
                self.requests.append(deepcopy(payload))
                return {'sequence_reset_ack':payload['sequence_context']['run_id']} if payload['type']=='session_start' else {'action':'move_forward','done':True}
        with tempfile.TemporaryDirectory() as tmp:
            archive,output=Path(tmp)/'archive',Path(tmp)/'run'
            write_dataset(archive,compiled,c,i);ds=BenchmarkDataset(archive)
            state=ds.canonical_state('synthetic_demo');env=WalkingEnv(state)
            SessionRunner(output,Walker(),max_steps=2).run_scene('synthetic_demo',
                ds.sessions('synthetic_demo'),state,lambda *a:env)
            summary=write_summary(ds,output)
            self.assertEqual(summary['task_success_rate'],0)
            self.assertGreater(summary['score_100'],0)
            self.assertLess(summary['score_100'],100)
            self.assertTrue(summary['metrics_complete'])
            (output/'run_config.json').write_text(json.dumps({'official':False}))
            report=analyze_run(output)
            self.assertEqual(report['issues'],[])
            self.assertAlmostEqual(report['summary']['formal_rule_average'],summary['score_100'])
            self.assertEqual(report['tasks'][0]['evaluation']['measurements']['steps'],1)
            # Emulate a historical run that predates live metric snapshots.
            from roboodyssey.artifacts.layout import sha256_file
            from roboodyssey.evaluation.rescore import ReconstructedReference
            from roboodyssey.contracts.doors import POLICY
            from roboodyssey.contracts.raster import derive_navigation
            setup={'policy':POLICY,'doors':[]}
            frozen = ds._requests
            sequence_root = ds.root
            manifest=json.loads((sequence_root/'dataset_manifest.json').read_text())
            metadata=json.loads(frozen.path('scene_metainfo').read_text())
            cache=json.loads(frozen.path('runtime_scene_cache').read_text())
            nav_path=frozen.path('navigation')
            with np.load(nav_path,allow_pickle=False) as nav:
                resolution=float(nav['resolution']);radius=float(nav['robot_radius'])
            objects=[{**o,**cache['objects'][o['object_id']]['initial']} for o in metadata['objects']]
            nav=derive_navigation(metadata['rooms'],objects,
                resolution=resolution,robot_radius=radius,environment_setup=setup)
            nav.pop('environment_setup');nav['environment_setup_json']=json.dumps(setup)
            np.savez_compressed(nav_path,**{k:v for k,v in nav.items() if v is not None})
            setup_path=frozen.path('environment_setup')
            setup_path.write_text(json.dumps(setup))
            revision=frozen.version
            version_manifest=revision.manifest
            for file in (nav_path,setup_path):
                version_manifest['files'][file.relative_to(revision.directory).as_posix()]=sha256_file(file)
            version_path=revision.directory/'scene_version.json'
            version_path.write_text(json.dumps(version_manifest))
            reference={'revision':revision.revision,'manifest_sha256':sha256_file(version_path)}
            manifest['scene_reference']=reference
            plan_path=frozen.path('plan')
            plan=json.loads(plan_path.read_text());plan['scene_reference']=reference
            plan_path.write_text(json.dumps(plan))
            manifest['files'][plan_path.relative_to(sequence_root).as_posix()]=sha256_file(plan_path)
            manifest_path=sequence_root/'dataset_manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            index_path=archive/'scene_manifest.json'
            index=json.loads(index_path.read_text())
            index['versions'][revision.revision]['sha256']=reference['manifest_sha256']
            index['sequences'][manifest['sequence_id']]['sha256']=sha256_file(manifest_path)
            index_path.write_text(json.dumps(index))
            config={'official':False,'dataset_manifest_sha256':sha256_file(manifest_path),
                'sequence_identity':ds.sequence_identity(),'motion':{'robot_radius_m':radius}}
            (output/'run_config.json').write_text(json.dumps(config))
            session=next(output.glob('sequences/*/sessions/*'))
            for name in ('runtime_tasks.jsonl','request_outcomes.jsonl'):
                rows=[json.loads(l) for l in session_file(session, name).read_text().splitlines()]
                for row in rows:
                    row.pop('evaluation',None);row.pop('metric_reference',None);row.pop('task_timing',None)
                session_file(session, name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
            # This fixture emulates a run without any original collision
            # snapshot evidence. Do not retain fingerprints of the synthetic
            # open-grid environment when reconstructing its archive geometry.
            changelog = session_file(session, 'action_changelog.jsonl')
            action_rows = [json.loads(line) for line in changelog.read_text().splitlines()]
            for row in action_rows:
                row.pop('navigation_collision_before_sha256', None)
                row.pop('navigation_collision_after_sha256', None)
            changelog.write_text(''.join(json.dumps(row)+'\n' for row in action_rows))
            metrics=json.loads((session/'metrics.json').read_text())
            for row in metrics['task_results']:row.pop('evaluation',None);row.pop('task_timing',None)
            (session/'metrics.json').write_text(json.dumps(metrics))
            sources={p:p.read_bytes() for p in output.rglob('*') if p.is_file()}
            rescored=analyze_run(output,rescore=True,reference_dataset=archive,reference_output=Path(tmp)/'rescored')
            self.assertEqual(rescored['issues'],[])
            self.assertGreater(rescored['summary']['formal_rule_average'],0)
            self.assertIsNone(rescored['tasks'][0]['original_task_score'])
            self.assertTrue(rescored['reference_reconstruction']['baseline_grid_exact_match'])
            self.assertFalse(rescored['reference_reconstruction']['original_live_snapshot'])
            for file,content in sources.items():self.assertEqual(file.read_bytes(),content)
            with self.assertRaisesRegex(ValueError,'manifest'):
                ReconstructedReference(archive,{**config,'dataset_manifest_sha256':'wrong'},Path(tmp)/'bad')
            nav_path.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'checksum'):
                ReconstructedReference(archive,config,Path(tmp)/'bad')

    def test_stream_continues_during_inference_without_inspection_scoring(self):
        from tests.test_observations import ManualTime
        timer = ManualTime()
        c, i, p = fixture_v3([task('ask', goal('book_red', 'answer', query='current_room'))])
        compiled = select_scenario(c, p)
        with tempfile.TemporaryDirectory() as tmp:
            archive, output = Path(tmp)/'archive', Path(tmp)/'run'
            write_dataset(archive, compiled, c, i)
            dataset = BenchmarkDataset(archive)
            state = dataset.canonical_state('synthetic_demo')
            env = Env(state, visible=False)
            class SlowAgent(Agent):
                def request(self, payload, on_wait=None):
                    self.requests.append(deepcopy(payload))
                    if payload['type'] == 'session_start':
                        return {'sequence_reset_ack':payload['sequence_context']['run_id']}
                    env.visible = True
                    for _ in range(2):
                        timer.sleep(.5)
                        on_wait()
                    return {'action': 'no_op', 'answer': 'study', 'done': True}
            agent = SlowAgent()
            runner = SessionRunner(output, agent, max_steps=2, observation_clock=timer.sampler())
            runner.run_scene('synthetic_demo', dataset.sessions('synthetic_demo'), state, lambda *a: env)
            self.assertEqual(runner.observations.frame_id, 3)
            self.assertEqual(env.observation_frame_id, 0)  # No instance evidence backend needed.
            dispatched = [r for r in agent.requests if r['type'] == 'step']
            self.assertEqual(len(dispatched), 1)
            notifications = [r for r in agent.requests if r['type'] == 'observation']
            self.assertEqual(len(notifications), 2)
            for notification in notifications:
                self.assertNotIn('visible_pixels', json.dumps(notification))
                self.assertNotIn('book_red', json.dumps(notification))
            self.assertEqual(dispatched[0]['observation']['timing']['frame_id'], 1)
            self.assertEqual(write_summary(dataset, output)['task_score'], 1)
            self.assertIsNone(runner.observations.source)
            self.assertTrue(env.closed)

    def test_unsupported_first_request_does_not_repeat_name_definition(self):
        c,i,p=fixture_v3(second=[task('ask',goal('book_red','answer',query='current_room',history=SOURCE))])
        result=select_scenario(c,p)
        class UnsupportedAgent(Agent):
            def request(self,payload,on_wait=None):
                self.requests.append(deepcopy(payload))
                if payload['type']=='session_start':return {'sequence_reset_ack':payload['sequence_context']['run_id']}
                return {'agent_status':'unsupported','reason_code':'unsupported_capability','done':True}
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'archive';output=Path(tmp)/'run';write_dataset(archive,result,c,i)
            ds=BenchmarkDataset(archive);agent=UnsupportedAgent();env=Env(ds.canonical_state('synthetic_demo'))
            SessionRunner(output,agent,max_steps=1).run_scene('synthetic_demo',ds.sessions('synthetic_demo'),
                ds.canonical_state('synthetic_demo'),lambda *a:env)
            requests=[r for r in agent.requests if r['type']=='step']
            self.assertEqual(len(requests),2)
            self.assertTrue(requests[0]['name_introductions'])
            self.assertEqual(requests[1]['name_introductions'],[])
            self.assertEqual(requests[1]['instruction'],result['sessions'][1]['public']['tasks'][0]['instruction'])
            report=write_summary(ds,output)
            self.assertEqual(report['tasks_evaluated'],2)
            self.assertEqual(report['task_score'],0)

    def test_actual_environment_observation_cannot_leak_private_affordances(self):
        cache={'names':{'objects':{'book_red':{'en':'reading book'}}}}
        ledger={'objects/book_red':{'task_id':'introduced'}}
        observation={'images':{},'pose':{'position':[0,0,0]},'held_object':'book_red','room_id':'private_room',
                     'nearby_manipulable_objects':[{'object_id':'book_red','room_id':'private_room'}],
                     'nearby_doors':[{'object_id':'private_door'}],
                     'action_feedback':{'success':True,'reason':'picked','object_id':'book_red','room_id':'private_room'}}
        public=public_observation(observation,cache,ledger)
        self.assertNotIn('book_red',json.dumps(public));self.assertNotIn('private_',json.dumps(public))
        self.assertEqual(public['held_object_name'],'reading book')
        self.assertEqual(resolve_action({'action':'pick','target':'reading book'},cache,ledger)['object_id'],'book_red')
        self.assertIsNone(resolve_action({'action':'pick','target':'reading book'},cache,{}))
        self.assertIsNone(resolve_action({'action':'pick','object_id':'book_red'},cache,ledger))

    def test_correct_answer_finishes_immediately_without_waiting_for_inspection(self):
        c,i,p=fixture_v3([task('ask',goal('book_red','answer',query='current_room'))]);r=select_scenario(c,p)
        class EarlyAgent(Agent):
            def request(self,payload,on_wait=None):
                self.requests.append(deepcopy(payload))
                if payload['type']=='session_start':return {'sequence_reset_ack':payload['sequence_context']['run_id']}
                return {'action':'no_op','answer':'study'} if len(self.requests)==2 else {'action':'no_op','done':True}
        class LateVisibility(Env):
            def observe_with_evidence(self,requirements):
                self.visible=self.observation_frame_id>0
                return super().observe_with_evidence(requirements)
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'archive';output=Path(tmp)/'run';write_dataset(archive,r,c,i)
            ds=BenchmarkDataset(archive);env=LateVisibility(ds.canonical_state('synthetic_demo'))
            agent=EarlyAgent()
            SessionRunner(output,agent,max_steps=2).run_scene('synthetic_demo',ds.sessions('synthetic_demo'),ds.canonical_state('synthetic_demo'),lambda *a:env)
            report=write_summary(ds,output)
            self.assertEqual(report['task_score'],1)
            self.assertEqual(report['task_success_rate'],1)
            self.assertEqual(sum(q['type']=='step' for q in agent.requests),1)

    def test_failed_move_actual_history_then_question_and_public_boundary(self):
        c,i,p=fixture_v3(second=[task('ask',goal('book_red','answer',query='room_after',history=SOURCE))])
        result=select_scenario(c,p)
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'archive';output=Path(tmp)/'run';write_dataset(archive,result,c,i)
            class PickingAgent(Agent):
                def request(self,payload,on_wait=None):
                    response=super().request(payload,on_wait)
                    if payload['type']=='step' and payload['task_id']=='move':
                        return {'action':{'action':'pick','target':'reading book'},'done':True}
                    return response
            class PickedEnv(Env):
                def step(self,action):
                    if isinstance(action,dict) and action.get('action')=='pick':
                        self.state['held_object']=action['object_id']
                        self.state['objects'][action['object_id']]['held']=True
                        return self.observe(),0,False,False,{'success':True,'object_id':action['object_id']}
                    return super().step(action)
            dataset=BenchmarkDataset(archive);agent=PickingAgent();env=PickedEnv(dataset.canonical_state('synthetic_demo'))
            SessionRunner(output,agent,max_steps=2).run_scene('synthetic_demo',dataset.sessions('synthetic_demo'),dataset.canonical_state('synthetic_demo'),lambda *a:env)
            report=write_summary(dataset,output)
            self.assertIsNone(report['task_score'])  # This fixture has no transfer cost reference.
            self.assertFalse(report['metrics_complete'])
            self.assertEqual(report['task_success_rate'],.5)
            self.assertEqual(report['tasks_evaluated'],2)
            self.assertTrue(env.closed)
            for payload in agent.requests:
                serialized=json.dumps(payload)
                for private in ('book_red','inspection_contract','ground_truth','template_id','visible_pixels'):
                    self.assertNotIn(private,serialized)
            events=[json.loads(s) for s in (output/'sequences/seq_fixture/sessions/session_000/traces/runtime_history.jsonl').read_text().splitlines()]
            self.assertFalse(events[0]['success']);self.assertTrue(events[0]['dispatched'])

    def test_correct_answer_scores_with_or_without_visibility(self):
        c,i,p=fixture_v3([task('ask',goal('book_red','answer',query='current_room'))]);r=select_scenario(c,p)
        for visible in (True,False):
            with self.subTest(visible=visible),tempfile.TemporaryDirectory() as tmp:
                archive=Path(tmp)/'archive';output=Path(tmp)/'run';write_dataset(archive,r,c,i)
                ds=BenchmarkDataset(archive);env=Env(ds.canonical_state('synthetic_demo'),visible=visible)
                SessionRunner(output,Agent(),max_steps=1).run_scene('synthetic_demo',ds.sessions('synthetic_demo'),ds.canonical_state('synthetic_demo'),lambda *a:env)
                self.assertEqual(write_summary(ds,output)['task_score'],1.0)

    def test_unused_visibility_backend_cannot_invalidate_answer_only_scoring(self):
        c,i,p=fixture_v3([task('ask',goal('book_red','answer',query='current_room'))]);r=select_scenario(c,p)
        class Broken(Env):
            def observe_with_evidence(self,requirements):raise RuntimeError('camera unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'archive';output=Path(tmp)/'run';write_dataset(archive,r,c,i)
            ds=BenchmarkDataset(archive);env=Broken(ds.canonical_state('synthetic_demo'))
            SessionRunner(output,Agent()).run_scene('synthetic_demo',ds.sessions('synthetic_demo'),ds.canonical_state('synthetic_demo'),lambda *a:env)
            summary=write_summary(ds,output)
            self.assertEqual(summary['task_score'],1);self.assertTrue(summary['evaluation_valid'])
            self.assertEqual(summary['tasks_excluded'],0)

    def test_stale_observation_boolean_polarity_and_order(self):
        c,i,p=fixture_v3(second=[task('ask',goal('book_red','answer',query='has_changed',history=SOURCE))]);r=select_scenario(c,p)
        truth=r['sessions'][1]['ground_truth'][0]
        state={'objects':{o['object_id']:deepcopy(o) for o in i['metadata']['objects']}}
        evidence=[{'object_id':'book_red','frame_id':4,'current_frame':True,'visible_pixels':20,'distance_m':.1}]
        expected=truth['goals'][0]['expected_outcome']['answer']
        answer='yes' if expected else 'no'
        self.assertTrue(evaluate_request(truth,state,[],[],evidence,answer)['success'])
        self.assertTrue(evaluate_request(truth,state,[],[],evidence,answer)['success'])
        self.assertFalse(evaluate_request(truth,state,[],[],evidence,'yes and no')['success'])
        c,i,p=fixture_v3([task('nav',goal('book_red'),goal('cabinet_dark'))]);truth=select_scenario(c,p)['sessions'][0]['ground_truth'][0]
        poses=[{'position':[.6,2.6,0]},{'position':[1.1,1.1,0]}]
        import numpy as np
        from roboodyssey.evaluation.navigation import GridDistance
        navigation = GridDistance(np.zeros((40, 60), bool), [0, 0], .1)
        self.assertFalse(evaluate_request(truth,state,poses,[],[],None,navigation=navigation)['success'])
        self.assertTrue(evaluate_request(truth,state,list(reversed(poses)),[],[],None,navigation=navigation)['success'])

    def test_wrong_or_missing_answer_never_earns_observation_credit(self):
        c,i,p=fixture_v3([task('ask',goal('book_red','answer',query='current_room'))])
        gt=select_scenario(c,p)['sessions'][0]['ground_truth'][0]
        state={'objects':{o['object_id']:deepcopy(o) for o in i['metadata']['objects']}}
        evidence=[{'object_id':'book_red','frame_id':4,'current_frame':True,'visible_pixels':200,'distance_m':.1}]
        for answer in (None,'kitchen','study or kitchen','private_study'):
            self.assertFalse(evaluate_request(gt,state,[],[],evidence,answer)['success'])
        for answer in ('study','In the study.','it is in the study'):
            self.assertTrue(evaluate_request(gt,state,[],[],[],answer)['success'])

    def test_transfer_requires_correct_region_floor_release_and_current_place(self):
        c,i,p=fixture_v3()
        gt=select_scenario(c,p)['sessions'][0]['ground_truth'][0]
        state={'objects':{o['object_id']:deepcopy(o) for o in i['metadata']['objects']},
               'held_object':None,'floor_heights':{'kitchen':0}}
        state['objects']['book_red'].update(room_id='kitchen',held=False,
            geometry={'world_aabb':{'min':[4,1,0],'max':[4.2,1.2,.2]}})
        place={'action':'place','result':{'success':True,'object_id':'book_red'}}
        self.assertTrue(evaluate_request(gt,state,[],[place],[],None)['success'])
        self.assertFalse(evaluate_request(gt,state,[],[],[],None)['success'])
        for changes in ({'held':True},{'room_id':'study'},
                        {'geometry':{'world_aabb':{'min':[4,1,.1],'max':[4.2,1.2,.3]}}}):
            invalid=deepcopy(state);invalid['objects']['book_red'].update(changes)
            self.assertFalse(evaluate_request(gt,invalid,[],[place],[],None)['success'])

    def test_unavailable_must_have_a_failed_source(self):
        with self.assertRaises(ValueError):request_totals([{'scored':False,'reason_code':'unavailable'}])
