"""CPU contract tests for partial credit, route costs and immutable references."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

from roboodyssey.evaluation.navigation import GridDistance, OrderedRoute
from roboodyssey.evaluation.task_metrics import (
    METRIC_VERSION, LEGACY_METRIC_VERSION, BINARY_TRANSFER_METRIC_VERSION,
    evaluate_metrics, navigation_metric, metric_totals, prepare_reference, load_reference,
)


def pose(x, y=2.5):
    return {'position': [x,y,1.5]}


def truth(*xs):
    goals=[{'goal_id':f'g{i}', 'goal_kind':'visit', 'target':{'object_ids':[f'o{i}']},
        'expected_outcome':{'target_geometry':{'world_aabb_xy':{'min':[x,2.5],'max':[x,2.5]}}},
        'success_criteria':{'position_tolerance_m':0}} for i,x in enumerate(xs)]
    return {'task_id':'t','state_assumption':'actual_predispatch_state','goals':goals,
        'success_criteria':{'goal_order':[g['goal_id'] for g in goals]}}


def trajectory(*xs):
    return [{'action':'move_forward','task_step':i,'before':pose(a),'after':pose(b),'result':{'success':True}}
            for i,(a,b) in enumerate(zip(xs,xs[1:]))]


def nav_score(grid, gt, *xs):
    actions=trajectory(*xs)
    poses=[a[k] for a in actions for k in ('before','after')] or [pose(xs[0])]
    return navigation_metric(gt,grid,poses,actions)


class PathMetricTests(unittest.TestCase):
    def setUp(self):
        self.grid=GridDistance(np.zeros((6,30),bool),[0,0],1,0)

    def test_failure_progress_is_discounted_by_detours_and_success_is_not_always_100(self):
        gt=truth(15.5)
        cases=[((5.5,),0), ((5.5,10.5),50), ((5.5,.5,10.5),25),
               ((5.5,15.5),100), ((5.5,.5,15.5),50)]
        for points,expected in cases:
            self.assertAlmostEqual(nav_score(self.grid,gt,*points)['score'],expected)

    def test_getting_close_then_leaving_uses_final_progress(self):
        result=nav_score(self.grid,truth(15.5),5.5,14.5,7.5)
        self.assertLess(result['score'],result['best_score'])
        self.assertEqual(result['best_progress_score'],90)
        self.assertAlmostEqual(result['progress_score'],20)

    def test_geodesic_does_not_cross_walls_or_snap_to_another_component(self):
        blocked=np.zeros((7,20),bool);blocked[:6,10]=True
        grid=GridDistance(blocked,[0,0],1,0)
        route=OrderedRoute(grid,truth(15.5)['goals'])
        self.assertGreater(route.distance([5.5,2.5],0),10)
        blocked[6,10]=True
        sealed=GridDistance(blocked,[0,0],1,0)
        self.assertTrue(np.isinf(OrderedRoute(sealed,truth(15.5)['goals']).distance([5.5,2.5],0)))
        self.assertEqual(sealed.connectors([10.5,2.5]),[])
        with self.assertRaisesRegex(ValueError,'crosses'):
            nav_score(grid,truth(15.5),5.5,15.5)

    def test_radius_and_diagonal_corner_rules_are_enforced(self):
        blocked=np.zeros((3,3),bool);blocked[0,1]=blocked[1,0]=True
        grid=GridDistance(blocked,[0,0],1,0)
        self.assertTrue(np.isinf(grid.between([.5,.5],[1.5,1.5])))
        grid=GridDistance(np.zeros((3,3),bool),[0,0],1,.6)
        self.assertFalse(grid.walkable([.5,.5]))
        self.assertTrue(grid.walkable([1.5,1.5]))

    def test_future_goal_passes_do_not_complete_goals_out_of_order(self):
        result=nav_score(self.grid,truth(15.5,7.5),5.5,7.5,15.5,14.5)
        self.assertEqual(result['goals_completed'],1)
        self.assertLess(result['progress_score'],100)
        self.assertGreater(result['progress_score'],50)
        self.assertEqual(result['goal_hits'],[3])
        same=nav_score(self.grid,truth(15.5,15.5),15.5)
        self.assertEqual(same['progress_score'],50)

    def test_shortest_multi_region_route_is_not_greedy_nearest_endpoint(self):
        gt=truth(10.5,5.5)
        gt['goals'][0]['expected_outcome']['target_geometry']['world_aabb_xy']={'min':[5.5,1.5],'max':[10.5,3.5]}
        route=OrderedRoute(self.grid,gt['goals'])
        self.assertAlmostEqual(route.distance([15.5,2.5],0,remaining_route=True),10)

    def test_reset_is_charged_and_zero_displacement_resets_are_not_free(self):
        actions=trajectory(5.5,5.5,15.5)
        actions[0].update(action='reset',result={'success':True,'reset_applied':True})
        result=navigation_metric(truth(15.5),self.grid,[a[k] for a in actions for k in ('before','after')],actions)
        self.assertEqual(result['walking_length_m'],10)
        self.assertEqual(result['reset_cost_m'],1)
        self.assertAlmostEqual(result['score'],1000/11)

    def test_reference_zero_and_unreachable_cases_are_explicit(self):
        self.assertEqual(nav_score(self.grid,truth(5.5),5.5)['score'],100)
        blocked=np.zeros((6,30),bool);blocked[:,10]=True
        with self.assertRaisesRegex(ValueError,'unreachable'):
            nav_score(GridDistance(blocked,[0,0],1,0),truth(15.5),5.5)

    def test_reference_is_reproducible_and_checksum_guards_tampering(self):
        snapshot={'blocked':np.zeros((6,30),bool),'origin':[0,0],'resolution':1,'radius':0}
        env=SimpleNamespace(evaluation_navigation=lambda:snapshot)
        with tempfile.TemporaryDirectory() as tmp:
            reference,grid=prepare_reference(env,truth(15.5),tmp)
            loaded=load_reference(reference,tmp)
            self.assertEqual(nav_score(loaded,truth(15.5),5.5,10.5)['score'],50)
            (Path(tmp)/reference['file']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_reference(reference,tmp)

    def test_environment_snapshot_matches_open_door_collision_rules_without_isaac(self):
        from roboodyssey.scenes.isaac_env import IsaacVLNEnv
        env=IsaacVLNEnv.__new__(IsaacVLNEnv)
        env._static_base=np.zeros((6,8),bool);env._static=env._static_base.copy()
        env._dynamic=np.zeros((6,8),bool);env._dynamic[:,4]=True
        env._static_base[0,4]=True
        env._door_cells={'door':{(4,y) for y in range(6)}}
        env._door_open={'door'};env._origin=np.array([0,0]);env._resolution=1
        env._np=np;env._held_object=None
        env.config=SimpleNamespace(robot_radius=.1)
        original=env._dynamic.copy()
        grid=GridDistance(**env.evaluation_navigation())
        for y in range(6):
            for x in range(8):
                point=[x+.5,y+.5]
                self.assertEqual(grid.walkable(point),env._walkable(point))
        self.assertFalse(grid.blocked[2,4]);self.assertTrue(grid.blocked[0,4])
        np.testing.assert_array_equal(env._dynamic,original)
        # Navigation can follow a failed transfer while still carrying an item.
        env._door_open=set();env._held_object='box'
        env._objects={'box':{'geometry':{'world_aabb':{'min':[0,0,0],'max':[2,2,1]}}}}
        grid=GridDistance(**env.evaluation_navigation())
        for point in ([3.5,2.5],[2.5,2.5],[1.5,2.5]):
            self.assertEqual(grid.walkable(point),env._walkable(point) and env._carried_door_clear(point))
        self.assertTrue(env._walkable([3.5,2.5]));self.assertFalse(grid.walkable([3.5,2.5]))
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'grid.npz';grid.save(file)
            self.assertFalse(GridDistance.load(file).walkable([3.5,2.5]))


class SemanticMetricTests(unittest.TestCase):
    def base(self, kind='transfer'):
        goal={'goal_id':'g','goal_kind':kind,'target':{'object_ids':['book']},
              'success_criteria':{'room_id':'study'}}
        before={'objects':{'book':{'room_id':'hall','held':False}},'held_object':None}
        result={'task_id':'t','scored':True,'success':False,
                'goal_results':[{'goal_id':'g','success':False}]}
        return {'task_id':'t','goals':[goal]},before,result

    def test_v2_binary_transfer_scores_remain_reproducible(self):
        gt,before,result=self.base()
        pick={'action':'pick','result':{'success':True,'object_id':'book'}}
        self.assertEqual(evaluate_metrics(gt,before,before,[],[],result,metric_version=BINARY_TRANSFER_METRIC_VERSION)['score'],0)
        self.assertEqual(evaluate_metrics(gt,before,before,[],[pick],result,metric_version=BINARY_TRANSFER_METRIC_VERSION)['score'],0)
        carried=deepcopy(before);carried['objects']['book'].update(room_id='study',held=True)
        partial=evaluate_metrics(gt,before,carried,[],[pick],result,metric_version=BINARY_TRANSFER_METRIC_VERSION)
        self.assertEqual(partial['score'],0)
        self.assertEqual(partial['goal_scores'][0]['components'],
            {'acquired':True,'at_destination':True,'valid_place_and_release':False})
        # Location alone without controlling the object must not earn delivery credit.
        self.assertEqual(evaluate_metrics(gt,before,carried,[],[],result,metric_version=BINARY_TRANSFER_METRIC_VERSION)['score'],0)
        result['success']=result['goal_results'][0]['success']=True
        self.assertEqual(evaluate_metrics(gt,before,carried,[],[pick],result,metric_version=BINARY_TRANSFER_METRIC_VERSION)['score'],100)

    def test_qa_only_answer_correctness_scores_regardless_of_inspection_or_old_success(self):
        gt,before,result=self.base('answer');gt['goals'][0]['inspection_contract']={'object_id':'book'}
        result['goal_results'][0].update(observation_valid=False,answer_correct=True)
        self.assertEqual(evaluate_metrics(gt,before,before,[],[],result)['score'],100)
        result['goal_results'][0].update(observation_valid=True,answer_correct=False)
        self.assertEqual(evaluate_metrics(gt,before,before,[],[],result)['score'],0)
        result['goal_results'][0]['answer_correct']=True
        self.assertEqual(evaluate_metrics(gt,before,before,[],[],result)['score'],100)

    def test_unsupported_and_excluded_and_old_action_rejection(self):
        gt,before,result=self.base('answer')
        result['goal_results'][0].update(success=True,answer_correct=True)
        self.assertEqual(evaluate_metrics(gt,before,before,[],[],result|{'agent_status':'unsupported'})['score'],0)
        self.assertEqual(evaluate_metrics(gt,before,before,[],[],result|{'reason_code':'forbidden_question_action'})['score'],100)
        excluded=evaluate_metrics(None,before,before,[],[],{'scored':False})
        self.assertIsNone(excluded['score'])
        self.assertEqual(excluded['status'],'excluded')

    def test_legacy_semantic_scores_are_reproducible_and_versions_never_mix(self):
        gt,before,result=self.base()
        pick={'action':'pick','result':{'success':True,'object_id':'book'}}
        old=evaluate_metrics(gt,before,before,[],[pick],result,metric_version=LEGACY_METRIC_VERSION)
        new=evaluate_metrics(gt,before,before,[],[pick],result,metric_version=BINARY_TRANSFER_METRIC_VERSION)
        self.assertAlmostEqual(old['score'],100/3)
        self.assertEqual(new['score'],0)
        self.assertAlmostEqual(metric_totals([result|{'evaluation':old}])['score_100'],100/3)
        mixed=metric_totals([result|{'evaluation':old},result|{'evaluation':new}])
        self.assertIsNone(mixed['score_100'])
        self.assertFalse(mixed['metrics_complete'])
        gt,before,result=self.base('answer')
        gt['goals'][0]['inspection_contract']={'object_id':'book'}
        result['goal_results'][0].update(observation_valid=True,answer_correct=False)
        self.assertEqual(evaluate_metrics(gt,before,before,[],[],result,metric_version=LEGACY_METRIC_VERSION)['score'],50)
        result['reason_code']='forbidden_question_action'
        self.assertEqual(evaluate_metrics(gt,before,before,[],[],result,metric_version=LEGACY_METRIC_VERSION)['score'],0)

    def test_qa_final_prediction_is_authoritative_and_missing_evidence_is_not_zero(self):
        gt,before,result=self.base('answer')
        gt['goals'][0]['expected_outcome']={'answer_type':'room','accepted_answers':['study']}
        result['goal_results'][0]['answer_correct']=True
        for answer,expected in [('in the study',100),('kitchen',0),(None,0)]:
            self.assertEqual(evaluate_metrics(gt,before,before,[],[],result|{'prediction':answer})['score'],expected)
        result['goal_results'][0].pop('answer_correct')
        metric=evaluate_metrics(gt,before,before,[],[],result)
        self.assertIsNone(metric['score'])
        self.assertEqual(metric['reason'],'answer_correctness_evidence_missing')

    def test_mixed_required_goals_are_averaged_without_milestone_weights(self):
        gt,before,result=self.base()
        gt['goals'].append({'goal_id':'a','goal_kind':'answer','target':{'object_ids':['book']}})
        result['goal_results'].append({'goal_id':'a','success':True,'answer_correct':True})
        self.assertEqual(evaluate_metrics(gt,before,before,[],[],result,metric_version=BINARY_TRANSFER_METRIC_VERSION)['score'],50)
        self.assertIsNone(evaluate_metrics(gt,before,before,[],[],result)['score'])

    def test_missing_reference_preserves_denominator_and_timing_does_not_invent_costs(self):
        result={'task_id':'t','goal_results':[{'goal_id':'g0','success':False}]}
        metric=evaluate_metrics(truth(15.5),{}, {},[pose(5.5)],[],result)
        self.assertEqual(metric['status'],'unavailable')
        self.assertIsNone(metric['score'])
        totals=metric_totals([result|{'evaluation':metric}])
        self.assertIsNone(totals['score_100'])
        self.assertEqual(totals['tasks_missing_metric'],['t'])
        self.assertIsNone(metric['step_efficiency_score'])
        self.assertIsNone(metric['time_efficiency_score'])


class ReproducibilityTests(unittest.TestCase):
    def test_old_qa_run_keeps_v1_score_until_explicit_rescore_and_preserves_success(self):
        from roboodyssey.evaluation.task_diagnostics import analyze_run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';session=root/'scenes/s/sessions/session_000'
            session.mkdir(parents=True)
            gt,before,result=SemanticMetricTests().base('answer')
            gt['state_assumption']='actual_predispatch_state'
            gt['goals'][0].update(expected_outcome={'answer_type':'room','accepted_answers':['study']},
                inspection_contract={'object_id':'book','min_visible_pixels':8,'max_distance_m':2})
            before['robot_pose']=pose(5.5)
            result.update(dispatched=True,agent_status='ok',reason_code='agent_done',steps=0,prediction='study')
            result['goal_results'][0].update(observation_valid=False,answer_correct=True)
            result['evaluation']=evaluate_metrics(gt,before,before,[pose(5.5)],[],result,metric_version=LEGACY_METRIC_VERSION)
            public={'task_id':'t','instruction':'Where is the book?'}
            task={'original':public,'original_ground_truth':gt,'state_before':before,
                'repair':{'status':'ready','public_task':public,'runtime_ground_truth':gt,'repair_record':{}}}
            for name,rows in [('runtime_tasks.jsonl',[task]),('request_outcomes.jsonl',[result]),
                              ('agent_io.jsonl',[{'direction':'request','request_type':'step','task_id':'t','payload':public}])]:
                (session/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
            (root/'run_config.json').write_text(json.dumps({'official':False}))
            (session/'metrics.json').write_text(json.dumps({'run_status':'completed',
                'evaluation_valid':True,'task_results':[result]}))
            saved={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            original=analyze_run(root)
            self.assertEqual(original['issues'],[])
            self.assertEqual(original['summary']['formal_rule_average'],0)
            self.assertEqual(original['tasks'][0]['metric_version'],LEGACY_METRIC_VERSION)
            current=analyze_run(root,rescore=True)
            self.assertEqual(current['issues'],[])
            self.assertEqual(current['summary']['formal_rule_average'],100)
            self.assertEqual(current['tasks'][0]['original_task_score'],0)
            self.assertFalse(current['tasks'][0]['success'])
            for file,data in saved.items():
                self.assertEqual(file.read_bytes(),data)

    def test_historical_timing_reports_spans_without_inventing_components(self):
        from roboodyssey.evaluation.rescore import historical_measurements
        measured=historical_measurements([
            {'event':'task_start','monotonic_ns':1_000_000_000},
            {'event':'task_complete','monotonic_ns':4_000_000_000}],
            [{'direction':'request','timestamp':10.}, {'direction':'response','timestamp':11.}])
        self.assertEqual(measured['recorded_task_span_seconds'],3)
        self.assertEqual(measured['recorded_agent_roundtrip_seconds'],1)
        self.assertNotIn('action_seconds',measured)
        missing=historical_measurements([], [{'direction':'request','timestamp':10.}])
        self.assertIsNone(missing['recorded_task_span_seconds'])
        self.assertIsNone(missing['recorded_agent_roundtrip_seconds'])

    def test_saved_runtime_score_recomputes_offline_and_detects_missing_or_tampered_grid(self):
        from roboodyssey.evaluation.task_diagnostics import analyze_run
        from roboodyssey.reporting.task_diagnostics_report import write_diagnostics
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';session=root/'scenes/s/sessions/session_000'
            session.mkdir(parents=True)
            gt=truth(15.5);actions=trajectory(5.5,10.5)
            actions[0]['task_id']='t';actions[0]['result']['collision']=False
            before={'robot_pose':pose(5.5),'objects':{}}
            after={'robot_pose':pose(10.5),'objects':{}}
            env=SimpleNamespace(evaluation_navigation=lambda:{'blocked':np.zeros((6,30),bool),
                'origin':[0,0],'resolution':1,'radius':0})
            reference,grid=prepare_reference(env,gt,session)
            result={'task_id':'t','scored':True,'dispatched':True,'agent_status':'ok',
                'success':False,'reason_code':'agent_done','steps':1,
                'goal_results':[{'goal_id':'g0','success':False}]}
            result['task_timing']={'started_monotonic_s':10.,'finished_monotonic_s':12.,
                'elapsed_seconds':2.,'request_count':1,'agent_wait_seconds':1.,'action_seconds':.1}
            result['evaluation']=evaluate_metrics(gt,before,after,[pose(5.5),pose(10.5)],actions,result,
                reference=reference,grid=grid,timing=result['task_timing'])
            self.assertEqual(result['evaluation']['score'],50)
            public={'task_id':'t','instruction':'Go to the target.'}
            original=deepcopy(gt)
            original['goals'][0]['expected_outcome']['target_geometry']['world_aabb_xy']={'min':[100,0],'max':[101,1]}
            task={'original':public,'original_ground_truth':original,'state_before':before,
                'metric_reference':reference,'repair':{'status':'ready','public_task':public,
                'runtime_ground_truth':gt,'repair_record':{'changed':True}}}
            def save(name,rows):
                (session/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
            (root/'run_config.json').write_text(json.dumps({'official':False}))
            save('runtime_tasks.jsonl',[task]);save('action_changelog.jsonl',actions)
            save('request_outcomes.jsonl',[result]);save('agent_io.jsonl',[
                {'direction':'request','request_type':'step','task_id':'t','payload':public}])
            (session/'metrics.json').write_text(json.dumps({'run_status':'completed',
                'evaluation_valid':True,'task_results':[result]}))
            report=analyze_run(root)
            self.assertEqual(report['issues'],[])
            self.assertEqual(report['summary']['formal_rule_average'],50)
            self.assertEqual(report['summary']['failures'],1)
            self.assertEqual(report['tasks'][0]['evaluation'],result['evaluation'])
            output=Path(tmp)/'report';write_diagnostics(report,output)
            self.assertIn(METRIC_VERSION,(output/'report.html').read_text())
            self.assertIn('metric_path_efficiency',(output/'tasks.csv').read_text())
            file=session/reference['file'];saved=file.read_bytes()
            file.write_bytes(b'tampered')
            report=analyze_run(root)
            self.assertIsNone(report['summary']['formal_rule_average'])
            self.assertTrue(any('checksum' in str(i) for i in report['issues']))
            file.write_bytes(saved);file.unlink()
            report=analyze_run(root)
            self.assertIsNone(report['summary']['formal_rule_average'])
            self.assertFalse(report['summary']['task_score_evidence_valid'])
            # Explicit rescoring of a legacy run must never fall back to its
            # original binary score when a path reference is unavailable.
            result.pop('evaluation');result.pop('task_timing');task.pop('metric_reference')
            save('request_outcomes.jsonl',[result]);save('runtime_tasks.jsonl',[task])
            (session/'metrics.json').write_text(json.dumps({'run_status':'completed',
                'evaluation_valid':True,'task_results':[result]}))
            self.assertEqual(analyze_run(root)['summary']['formal_rule_average'],0)
            report=analyze_run(root,rescore=True)
            self.assertIsNone(report['summary']['formal_rule_average'])
            self.assertEqual(report['tasks'][0]['original_task_score'],0)
            self.assertEqual(report['tasks'][0]['metric_version'],METRIC_VERSION)

    def test_sequence_aggregation_does_not_mix_metric_versions_or_hide_missing_scores(self):
        from roboodyssey.launch.sequences import aggregate_runs
        good={'exit_code':0,'summary':{'evaluation_valid':True,'metric_version':METRIC_VERSION,
            'metrics_complete':True,'task_score':.4,'task_success_rate':.2}}
        total=aggregate_runs([good])
        self.assertEqual(total['score_100'],40)
        self.assertEqual(total['task_success_rate'],.2)
        incomplete=deepcopy(good)
        incomplete['summary'].update(metrics_complete=False,task_score=None)
        self.assertIsNone(aggregate_runs([good,incomplete])['score_100'])
        incomplete['summary'].update(metrics_complete=True,tasks_evaluated=1)
        self.assertIsNone(aggregate_runs([good,incomplete])['score_100'])
        legacy=deepcopy(good);legacy['summary']['metric_version']='binary_success_v1'
        self.assertIsNone(aggregate_runs([good,legacy])['score_100'])
        empty=deepcopy(good);empty['summary'].update(task_score=None,task_success_rate=None,tasks_evaluated=0)
        self.assertEqual(aggregate_runs([good,empty])['score_100'],40)


if __name__=='__main__':
    unittest.main()
