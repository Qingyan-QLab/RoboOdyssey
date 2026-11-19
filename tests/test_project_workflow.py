"""CPU contracts for versioned artifacts, CLI dispatch and process ownership."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from roboodyssey.artifacts.layout import initialize_run, iter_sessions, session_file, read_json, write_json
from roboodyssey.launch.experiment import execute, stop_owned, Progress
from roboodyssey.reporting.analyze import analyze
from roboodyssey.scenes.dataset_loader import BenchmarkDataset
from roboodyssey.runtime.session_runner import SessionRunner
from roboodyssey.evaluation.report import write_summary
from tests.test_contextual_requests import Env, Agent
from tests.frozen_data import fixture_v3, task, goal
from tests.frozen_data import select_scenario
from tests.frozen_data import write_dataset


class ArtifactWorkflowTests(unittest.TestCase):
    def test_structured_run_preserves_scores_and_removes_duplicate_action_arrays(self):
        c, inputs, plan = fixture_v3([task('visit', goal('cabinet_dark'))])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); archive = root/'dataset'
            write_dataset(archive, select_scenario(c, plan), c, inputs)
            dataset = BenchmarkDataset(archive)
            summaries = []
            for structured in (False, True):
                output = root/str(structured)
                if structured:
                    initialize_run(output, dataset)
                state = dataset.canonical_state('synthetic_demo')
                SessionRunner(output, Agent(), max_steps=1).run_scene('synthetic_demo', dataset.sessions('synthetic_demo'), state, lambda *args: Env(state))
                summaries.append(write_summary(dataset, output))
                scene, sid, session = next(iter_sessions(output))
                outcomes = [json.loads(line) for line in session_file(session,'request_outcomes.jsonl').read_text().splitlines()]
                self.assertEqual('actions' in outcomes[0]['goal_results'][0], not structured)
                if structured:
                    self.assertIn('action_evidence', outcomes[0]['goal_results'][0])
                    self.assertTrue((session/'traces/agent_io.jsonl').exists())
                    self.assertTrue((output/'reports/request_audit.html').exists())
                    self.assertFalse((output/'scenes').exists())
                    write_json(output/'run_config.json', {'dataset_root': str(archive), 'official': False})
                    report = analyze(output, root/'analysis')
                    self.assertTrue(report['ok'], report)
                    self.assertTrue((root/'analysis/scores/report.html').exists())
                    progress = Progress()
                    once = deepcopy(progress.read(output))
                    self.assertEqual(once, progress.read(output))
                    self.assertEqual(once['completed_tasks'], 1)
                    legacy=read_json(output/'manifest.json');legacy['runtime_contract']['version']=2
                    write_json(output/'manifest.json',legacy)
                    with patch('roboodyssey.evaluation.task_diagnostics.analyze_run') as diagnose:
                        old_report=analyze(output,root/'legacy_contract_report')
                        diagnose.assert_not_called()
                    self.assertTrue(old_report['ok']);self.assertTrue(old_report['notes'])
            for key in ('task_score','tasks_evaluated','tasks_excluded','evaluation_valid'):
                self.assertEqual(summaries[0][key],summaries[1][key])

    def test_collection_reports_include_missing_runs_as_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';root.mkdir()
            write_json(root/'manifest.json', {'artifact_schema':'benchmark_results','schema_version':2,'kind':'experiment',
                'runs':[{'identity':{'sequence_id':'missing'},'directory':'sequences/missing'}]})
            report=analyze(root,Path(temp)/'report')
            self.assertFalse(report['ok'])
            self.assertTrue((Path(temp)/'report/index.html').exists())

    def test_unrecognized_artifact_version_and_escape_are_rejected(self):
        from roboodyssey.artifacts.layout import manifest, local_path
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_json(root/'manifest.json', {'schema_version':999})
            with self.assertRaisesRegex(ValueError,'unsupported'):manifest(root)
            with self.assertRaisesRegex(ValueError,'escapes'):local_path(root,'../outside')

    def test_video_artifacts_are_written_beside_their_index(self):
        from PIL import Image
        from roboodyssey.runtime.request_events import RequestEvents
        from roboodyssey.artifacts.layout import create_session
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_json(root/'manifest.json', {'artifact_schema':'benchmark_results','schema_version':2,'kind':'run',
                'sequences':[{'scene_id':'s','sessions_root':'sequences/A/sessions','session_ids':['session_000']}]})
            folder=create_session(root,'s','session_000')
            runner=SimpleNamespace(video_options={'layout':'front','fps':2,'event_seconds':.5,'ffmpeg':'ffmpeg'},
                viewer_sync_callback=None,_show=lambda **kw:None,_log=lambda message:None)
            env=SimpleNamespace(capture_review_frames=lambda layout:{'front':Image.new('RGB',(32,32),'red')})
            events=RequestEvents(runner,env,folder,'s','session_000')
            events.emit('task_start',sample=True,task_id='t',instruction='Go to target.')
            events.close('completed')
            self.assertTrue(session_file(folder,'review.mp4').is_file())
            self.assertTrue(session_file(folder,'video_index.jsonl').is_file())
            self.assertTrue(session_file(folder,'runtime_events.jsonl').is_file())
            self.assertFalse((folder/'review.mp4').exists())


class ExperimentWorkflowTests(unittest.TestCase):
    def config(self, root, mode='managed'):
        return {'output':str(root),'dataset':'dataset','simplenav_root':str(root), 'checkpoint':'checkpoint',
            'agent_python':sys.executable,'isaac_python':'not-a-real-isaac-executable','url':'ws://127.0.0.1:12345',
            'model_service':{'mode':mode,'gpu_uuid':'GPU-test','startup_timeout':1},'environment':{},'auto_analyze':False}

    def dataset(self):
        return SimpleNamespace(root=Path('dataset'), sequence_identity=lambda:{'sequence_id':'A'})

    def test_managed_success_uses_config_and_reaps_only_owned_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            config=self.config(Path(temp)); processes=[]
            def spawn(argv, **kwargs):
                process=SimpleNamespace(pid=990001+len(processes),returncode=0, poll=lambda:0,wait=lambda **kw:0)
                processes.append((argv,kwargs,process))
                if '--run-directory' in argv:
                    root=Path(argv[argv.index('--run-directory')+1]);root.mkdir(parents=True)
                    write_json(root/'summary.json', {'evaluation_valid':True,'task_score':.5})
                return process
            with patch('roboodyssey.launch.experiment.selected_datasets',return_value=[self.dataset()]), \
                 patch('roboodyssey.launch.experiment.capture_provenance'), \
                 patch('roboodyssey.launch.experiment.commands',return_value={'cwd':temp,'benchmark_argv':['fake-isaac']}), \
                 patch('roboodyssey.launch.experiment.port_ready',side_effect=[False,True]), \
                 patch('roboodyssey.launch.experiment.subprocess.Popen',side_effect=spawn), \
                 patch('roboodyssey.launch.experiment.os.killpg') as kill:
                code,root=execute(config,checked={'ok':True})
            self.assertEqual(code,0)
            self.assertEqual(read_json(root/'status.json')['cleanup'],'completed')
            self.assertEqual(read_json(root/'summary.json')['task_score'],.5)
            self.assertEqual(processes[0][1]['env']['CUDA_VISIBLE_DEVICES'],'GPU-test')
            self.assertIn('--run-directory',processes[1][0])
            self.assertEqual({call.args[0] for call in kill.call_args_list},{p.pid for _,_,p in processes})

    def test_external_mode_does_not_spawn_or_stop_a_model(self):
        with tempfile.TemporaryDirectory() as temp:
            config=self.config(Path(temp),'external')
            with patch('roboodyssey.launch.experiment.selected_datasets',return_value=[self.dataset()]), \
                 patch('roboodyssey.launch.experiment.capture_provenance'), \
                 patch('roboodyssey.launch.experiment.port_ready',return_value=False), \
                 patch('roboodyssey.launch.experiment.subprocess.Popen') as spawn, \
                 patch('roboodyssey.launch.experiment.stop_owned') as stop:
                code,root=execute(config,checked={'ok':True})
            self.assertEqual(code,1);spawn.assert_not_called()
            self.assertTrue(all(c.args==(None,) for c in stop.call_args_list))
            self.assertFalse(read_json(root/'summary.json')['evaluation_valid'])

    def test_startup_error_keeps_explicit_invalid_summary_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            process=SimpleNamespace(pid=990003,returncode=3,poll=lambda:3,wait=lambda **kw:3)
            with patch('roboodyssey.launch.experiment.selected_datasets',return_value=[self.dataset()]), \
                 patch('roboodyssey.launch.experiment.capture_provenance'), \
                 patch('roboodyssey.launch.experiment.port_ready',return_value=False), \
                 patch('roboodyssey.launch.experiment.subprocess.Popen',return_value=process), \
                 patch('roboodyssey.launch.experiment.os.killpg') as kill:
                code,root=execute(self.config(Path(temp)),checked={'ok':True})
            self.assertEqual(code,1)
            status=read_json(root/'status.json')
            self.assertIn('startup',status['error']);self.assertEqual(status['cleanup'],'completed')
            self.assertEqual(kill.call_args.args[0],990003)

    def test_stop_during_preparation_never_launches_a_child(self):
        with tempfile.TemporaryDirectory() as temp:
            def prepare(path,**kwargs):
                (path.parent/'STOP').touch()
            with patch('roboodyssey.launch.experiment.selected_datasets',return_value=[self.dataset()]), \
                 patch('roboodyssey.launch.experiment.capture_provenance',side_effect=prepare), \
                 patch('roboodyssey.launch.experiment.subprocess.Popen') as spawn:
                code,root=execute(self.config(Path(temp)),checked={'ok':True})
            self.assertEqual(code,1);spawn.assert_not_called()
            self.assertEqual(read_json(root/'status.json')['phase'],'interrupted')

    def test_real_cpu_child_cleanup(self):
        process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
        try:
            stop_owned(process,timeout=2)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:process.kill();process.wait()

    def test_cli_planning_cannot_execute_or_probe_the_service(self):
        from roboodyssey.cli import main
        with patch('roboodyssey.launch.simplenav.load_config',return_value={}), \
             patch('roboodyssey.launch.simplenav.preflight',return_value={'ok':True}), \
             patch('roboodyssey.launch.experiment.execute') as launch, \
             patch('roboodyssey.launch.experiment.port_ready') as connect, redirect_stdout(StringIO()):
            self.assertEqual(main(['preflight','--config','config.json']),0)
            self.assertEqual(main(['experiment','--config','config.json']),0)
            launch.assert_not_called();connect.assert_not_called()

    def test_config_resolves_paths_and_rejects_ambiguous_model_or_sequences(self):
        from roboodyssey.launch.simplenav import load_config, commands
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'config.json';base=self.config(root)
            base.update(dataset='relative-dataset',viewer=False,record_video=False)
            write_json(path,base)
            config=load_config(path)
            self.assertEqual(config['dataset'],str(root/'relative-dataset'))
            self.assertNotIn('--viewer',commands(config)['benchmark_argv'])
            for update in ({'sequences':['A','A']}, {'sequences':['A'],'all_sequences':True},
                           {'model_service':{'mode':'managed'}},
                           {'model_service':{'mode':'managed','gpu_uuid':'GPU-test','extra_args':['--port=1']}}):
                write_json(path,{**base,**update})
                with self.assertRaises(ValueError):load_config(path)


class ReusedEntryTests(unittest.TestCase):
    def test_historical_imports_keep_identity_and_patch_targets_in_fresh_processes(self):
        script = '''
from importlib import import_module
import sys
from unittest.mock import patch

pairs = (
    ('roboodyssey.runtime.simplenav_launch', 'roboodyssey.launch.simplenav'),
    ('roboodyssey.runtime.sequence_runs', 'roboodyssey.launch.sequences'),
    ('roboodyssey.runtime.task_repair', 'roboodyssey.legacy.task_repair'),
    ('roboodyssey.runtime.task_history', 'roboodyssey.legacy.task_history'),
    ('roboodyssey.runtime.repair_audit', 'roboodyssey.reporting.legacy'),
    ('roboodyssey.evaluation.request_audit', 'roboodyssey.reporting.request_audit'),
    ('roboodyssey.evaluation.task_diagnostics_report', 'roboodyssey.reporting.task_diagnostics_report'),
    ('roboodyssey.evaluation.accounting', 'roboodyssey.evaluation.report'),
    ('roboodyssey.evaluation.request_report', 'roboodyssey.evaluation.report'),
    ('roboodyssey.contracts.calendar', 'roboodyssey.contracts.language'),
    ('roboodyssey.contracts.object_references', 'roboodyssey.contracts.references'),
    ('roboodyssey.contracts.routes', 'roboodyssey.contracts.navigation'),
    ('roboodyssey.reporting.legacy_repair', 'roboodyssey.reporting.legacy'),
)
for historical, canonical in pairs:
    names = (historical, canonical) if sys.argv[1] == 'historical_first' else (canonical, historical)
    assert import_module(names[0]) is import_module(names[1]), names
with patch('roboodyssey.runtime.simplenav_launch.preflight', return_value={'ok': True}):
    assert import_module('roboodyssey.launch.simplenav').preflight({}) == {'ok': True}
for forbidden in ('torch', 'isaacsim', 'omni', 'CL_task_builder', 'CL_session_builder'):
    assert forbidden not in sys.modules, forbidden
'''
        for order in ('historical_first', 'canonical_first'):
            with self.subTest(order=order):
                result = subprocess.run([sys.executable, '-c', script, order],
                    cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_shared_jsonl_reader_keeps_strict_scoring_and_tolerant_audit_errors(self):
        from roboodyssey.artifacts.layout import read_json_lines
        from roboodyssey.evaluation.task_diagnostics import read_lines
        from roboodyssey.reporting.request_audit import lines
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'events.jsonl'
            path.write_text('{"task_id":"first"}\n\n[]\n{"partial":\n{"task_id":"last"}\n')
            self.assertIs(read_lines, read_json_lines)
            with self.assertRaisesRegex(ValueError, 'expected an object'):
                list(read_lines(path))
            records = lines(path)
            self.assertEqual([r.get('task_id') for r in records], ['first', None, None, 'last'])
            self.assertEqual([r['line'] for r in records if 'log_error' in r], [3, 4])
            self.assertEqual(lines(Path(temp)/'missing.jsonl'), [])
            with self.assertRaises(FileNotFoundError):
                list(read_lines(Path(temp)/'missing.jsonl'))

    def test_preflight_entries_emit_identical_reports_and_refuse_overwrite(self):
        from roboodyssey.cli import main
        with tempfile.TemporaryDirectory() as temp, \
             patch('roboodyssey.launch.simplenav.load_config', return_value={}), \
             patch('roboodyssey.launch.simplenav.preflight', return_value={'ok': True}), \
             patch('roboodyssey.launch.experiment.execute') as execute, redirect_stdout(StringIO()):
            first, second = Path(temp)/'first.json', Path(temp)/'second.json'
            self.assertEqual(main(['preflight', '--config', 'unused', '--report', str(first)]), 0)
            self.assertEqual(main(['experiment', '--config', 'unused', '--report', str(second)]), 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with self.assertRaises(FileExistsError):
                main(['experiment', '--config', 'unused', '--report', str(first), '--execute'])
            execute.assert_not_called()

    def test_failed_preflight_never_starts_either_launch_mode(self):
        from roboodyssey.cli import main
        with patch('roboodyssey.launch.simplenav.load_config', return_value={}), \
             patch('roboodyssey.launch.simplenav.preflight', return_value={'ok': False}), \
             patch('roboodyssey.launch.experiment.execute') as execute, redirect_stdout(StringIO()):
            self.assertEqual(main(['experiment', '--config', 'unused', '--execute']), 1)
            self.assertEqual(main(['preflight', '--config', 'unused']), 1)
            execute.assert_not_called()

    def test_analysis_preserves_non_overwrite_contract(self):
        from roboodyssey.cli import main
        with tempfile.TemporaryDirectory() as temp, \
             patch('roboodyssey.evaluation.task_diagnostics.analyze_run') as analyze_run:
            with self.assertRaises(FileExistsError):
                main(['analyze', 'unused-run', '--output', temp])
            analyze_run.assert_not_called()

    def test_legacy_html_embedded_json_preserves_angle_brackets_safely(self):
        from roboodyssey.reporting.legacy import write_report
        with tempfile.TemporaryDirectory() as temp, \
             patch('roboodyssey.reporting.legacy.collect', return_value={'text': '</script><example>'}):
            output = Path(temp)/'report.html'
            write_report(Path(temp), output)
            embedded = output.read_text().split('const DATA=', 1)[1].split(';', 1)[0]
            self.assertNotIn('<', embedded)
            self.assertEqual(json.loads(embedded)['text'], '</script><example>')

    def test_file_map_is_complete_and_current(self):
        from roboodyssey.validation.inventory import render, ROOT, DOCUMENT
        self.assertEqual((ROOT/DOCUMENT).read_text(encoding='utf-8'), render())

    def test_inventory_detects_unlisted_files_without_importing_them(self):
        from roboodyssey.validation.inventory import render, DOCUMENT
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict('roboodyssey.validation.inventory.PURPOSES', {DOCUMENT: '清单'}, clear=True):
            root = Path(temp)
            (root/'new.py').write_text('raise RuntimeError("must never import")')
            with self.assertRaisesRegex(ValueError, 'new.py'):
                render(root)
            with patch.dict('roboodyssey.validation.inventory.PURPOSES', {'new.py': '静态检查示例'}):
                first = render(root)
                (root/'results').mkdir()
                (root/'results/unlisted.py').write_text('invalid Python!!!')
                self.assertEqual(first, render(root))
                self.assertIn('静态检查示例', first)


if __name__ == '__main__':unittest.main()
