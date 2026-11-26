from roboodyssey.artifacts.layout import session_file
"""CPU contracts for versioned artifacts, CLI dispatch and process ownership."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from roboodyssey.artifacts.layout import initialize_run, iter_sessions, session_file, read_json, write_json
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
            for structured in (True,):
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




class ReusedEntryTests(unittest.TestCase):

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



    def test_analysis_preserves_non_overwrite_contract(self):
        from roboodyssey.cli import main
        with tempfile.TemporaryDirectory() as temp, \
             patch('roboodyssey.evaluation.task_diagnostics.analyze_run') as analyze_run:
            with self.assertRaises(FileExistsError):
                main(['analyze', 'unused-run', '--output', temp])
            analyze_run.assert_not_called()


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
