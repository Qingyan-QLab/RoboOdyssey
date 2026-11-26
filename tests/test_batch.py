import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from roboodyssey.launch.batch import build_plan, execute
from tests.frozen_data import fixture_v3, select_scenario, write_dataset


class BatchTests(unittest.TestCase):
    def fixture(self, behavior, *, timeout=10, jobs=1):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        c, inputs, plan = fixture_v3()
        write_dataset(root/'data', select_scenario(c, plan), c, inputs)
        config = {'schema_version': 1, 'python': sys.executable,
                  'agent_argv': [sys.executable, '-m', 'roboodyssey.agents.random_agent'],
                  'max_attempts': 2, 'attempt_timeout_seconds': timeout,
                  'jobs': [{'id': f'job_{i}', 'dataset': str(root/'data')} for i in range(jobs)]}
        prepared = build_plan(config)
        self.assertFalse(prepared['preflight_ok'])  # Frozen CPU fixture has no simulator asset.
        # Inject a separate CPU supervisor fixture, never claimed as Isaac evidence.
        prepared['preflight_ok'] = True
        script = root/'child.py'
        script.write_text('import json,sys,time\nfrom pathlib import Path\n'
            "run=Path(sys.argv[sys.argv.index('--run-directory')+1]); run.mkdir(parents=True)\n"
            + behavior + '\n'
            + 'summary='+repr({'sequence_identity': prepared['jobs'][0]['identity'],
                              'evaluation_valid': True, 'task_success_rate': 0,
                              'metrics_complete': True, 'backend': 'cpu_supervisor_fixture'})+'\n'
            + "(run/'summary.json').write_text(json.dumps(summary))\n")
        for job in prepared['jobs']:
            job['argv'] = [sys.executable, str(script)]
        return root, prepared, config

    def test_infrastructure_retry_keeps_first_attempt_and_never_retries_task_failure(self):
        root, plan, _ = self.fixture("if run.name=='001': sys.exit(23)")
        self.assertEqual(execute(plan, root/'run'), 0)
        manifest = json.loads((root/'run/manifest.json').read_text())
        first, second = manifest['runs']
        self.assertEqual(first['exit_code'], 23)
        self.assertEqual(first['state'], 'failed')
        self.assertEqual(second['previous_attempt'], first['directory'])
        self.assertEqual(second['state'], 'completed')
        self.assertEqual(second['summary']['task_success_rate'], 0)
        self.assertTrue((root/'run'/first['directory']).is_dir())
        self.assertTrue((root/'run'/second['directory']).is_dir())
        jobs = json.loads((root/'run/jobs.json').read_text())
        self.assertEqual(jobs[0]['selected_attempt'], second['directory'])

    def test_completed_failed_tasks_are_final_without_retry(self):
        root, plan, _ = self.fixture('pass')
        self.assertEqual(execute(plan, root/'run'), 0)
        self.assertEqual(len(json.loads((root/'run/manifest.json').read_text())['runs']), 1)

    def test_valid_result_survives_sdk_shutdown_nonzero_exit_without_retry(self):
        root, plan, _ = self.fixture('pass')
        script = root/'child.py'
        script.write_text(script.read_text()+'sys.exit(23)\n')
        self.assertEqual(execute(plan, root/'run'), 0)
        attempts = json.loads((root/'run/manifest.json').read_text())['runs']
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]['summary']['task_success_rate'], 0)
        self.assertEqual(attempts[0]['exit_code'], 23)
        self.assertEqual(attempts[0]['cleanup_issue'], 'process_exit_after_valid_completion')

    def test_input_tampering_never_launches_or_retries_and_has_attempt_record(self):
        root, plan, _ = self.fixture('pass')
        dataset = Path(plan['jobs'][0]['dataset'])
        (dataset/'dataset_manifest.json').write_text('{}')
        with patch('roboodyssey.launch.batch.subprocess.Popen', side_effect=AssertionError('process started')):
            self.assertEqual(execute(plan, root/'run'), 1)
        attempts = json.loads((root/'run/manifest.json').read_text())['runs']
        self.assertEqual(len(attempts), 1)
        self.assertFalse(attempts[0]['retryable'])
        self.assertTrue((root/'run'/attempts[0]['directory']/'attempt_status.json').is_file())

    def test_wrong_output_identity_is_nonretryable(self):
        root, plan, _ = self.fixture('pass')
        script = root/'child.py'
        script.write_text(script.read_text().replace("'sequence_id': 'seq_fixture'", "'sequence_id': 'wrong'"))
        self.assertEqual(execute(plan, root/'run'), 1)
        attempts = json.loads((root/'run/manifest.json').read_text())['runs']
        self.assertEqual(len(attempts), 1)
        self.assertFalse(attempts[0]['retryable'])
        self.assertIn('identity mismatch', attempts[0]['reason'])

    def test_stop_cancels_running_process_and_every_pending_job(self):
        root, plan, _ = self.fixture("(run.parents[2]/'STOP').touch()\ntime.sleep(20)", jobs=2)
        self.assertEqual(execute(plan, root/'run'), 130)
        jobs = json.loads((root/'run/jobs.json').read_text())
        self.assertEqual([j['state'] for j in jobs], ['cancelled', 'cancelled'])
        attempts = json.loads((root/'run/manifest.json').read_text())['runs']
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]['state'], 'cancelled')
        self.assertNotEqual(attempts[0]['exit_code'], 0)

    def test_timeouts_exhaust_bounded_attempts_and_keep_terminal_records(self):
        root, plan, _ = self.fixture('time.sleep(20)', timeout=.15)
        self.assertEqual(execute(plan, root/'run'), 1)
        attempts = json.loads((root/'run/manifest.json').read_text())['runs']
        self.assertEqual([a['state'] for a in attempts], ['timed_out', 'timed_out'])
        self.assertTrue(all(a['exit_code'] is not None for a in attempts))
        self.assertEqual(json.loads((root/'run/jobs.json').read_text())[0]['state'], 'failed')

    def test_cpu_plan_never_starts_process_and_missing_asset_blocks_execution(self):
        root, _, config = self.fixture('pass')
        with patch('roboodyssey.launch.batch.subprocess.Popen', side_effect=AssertionError('process started')):
            plan = build_plan(config)
            with self.assertRaisesRegex(ValueError, 'simulation asset'):
                execute(plan, root/'run')
        self.assertFalse((root/'run').exists())


if __name__ == '__main__':
    unittest.main()
