"""Generic serial sequence attempts with bounded infrastructure retries.

Every attempt is a new process and output directory. Completed task failures
are final results; retries never select for success rate or restore checkpoints.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

from ..artifacts.layout import read_json, write_json, sha256_file, safe_id, update_status
from ..data import load_sequence
from .sequences import stop_owned

OPTIONS = {'max_steps', 'agent_timeout', 'gpu', 'viewer', 'record_video', 'official',
           'robot', 'robot_assets', 'robot_profile', 'horizontal_fov_degrees'}
BOOL_OPTIONS = {'viewer', 'record_video', 'official'}


def build_plan(config, *, base=None):
    """CPU-only immutable input and command selection; no service/socket probes."""
    base = Path(base or Path.cwd()).resolve()
    config = deepcopy(config)
    allowed = {'schema_version', 'python', 'agent_argv', 'jobs', 'run_options',
               'max_attempts', 'attempt_timeout_seconds', 'environment'}
    if set(config)-allowed or config.get('schema_version') != 1:
        raise ValueError('unsupported batch config fields/schema')
    executable = config.get('python', sys.executable)
    executable = shutil.which(str(base/executable)) if '/' in executable else shutil.which(executable)
    if not executable:
        raise ValueError('batch Python executable not found')
    agent = config.get('agent_argv')
    if not isinstance(agent, list) or not agent or any(not isinstance(v, str) or not v for v in agent):
        raise ValueError('agent_argv must be a nonempty argument array')
    agent_executable = shutil.which(str(base/agent[0])) if '/' in agent[0] else shutil.which(agent[0])
    if not agent_executable:
        raise ValueError('agent executable not found')
    agent[0] = agent_executable
    attempts = config.get('max_attempts', 1)
    timeout = config.get('attempt_timeout_seconds', 86400.)
    if type(attempts) is not int or not 1 <= attempts <= 10:
        raise ValueError('max_attempts must be an integer in [1,10]')
    if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('attempt timeout must be positive finite seconds')
    options = config.get('run_options', {})
    if not isinstance(options, dict) or set(options)-OPTIONS:
        raise ValueError('unsupported run_options')
    for key, value in options.items():
        if key in BOOL_OPTIONS and type(value) is not bool:
            raise ValueError(key+' must be boolean')
        if key in {'max_steps', 'gpu'} and (type(value) is not int or value < (0 if key == 'gpu' else 1)):
            raise ValueError('invalid '+key)
        if key in {'agent_timeout', 'horizontal_fov_degrees'} and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
            raise ValueError('invalid '+key)
        if key in {'robot_profile', 'robot_assets'}:
            options[key] = str((base/Path(value)).resolve())
    if options.get('official') and (options.get('viewer') or options.get('record_video') or
            options.get('max_steps', 2000) != 2000 or options.get('agent_timeout', 120.) != 120. or
            options.get('horizontal_fov_degrees', 100.) != 100. or options.get('robot_profile') or
            options.get('robot', 'stretch3') != 'stretch3'):
        raise ValueError('official batch options violate the frozen profile')
    if not 1 < options.get('horizontal_fov_degrees', 100.) < 179:
        raise ValueError('horizontal_fov_degrees must be between 1 and 179')
    from ..scenes.robot_body import resolve_robot, validate_official_robot, validate_scene_robot
    robot = resolve_robot(options.get('robot'), profile=options.get('robot_profile'), assets=options.get('robot_assets'))
    if options.get('official'):
        validate_official_robot(robot)
    environment = config.get('environment', {})
    if not isinstance(environment, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k,v in environment.items()):
        raise ValueError('environment must map names to strings')
    raw_jobs = config.get('jobs')
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError('jobs must be a nonempty list')
    jobs, ids = [], set()
    for spec in raw_jobs:
        if not isinstance(spec, dict) or set(spec)-{'id', 'dataset', 'sequence_id'}:
            raise ValueError('invalid job specification')
        job_id = safe_id(spec['id'])
        if job_id in ids:
            raise ValueError('duplicate batch job id')
        ids.add(job_id)
        source = load_sequence((base/Path(spec['dataset'])).resolve(), sequence_id=spec.get('sequence_id'))
        validate_scene_robot(source, robot.identity(), require_declared=bool(options.get('official')))
        identity = {key: source.manifest[key] for key in ('scene_id', 'sequence_id', 'scene_reference')}
        argv = [executable, '-m', 'roboodyssey.run', '--dataset', str(source.root),
                '--agent-cmd', shlex.join(agent)]
        for key, value in options.items():
            if key in BOOL_OPTIONS:
                if value:
                    argv.append('--'+key.replace('_', '-'))
            else:
                argv += ['--'+key.replace('_', '-'), str(value)]
        jobs.append({'id': job_id, 'dataset': str(source.root), 'identity': identity,
                     'dataset_manifest_sha256': sha256_file(source.root/'dataset_manifest.json'),
                     'simulation_asset_available': source.version.asset_path() is not None,
                     'argv': argv, 'state': 'pending', 'attempts': []})
    return {'schema_version': 1, 'format': 'benchmark_batch_plan_v1', 'cwd': str(base),
            'max_attempts': attempts, 'attempt_timeout_seconds': float(timeout),
            'environment': environment, 'jobs': jobs,
            'retry_policy': 'infrastructure_failure_only; first_valid_completion; no_checkpoint_restore',
            'preflight_ok': all(job['simulation_asset_available'] for job in jobs)}


def execute(plan, output):
    """Preserve every attempt and give every planned job a terminal state."""
    if plan.get('format') != 'benchmark_batch_plan_v1' or not plan.get('preflight_ok'):
        raise ValueError('batch CPU preflight failed (a simulation asset is required)')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'logs').mkdir()
    write_json(root/'plan.json', plan)
    jobs = deepcopy(plan['jobs'])
    attempts = []
    process = None
    cancelled = False
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def interrupt(signum, frame):
        nonlocal cancelled
        cancelled = True
    for sig in handlers:
        signal.signal(sig, interrupt)
    def save():
        write_json(root/'jobs.json', jobs)
        write_json(root/'manifest.json', {'artifact_schema': 'benchmark_results', 'schema_version': 2,
            'kind': 'experiment', 'runs': attempts, 'plan': 'plan.json', 'jobs': 'jobs.json'})
    def stopped():
        return cancelled or (root/'STOP').exists()
    update_status(root, phase='running', execution='running')
    save()
    try:
        for job in jobs:
            if stopped():
                job.update(state='cancelled', reason='cancelled_before_start')
                continue
            for index in range(1, plan['max_attempts']+1):
                if stopped():
                    job.update(state='cancelled', reason='cancelled_before_retry')
                    break
                attempt = {'job_id': job['id'], 'attempt': index,
                    'identity': job['identity'], 'directory': f"attempts/{job['id']}/{index:03d}",
                    'previous_attempt': job['attempts'][-1] if job['attempts'] else None,
                    'state': 'preparing', 'exit_code': None, 'retryable': True}
                job['attempts'].append(attempt['directory'])
                attempts.append(attempt)
                job['state'] = 'running'
                run = root/attempt['directory']
                run.parent.mkdir(parents=True, exist_ok=True)
                argv = [*job['argv'], '--run-directory', str(run)]
                attempt['argv'] = argv
                save()
                stage = 'input_validation'
                try:
                    # Revalidate all referenced content immediately before launch.
                    current = load_sequence(job['dataset'])
                    if sha256_file(current.root/'dataset_manifest.json') != job['dataset_manifest_sha256']:
                        raise ValueError('frozen dataset changed after planning')
                    stage = 'process_start'
                    env = os.environ.copy()
                    env.update(plan['environment'])
                    env['PYTHONUNBUFFERED'] = '1'
                    started = time.monotonic()
                    with (root/'logs'/f"{job['id']}_{index:03d}.log").open('x') as log:
                        process = subprocess.Popen(argv, cwd=plan['cwd'], env=env,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
                        attempt.update(state='running', pid=process.pid)
                        save()
                        while process.poll() is None:
                            if stopped():
                                attempt['state'] = 'cancelled'
                                break
                            if time.monotonic()-started > plan['attempt_timeout_seconds']:
                                attempt['state'] = 'timed_out'
                                break
                            time.sleep(.1)
                        attempt['exit_code'] = stop_owned(process)
                        process = None
                    attempt['elapsed_seconds'] = time.monotonic()-started
                    stage = 'output_validation'
                    if stopped():
                        attempt.update(state='cancelled', retryable=False)
                    elif (run/'summary.json').is_file():
                        summary = read_json(run/'summary.json')
                        if summary.get('sequence_identity') != job['identity']:
                            raise ValueError('attempt summary identity mismatch')
                        attempt['summary'] = summary
                        if summary.get('evaluation_valid') is True:
                            # The evaluator can finish before SDK shutdown fails.
                            # Never redo an already scored experiment to fix cleanup.
                            attempt.update(state='completed', retryable=False)
                            if attempt['exit_code'] != 0:
                                attempt['cleanup_issue'] = 'process_exit_after_valid_completion'
                        elif attempt['state'] != 'timed_out':
                            attempt.update(state='failed', reason='invalid_evaluation_evidence')
                    elif attempt['state'] != 'timed_out':
                        attempt.update(state='failed', reason='process_exit' if attempt['exit_code'] else 'missing_summary')
                    if attempt['state'] == 'timed_out':
                        attempt['reason'] = 'attempt_timeout'
                except Exception as exc:
                    if process is not None:
                        attempt['exit_code'] = stop_owned(process)
                        process = None
                    retryable = stage != 'input_validation' and not isinstance(exc, (ValueError, KeyError, TypeError))
                    attempt.update(state='failed', reason=stage+': '+type(exc).__name__+': '+str(exc),
                                   retryable=retryable)
                # A prelaunch failure still owns a concrete attempt directory.
                run.mkdir(parents=True, exist_ok=True)
                write_json(run/'attempt_status.json', {k: v for k, v in attempt.items() if k != 'summary'})
                save()
                if attempt['state'] in {'completed', 'cancelled'}:
                    job['state'] = attempt['state']
                    job['selected_attempt'] = attempt['directory'] if attempt['state'] == 'completed' else None
                    break
                if not attempt['retryable']:
                    job.update(state='failed', reason='nonretryable_contract_error')
                    break
                if index == plan['max_attempts']:
                    job.update(state='failed', reason='attempts_exhausted')
            save()
    finally:
        if process is not None:
            stop_owned(process)
        for job in jobs:
            if job['state'] in {'pending', 'running'}:
                job.update(state='cancelled' if stopped() else 'failed', reason='supervisor_terminated')
        try:
            save()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        complete = all(job['state'] == 'completed' for job in jobs)
        summary = {'schema_version': 1, 'jobs': len(jobs), 'attempts': len(attempts),
                   'execution_complete': complete, 'states': {state: sum(job['state']==state for job in jobs)
                    for state in ('completed', 'failed', 'cancelled')},
                   'formal_score': None, 'score_policy': 'compare explicit selected attempts; never best-of-retry'}
        write_json(root/'summary.json', summary)
        update_status(root, phase='completed' if complete else 'cancelled' if stopped() else 'failed',
                      execution='completed' if complete else 'incomplete', cleanup='completed')
    return 0 if summary['execution_complete'] else 130 if stopped() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='new plan or batch directory')
    parser.add_argument('--execute', action='store_true', help='otherwise write a CPU plan only')
    args = parser.parse_args(argv)
    plan = build_plan(read_json(args.config), base=args.config.resolve().parent)
    if args.execute:
        return execute(plan, args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'plan.json', plan)
    print(args.output/'plan.json')
    return 0 if plan['preflight_ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
