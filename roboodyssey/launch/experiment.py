"""Explicit real-run supervisor; CPU planning never starts services or probes ports."""
from datetime import datetime
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.parse import urlparse
import uuid

from ..artifacts.layout import read_json, write_json, update_status, safe_id, iter_sessions, session_file
from .provenance import capture_provenance
from .simplenav import commands, preflight, selected_datasets
from .sequences import aggregate_runs, stop_owned


def service_command(config):
    parsed = urlparse(config['url'])
    return [config['agent_python'], '-u', '-m', 'deployment.model_server.server_policy',
            '--ckpt_path', config['checkpoint'], '--port', str(parsed.port or 80),
            *config['model_service'].get('extra_args', [])]


def port_ready(url):
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=.5):
            return True
    except OSError:
        return False


class Progress:
    def __init__(self):
        self.offsets = {}
        self.tasks = set()
        self.counts = {'completed_tasks': 0, 'successes': 0, 'failures': 0, 'excluded': 0,
                       'actions': 0, 'collisions': 0, 'resets': 0}

    def read(self, run):
        if not (run/'manifest.json').exists():
            return self.counts
        for scene, sid, folder in iter_sessions(run):
            path = session_file(folder, 'runtime_events.jsonl')
            if not path.exists():
                continue
            with path.open('rb') as stream:
                stream.seek(self.offsets.get(path, 0))
                data = stream.read()
            end = data.rfind(b'\n')
            if end < 0:
                continue
            self.offsets[path] = self.offsets.get(path, 0) + end + 1
            import json
            for line in data[:end+1].splitlines():
                event = json.loads(line)
                if event['event'] in {'step', 'reset'}:
                    self.counts['actions'] += 1
                    self.counts['collisions'] += event.get('result', {}).get('collision') is True
                    self.counts['resets'] += event.get('result', {}).get('reset_applied') is True
                if event['event'] == 'task_complete':
                    key = (str(run), scene, sid, event['task_id'])
                    if key not in self.tasks:
                        self.tasks.add(key)
                        self.counts['completed_tasks'] += 1
                        name = 'successes' if event.get('success') is True else 'failures' if event.get('success') is False else 'excluded'
                        self.counts[name] += 1
        return self.counts


def execute(config, *, checked=None):
    checked = checked or preflight(config)
    if not checked['ok']:
        raise ValueError('CPU preflight failed; no model or simulator started')
    datasets = selected_datasets(config)
    root = Path(config['output']) / ('experiment_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    (root/'logs').mkdir()
    entries = []
    for i, data in enumerate(datasets):
        identity = data.sequence_identity()
        key = safe_id(identity.get('sequence_id') or f'default_{i:03d}')
        entries.append({'identity': identity, 'dataset': str(data.root), 'directory': f'sequences/{key}', 'exit_code': None})
    write_json(root/'manifest.json', {'artifact_schema': 'benchmark_results', 'schema_version': 2, 'kind': 'experiment',
               'runs': entries, 'config': 'config.json', 'status': 'status.json', 'reports': 'reports'})
    write_json(root/'config.json', config)
    write_json(root/'preflight.json', checked)
    update_status(root, phase='preparing', execution='pending', analysis='pending', cleanup='pending')
    model = child = None
    records = []
    handles = []
    progress = Progress()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'experiment interrupted by signal {signum}')
    for sig in previous:
        signal.signal(sig, interrupted)
    code = 1
    execution_finished = False
    try:
        capture_provenance(root/'provenance.json', dataset_root=config['dataset'], model_root=config['simplenav_root'],
                           checkpoint=config['checkpoint'], python_executables=[config['agent_python']])
        if (root/'STOP').exists():
            raise KeyboardInterrupt('STOP file requested before process startup')
        service = config.get('model_service', {'mode': 'external'})
        if service['mode'] == 'managed':
            if port_ready(config['url']):
                raise RuntimeError('managed model port is occupied; select external mode to reuse a service')
            env = os.environ.copy()
            env.update(service.get('environment', {}))
            env.update(CUDA_VISIBLE_DEVICES=service['gpu_uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONPATH=config['simplenav_root'])
            handle = (root/'logs/model.log').open('x'); handles.append(handle)
            argv = service_command(config)
            model = subprocess.Popen(argv, cwd=config['simplenav_root'], env=env, stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(root/'processes.json', {'model_pid': model.pid, 'model_argv': argv, 'external_model': False})
            update_status(root, phase='loading_model')
            deadline = time.monotonic() + service.get('startup_timeout', 600)
            while not port_ready(config['url']):
                if (root/'STOP').exists():
                    raise KeyboardInterrupt('STOP file requested during model startup')
                if model.poll() is not None:
                    raise RuntimeError(f'model exited during startup: {model.returncode}')
                if time.monotonic() >= deadline:
                    raise TimeoutError('model startup timeout')
                time.sleep(.25)
        elif not port_ready(config['url']):
            raise RuntimeError('external model service is not reachable')
        else:
            write_json(root/'processes.json', {'external_model': True, 'url': config['url']})
        update_status(root, phase='running', execution='running')
        for entry in entries:
            if (root/'STOP').exists():
                raise KeyboardInterrupt('STOP file requested before next sequence')
            record = {**entry}; records.append(record)
            run = root/entry['directory']
            selected = {**config, 'dataset': entry['dataset'], 'sequences': [], 'all_sequences': False}
            command = commands(selected)
            argv = [*command['benchmark_argv'], '--run-directory', str(run)]
            handle = (root/'logs'/f'{run.name}.isaac.log').open('x'); handles.append(handle)
            env = os.environ.copy()
            env.pop('CUDA_VISIBLE_DEVICES', None)
            env.pop('CONDA_PREFIX', None)
            env.update(config.get('environment', {}))
            env['PYTHONUNBUFFERED'] = '1'
            child = subprocess.Popen(argv, cwd=command['cwd'], env=env, stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            record.update(benchmark_pid=child.pid, benchmark_argv=argv)
            write_json(root/'runs.json', records)
            last_refresh = 0
            while child.poll() is None:
                if model is not None and model.poll() is not None:
                    raise RuntimeError('managed model exited during benchmark execution')
                if (root/'STOP').exists():
                    raise KeyboardInterrupt('STOP file requested')
                if time.monotonic() - last_refresh >= 2:
                    write_json(root/'progress.json', progress.read(run))
                    last_refresh = time.monotonic()
                time.sleep(.25)
            record['exit_code'] = child.returncode
            stop_owned(child); child = None
            write_json(root/'progress.json', progress.read(run))
            if (run/'summary.json').is_file():
                record['summary'] = read_json(run/'summary.json')
            record['result_directory'] = str(run)
            write_json(root/'runs.json', records)
        aggregate = aggregate_runs(records)
        write_json(root/'summary.json', aggregate)
        execution_finished = True
        update_status(root, execution='completed' if aggregate['evaluation_valid'] else 'failed', summary='completed')
        if config.get('auto_analyze', True):
            from ..reporting.analyze import analyze
            report = analyze(root, root/'reports')
            update_status(root, analysis='completed' if report['ok'] else 'failed')
            if not report['ok']:
                raise RuntimeError('analysis contains evidence errors; see reports/index.html')
        else:
            update_status(root, analysis='disabled')
        code = 0 if aggregate['evaluation_valid'] else 1
    except BaseException as exc:
        update_status(root, phase='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      **({} if execution_finished else {'execution':'failed'}),
                      error=f'{type(exc).__name__}: {exc}')
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        errors = []
        for process in (child, model):
            try:
                stop_owned(process)
            except Exception as exc:
                errors.append(str(exc))
        if child is not None and records:
            records[-1]['exit_code'] = child.poll()
            run = root/records[-1]['directory']
            if (run/'summary.json').is_file():
                records[-1]['summary'] = read_json(run/'summary.json')
                records[-1]['result_directory'] = str(run)
        for handle in handles:
            handle.close()
        if not (root/'summary.json').exists():
            aggregate = aggregate_runs(records)
            aggregate.update(evaluation_valid=False, task_score=None, score_100=None,
                             task_success_rate=None, metrics_complete=False, interrupted=len(records) < len(entries))
            write_json(root/'summary.json', aggregate)
        write_json(root/'runs.json', records)
        if config.get('auto_analyze', True) and not (root/'reports').exists():
            try:
                from ..reporting.analyze import analyze
                report = analyze(root, root/'reports')
                update_status(root, analysis='completed' if report['ok'] else 'failed')
            except Exception as exc:
                update_status(root, analysis='failed', analysis_error=str(exc))
        final_phase = 'completed' if code == 0 and not errors else 'interrupted' if read_json(root/'status.json').get('phase') == 'interrupted' else 'failed'
        update_status(root, phase=final_phase,
                      cleanup='failed' if errors else 'completed', cleanup_errors=errors, exit_code=code if not errors else 1)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(root)
    return (0 if code == 0 and not errors else 1), root
