"""Run collection sequences in fresh processes and aggregate explicit results."""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import sys
import uuid

from ..artifacts.layout import safe_id, write_json, update_status


def stop_owned(process, timeout=10):
    """Kill only the process group created for this managed child, then reap it."""
    if process is None:
        return None
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=timeout)
    # A child may exit without cleaning its Agent descendants.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.poll()


def selected_paths(root, ids=None):

    from ..data.frozen import select_sequences
    return select_sequences(root, ids)


def child_arguments(argv, dataset, output):
    result = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in {'--dataset', '--output', '--sequence', '--run-directory'}:
            skip = True
            continue
        if arg == '--all-sequences' or any(arg.startswith(k + '=') for k in ('--dataset', '--output', '--sequence', '--run-directory')):
            continue
        result.append(arg)
    return [*result, '--dataset', str(dataset), '--output', str(output)]


def aggregate_runs(records):
    valid = bool(records) and all(r.get('exit_code') == 0 and r.get('summary', {}).get('evaluation_valid') is True for r in records)
    scores = [r.get('summary', {}).get('task_score') for r in records]
    # Zero-denominator sequences remain visible but contribute no numeric score.
    numeric = [score for score in scores if type(score) in (int,float) and math.isfinite(score) and 0 <= score <= 1]
    versions={r.get('summary',{}).get('metric_version','binary_success_v1') for r in records}
    metrics_complete = len(versions) == 1 and all(
        r.get('summary',{}).get('metrics_complete',True)
        and ((type(score) in (int,float) and math.isfinite(score) and 0 <= score <= 1)
             or (score is None and r.get('summary',{}).get('tasks_evaluated') == 0))
        for r,score in zip(records,scores))
    success_rates=[r.get('summary',{}).get('task_success_rate',r.get('summary',{}).get('task_score')) for r in records]
    success_rates=[r for r in success_rates if type(r) in (int,float) and math.isfinite(r) and 0 <= r <= 1]
    score=sum(numeric)/len(numeric) if valid and metrics_complete and numeric else None
    return {'schema_version': 1, 'aggregation': 'mean_of_sequence_task_scores',
            'evaluation_valid': valid, 'sequence_count': len(records),
            'sequences_with_score': len(numeric),
            'metric_version':next(iter(versions)) if len(versions)==1 else 'mixed_incomparable',
            'metrics_complete':metrics_complete,'task_score':score,'score_max':100,
            'score_100':score*100 if score is not None else None,
            'task_success_rate':sum(success_rates)/len(success_rates) if valid and success_rates else None,
            'sequences': records}


def run_collection(paths, output, argv, *, dry_run=False):
    from ..scenes.dataset_loader import BenchmarkDataset
    datasets = [BenchmarkDataset(path) for path in paths]
    if dry_run:
        for data in datasets:
            print(json.dumps({'dataset': str(data.root), **data.sequence_identity(),
                'sessions': sum(len(data.sessions(sid)) for sid in data.scene_ids())}, ensure_ascii=False))
        return 0
    root = Path(output) / ('sequences_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'manifest.json', {'artifact_schema':'benchmark_results','schema_version':2,'kind':'collection',
        'runs':[{'identity':d.sequence_identity(), 'dataset':str(d.root),
                 'directory':'sequences/'+safe_id(d.manifest['sequence_id'])} for d in datasets]})
    update_status(root, phase='running', execution='running', summary='pending', cleanup='pending')
    records = []
    previous_term = signal.getsignal(signal.SIGTERM)
    def terminate_batch(signum, frame):
        raise KeyboardInterrupt('sequence batch terminated')
    signal.signal(signal.SIGTERM, terminate_batch)
    try:
        for data in datasets:
            record = {'identity': data.sequence_identity(), 'dataset': str(data.root), 'exit_code': None}
            records.append(record)
            target = root / 'sequences' / safe_id(data.manifest['sequence_id'])
            command = [sys.executable, '-m', 'roboodyssey', 'run', *child_arguments(argv, data.root, target.parent),
                       '--run-directory', str(target)]
            process = subprocess.Popen(command, start_new_session=True)
            try:
                record['exit_code'] = process.wait()
            finally:
                stop_owned(process)
            if (target/'summary.json').is_file():
                record['result_directory'] = str(target)
                record['summary'] = json.loads((target/'summary.json').read_text())
            # A failed child remains an explicit invalid result; it cannot be
            # omitted from the aggregate or contaminate the next fresh process.
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        summary = aggregate_runs(records)
        if len(records) != len(datasets):
            summary.update(evaluation_valid=False, task_score=None, score_100=None,
                           task_success_rate=None, metrics_complete=False, interrupted=True)
        (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        update_status(root, phase='completed' if summary['evaluation_valid'] else 'failed',
                      execution='completed' if len(records)==len(datasets) else 'interrupted', summary='completed', cleanup='completed')
        print('Sequence results:', root)
    return 0 if summary['evaluation_valid'] else 1
