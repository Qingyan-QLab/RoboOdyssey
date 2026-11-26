"""Pair explicit run attempts without mixing data, rules or evaluation budgets.

Each group is a model/configuration label. Repeated sequence runs need explicit
replicate labels (usually seeds); this module never selects the best retry.
Saved results are compared, not rerun or silently rescored.
"""
from collections import defaultdict
from copy import deepcopy
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from ..artifacts.layout import (manifest, read_json, sha256_file, write_json, json_digest,
                               iter_sessions, session_file, read_json_lines)


def _profile_hash(value):
    # Match the public run writer, including its JSON Unicode encoding.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _history_spans(root, tasks):
    """Optional exact request gaps from saved source bindings; no inferred bins."""
    ranks = {key: index for index, key in enumerate(tasks)}
    spans = {key: None for key in tasks}
    hashes = {}
    for _, session, directory in iter_sessions(root):
        path = session_file(directory, 'runtime_tasks.jsonl')
        if not path.is_file():
            continue
        hashes[str(path.relative_to(root))] = sha256_file(path)
        for _, row in read_json_lines(path):
            key = session+'/'+row['original']['task_id']
            if key not in ranks:
                continue
            parents = [source['session_id']+'/'+source['task_id'] for source in row.get('source_context', [])]
            if all(parent in ranks and ranks[parent] < ranks[key] for parent in parents):
                spans[key] = max((ranks[key]-ranks[parent]-1 for parent in parents), default=0)
    return spans, hashes


def _load(value):
    descriptor = {'run': value} if isinstance(value, (str, Path)) else value
    root = Path(descriptor['run']).resolve()
    result = {'run': str(root), 'replicate': str(descriptor.get('replicate', 'default')), 'issues': []}
    try:
        layout = manifest(root)
        if layout is None or layout.get('kind') != 'run':
            raise ValueError('select explicit single-run attempts, not an experiment directory')
        identity = layout['sequence_identity']
        result['identity'] = deepcopy(identity)
        result['key'] = (identity['scene_id'], identity['sequence_id'], result['replicate'])
        config, summary = read_json(root/'run_config.json'), read_json(root/'summary.json')
        if not isinstance(config, dict) or not isinstance(summary, dict):
            raise ValueError('config and summary must be JSON objects')
        result['source_sha256'] = {name: sha256_file(root/name)
                                  for name in ('manifest.json', 'run_config.json', 'summary.json')}
        if config.get('sequence_identity') != identity or summary.get('sequence_identity') != identity:
            raise ValueError('run/config/summary sequence identities disagree')
        profile = config.get('evaluation_profile')
        required = {'dataset_manifest_sha256', 'dataset_tree_sha256', 'sequence_identity',
                    'max_steps_per_task', 'agent_timeout_seconds', 'camera_resolution',
                    'observation_policy', 'metric_policy', 'protocol_version', 'robot_body',
                    'horizontal_fov_degrees', 'robot_radius_m', 'camera_height_m',
                    'success_rule_version', 'runtime_contract', 'reference_models', 'interaction_model', 'lighting'}
        if not isinstance(profile, dict) or required - profile.keys():
            raise ValueError('missing complete public evaluation profile')
        if _profile_hash(profile) != config.get('evaluation_profile_sha256'):
            raise ValueError('evaluation profile hash mismatch')
        if profile['sequence_identity'] != identity:
            raise ValueError('profile sequence identity mismatch')
        for field in ('dataset_manifest_sha256', 'dataset_tree_sha256', 'metric_policy',
                      'observation_policy', 'max_steps_per_task', 'agent_timeout_seconds'):
            if field not in config or config[field] != profile[field]:
                raise ValueError('profile/config mismatch: '+field)
        context_fields = ('motion', 'camera', 'recording', 'holoagent_sensors', 'step_delay_seconds', 'official')
        if any(key not in config for key in context_fields):
            raise ValueError('missing action, sensor or diagnostic configuration')
        if not isinstance(config['motion'], dict) or not isinstance(config['camera'], dict):
            raise ValueError('motion and camera must be JSON objects')
        if (config['motion'].get('interaction_model') != profile['interaction_model']
                or layout['runtime_contract'] != profile['runtime_contract']
                or config.get('lighting') != profile['lighting']):
            raise ValueError('action, lighting or runtime contract identity mismatch')
        context = {key: config[key] for key in context_fields}
        context['profile'] = {key: val for key, val in profile.items() if key not in {
            'dataset_manifest_sha256', 'dataset_tree_sha256', 'sequence_identity', 'scene_ids'}}
        context['runtime_contract'] = layout['runtime_contract']
        context['metric_version'] = summary.get('metric_version')
        if not context['metric_version']:
            raise ValueError('missing metric version')
        result.update(profile_sha256=config['evaluation_profile_sha256'],
                      condition_sha256=json_digest(context, allow_nan=False),
                      official=bool(config['official']), score_100=summary.get('score_100'),
                      task_success_rate=summary.get('task_success_rate'),
                      tasks_total=summary.get('tasks_total'), tasks_excluded=summary.get('tasks_excluded'))
        sessions = summary.get('sessions', [])
        expected_sessions = [(entry['scene_id'], sid) for entry in layout['sequences'] for sid in entry['session_ids']]
        observed_sessions = [(session.get('scene_id'), session.get('session_id')) for session in sessions]
        if len(set(observed_sessions)) != len(observed_sessions) or observed_sessions != expected_sessions:
            raise ValueError('summary session population or order differs from the manifest')
        tasks = {}
        for session in sessions:
            for task in session['task_results']:
                key = session['session_id']+'/'+task['task_id']
                if key in tasks:
                    raise ValueError('duplicate task in summary')
                tasks[key] = task
        result['tasks'] = tasks
        result['history_spans'], trace_hashes = _history_spans(root, tasks)
        result['source_sha256'].update(trace_hashes)
        if len(tasks) != summary.get('tasks_total'):
            raise ValueError('summary task count mismatch')
        from ..evaluation.requests import request_totals
        totals = request_totals(list(tasks.values()), valid=bool(summary.get('evaluation_valid')))
        for field in ('tasks_evaluated', 'tasks_excluded', 'metrics_complete', 'metric_version',
                      'score_100', 'task_success_rate'):
            expected, recorded = totals.get(field), summary.get(field)
            equal = math.isclose(expected, recorded, rel_tol=1e-12, abs_tol=1e-9) if (
                _finite(expected) and _finite(recorded)) else expected == recorded
            if not equal:
                raise ValueError('aggregate disagrees with task evidence: '+field)
        if not summary.get('evaluation_valid'):
            result['issues'].append('invalid_or_incomplete_execution')
        if not summary.get('metrics_complete') or not _finite(result['score_100']):
            result['issues'].append('missing_formal_metrics')
        if not _finite(result['task_success_rate']):
            result['issues'].append('missing_success_rate')
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        result['issues'].append(str(exc))
    return result


def _task_pairs(left, right):
    result = []
    for key in sorted(set(left['tasks']) | set(right['tasks'])):
        a, b = left['tasks'].get(key), right['tasks'].get(key)
        row = {'task': key, 'status': 'paired', 'score_delta': None, 'success_delta': None}
        if a is None or b is None:
            row['status'] = 'missing_task'
        elif a.get('scored', True) != b.get('scored', True):
            row['status'] = 'different_exclusion'
        elif not a.get('scored', True):
            row['status'] = 'both_excluded'
        else:
            sa, sb = (item.get('evaluation', {}).get('score') for item in (a, b))
            if not _finite(sa) or not _finite(sb):
                row['status'] = 'missing_metric'
            else:
                row['score_delta'] = sb-sa
                row['success_delta'] = int(b.get('success') is True)-int(a.get('success') is True)
        for name, task in (('baseline', a), ('candidate', b)):
            row[name] = None if task is None else {
                'success': task.get('success'), 'scored': task.get('scored', True),
                'reason_code': task.get('reason_code'),
                'score': task.get('evaluation', {}).get('score')}
        task = a or b
        row['capabilities'] = task.get('capabilities', [])
        row['task_type'] = task.get('task_type', 'unknown')
        left_span, right_span = left['history_spans'].get(key), right['history_spans'].get(key)
        row['history_intervening_requests'] = left_span if left_span == right_span else None
        result.append(row)
    return result


def compare_runs(groups, *, baseline=None):
    """Compare label -> explicit attempts; retain every missing/failed attempt.

    Entries are paths or ``{'run': path, 'replicate': 'seed-1'}``. Replicate
    labels pair declared experiments; they do not certify simulator determinism.
    Different condition profiles are reported separately. Aggregates are null
    if any planned pair is missing, incompatible or invalid.
    """
    if not isinstance(groups, dict) or len(groups) < 2 or any(not runs for runs in groups.values()):
        raise ValueError('provide at least two nonempty labelled groups')
    baseline = baseline or next(iter(groups))
    if baseline not in groups:
        raise ValueError('unknown baseline label')
    loaded, failures = {}, []
    for label, paths in groups.items():
        loaded[label] = {}
        for path in paths:
            run = _load(path)
            if 'key' not in run:
                failures.append({'label': label, **run})
                continue
            if run['key'] in loaded[label]:
                raise ValueError('duplicate sequence/replicate in '+label+'; select explicit attempts or distinct replicates')
            loaded[label][run['key']] = run
    comparisons = []
    for label, candidates in loaded.items():
        if label == baseline:
            continue
        pairs, condition_groups = [], defaultdict(list)
        for key in sorted(set(loaded[baseline]) | set(candidates)):
            left, right = loaded[baseline].get(key), candidates.get(key)
            row = {'scene_id': key[0], 'sequence_id': key[1], 'replicate': key[2],
                   'issues': [], 'score_delta': None, 'task_success_rate_delta': None, 'tasks': []}
            for name, run in (('baseline', left), ('candidate', right)):
                row[name] = None if run is None else {k: v for k, v in run.items() if k not in ('key', 'tasks', 'history_spans')}
                row['issues'].extend([name+': '+issue for issue in run['issues']] if run else [name+': missing_run'])
            if not row['issues']:
                if left['identity'] != right['identity'] or left['profile_sha256'] != right['profile_sha256']:
                    row['issues'].append('data_or_evaluation_profile_mismatch')
                if left['condition_sha256'] != right['condition_sha256']:
                    row['issues'].append('metric_action_sensor_or_budget_mismatch')
            if not row['issues']:
                row['tasks'] = _task_pairs(left, right)
                if any(task['status'] == 'missing_task' for task in row['tasks']):
                    row['issues'].append('different_task_population')
            row['comparable'] = not row['issues']
            if row['comparable']:
                row.update(score_delta=right['score_100']-left['score_100'],
                           task_success_rate_delta=right['task_success_rate']-left['task_success_rate'],
                           condition_sha256=left['condition_sha256'])
                condition_groups[left['condition_sha256']].append(row)
            pairs.append(row)
        complete = bool(pairs) and all(pair['comparable'] for pair in pairs) and not any(
            error['label'] in (baseline, label) for error in failures)
        summaries = []
        for condition, rows in sorted(condition_groups.items()):
            capability_groups, task_type_groups, history_groups = defaultdict(list), defaultdict(list), defaultdict(list)
            for row in rows:
                for task in row['tasks']:
                    if task['score_delta'] is not None:
                        task_type_groups[task['task_type']].append(task['score_delta'])
                        span = task['history_intervening_requests']
                        history_groups['unknown' if span is None else str(span)].append(task['score_delta'])
                        for capability in task['capabilities']:
                            capability_groups[capability].append(task['score_delta'])
            grouped = lambda data: {key: {'paired_tasks': len(values),
                'mean_score_delta': sum(values)/len(values) if complete else None} for key, values in sorted(data.items())}
            summaries.append({'condition_sha256': condition, 'paired_sequences': len(rows),
                'mean_score_delta': sum(row['score_delta'] for row in rows)/len(rows) if complete else None,
                'mean_task_success_rate_delta': sum(row['task_success_rate_delta'] for row in rows)/len(rows) if complete else None,
                'by_capability': grouped(capability_groups), 'by_task_type': grouped(task_type_groups),
                'by_history_intervening_requests': grouped(history_groups)})
        comparisons.append({'candidate': label, 'population_complete': complete,
                            'planned_pairs': len(pairs), 'comparable_pairs': sum(p['comparable'] for p in pairs),
                            'conditions': summaries, 'pairs': pairs})
    return {'schema_version': 1, 'format': 'benchmark_paired_comparison_v1',
            'baseline': baseline, 'comparable': all(c['population_complete'] for c in comparisons),
            'method': {'difference': 'candidate_minus_baseline', 'aggregate_unit': 'sequence_replicate',
                       'aggregation': 'unweighted_mean_within_identical_conditions', 'uncertainty_estimate': None,
                       'task_groups': 'task_weighted; capability groups overlap',
                       'history_groups': 'exact intervening request count from saved runtime source bindings; absent or differing bindings are unknown',
                       'exclusions': 'sequence scores keep each run denominator; differing task exclusions are reported without task delta',
                       'retry_selection': 'explicit_attempts_only', 'rescored': False},
            'unreadable_runs': failures, 'comparisons': comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, help='JSON {groups: {label: [run or {run, replicate}]}}, baseline optional')
    parser.add_argument('--output', type=Path, required=True, help='new report directory')
    args = parser.parse_args(argv)
    config = read_json(args.config)
    groups = deepcopy(config['groups'])
    for label, values in groups.items():
        for index, value in enumerate(values):
            value = {'run': value} if isinstance(value, str) else value
            value['run'] = str((args.config.resolve().parent/Path(value['run'])).resolve())
            values[index] = value
    report = compare_runs(groups, baseline=config.get('baseline'))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'comparison.json', report)
    with (args.output/'pairs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['candidate', 'scene_id', 'sequence_id', 'replicate',
                                                  'comparable', 'score_delta', 'task_success_rate_delta', 'issues'])
        writer.writeheader()
        for comparison in report['comparisons']:
            for pair in comparison['pairs']:
                writer.writerow({'candidate': comparison['candidate'], **{
                    key: pair[key] for key in writer.fieldnames if key != 'candidate'}})
    print(args.output/'comparison.json')
    return 0 if report['comparable'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
