#!/usr/bin/env python3
"""Offline template checks with archived geometry and injected failures; no physics."""
import argparse
from collections import Counter
from copy import deepcopy
import html
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roboodyssey.data.frozen import LegacyData as Dataset, read
from roboodyssey.artifacts.layout import write_json as write
from roboodyssey.legacy.task_repair import instantiate_task
from roboodyssey.legacy.task_history import execution_events


def check(dataset, output=None):
    ds = Dataset(dataset)
    ds.load_result()
    cache = read(ds.path('runtime_scene_cache'))
    metadata = read(ds.path('scene_metainfo'))
    scenarios = {}
    rows = []
    for scenario in ('planned_success', 'first_transfer_failed', 'first_transfer_held'):
        state = {'objects': {o['object_id']: deepcopy(o) for o in metadata['objects']}}
        history, records = [], []
        injected = False
        for entry in ds.manifest['sessions']:
            sid = entry['session_id']
            if 'mutation' in entry:
                mutation = read(ds.session_path(entry, 'mutation'))
                change = mutation['changes'][0]
                obj = state['objects'][change['object_id']]
                obj.update(geometry=deepcopy(mutation['geometry_after']),
                           room_id=change['relation_after']['room_id'], held=False, support_object_id=None)
                if state.get('held_object') == change['object_id']:
                    state['held_object'] = None
                history.append({'session_id': sid, 'kind': 'hidden_change', 'agent_information': 'hidden',
                                'mutation': mutation})
            templates = {t['task_id']: t for t in read(ds.session_path(entry, 'task_templates'))['tasks']}
            truths = {t['task_id']: t for t in read(ds.session_path(entry, 'ground_truth'))['tasks']}
            for task in read(ds.session_path(entry, 'session'))['tasks']:
                tid = task['task_id']; template = templates[tid]; gt = truths[tid]
                before = deepcopy(state)
                result = instantiate_task(template, cache, state, history, task=task, ground_truth=gt)
                record = {'session_id': sid, **result['repair_record']}
                success = result['status'] == 'ready'
                actions = []
                if success and task['task_type'] == 'manipulation':
                    oid = template['target_object_id']
                    obj = state['objects'][oid]
                    if scenario != 'planned_success' and not injected:
                        injected = True; success = False
                        record['injected_failure'] = scenario
                        if scenario == 'first_transfer_held':
                            state['held_object'] = oid; obj['held'] = True
                    else:
                        destination = result['runtime_ground_truth']['expected_outcome']['state_changes'][0]['relation_after']['room_id']
                        # Use archived placement only for this synthetic replay.
                        obj.update(geometry=deepcopy(gt['expected_outcome']['placement_geometry']),
                                   room_id=destination, held=False, support_object_id=None)
                        state['held_object'] = None
                        actions = [{'action': {'action': 'place'}, 'result': {'success': True, 'object_id': oid,
                                                                                         'room_id': destination}}]
                history.extend(execution_events(sid, template, task, before, state,
                    {'success': success if result['status'] == 'ready' else None,
                     'dispatched': result['status'] == 'ready',
                     'termination_reason': (record.get('injected_failure') or 'succeeded') if result['status'] == 'ready' else result['status'],
                     'reason_code': record.get('reason_code') or record.get('injected_failure')}, actions))
                records.append(record)
                cells = [scenario, sid + '/' + tid, record['original_text'], record.get('rendered_text') or '未下发',
                         record['status'], ', '.join(record['reasons'])]
                rows.append('<tr>' + ''.join('<td>' + html.escape(str(c)) + '</td>' for c in cells) + '</tr>')
        scenarios[scenario] = {'status_counts': dict(Counter(r['status'] for r in records)),
                               'changed_tasks': sum(r['changed'] for r in records), 'tasks': records}
    report = {'schema_version': 1, 'scope': 'offline_synthetic_state_replay',
              'physics_execution_verified': False, 'scenarios': scenarios}
    output = Path(output) if output else Path(__file__).resolve().parents[1] / 'outputs/runtime_repair_validation' / ds.root.name
    write(output / 'report.json', report)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>任务修补离线验证</title>'
        '<style>body{font:14px sans-serif;margin:24px}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}</style>'
        '<h1>任务修补离线验证</h1><p>计划几何与注入失败的合成状态重放；不是实际 Agent 或物理执行结果。</p>'
        '<table><tr><th>场景</th><th>任务</th><th>原文</th><th>修补后</th><th>状态</th><th>原因</th></tr>'
        + ''.join(rows) + '</table>', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    report = check(args.dataset, args.output)
    for name, value in report['scenarios'].items():
        print(name, value['status_counts'], 'changed:', value['changed_tasks'])
    if set(report['scenarios']['planned_success']['status_counts']) != {'ready'}:
        raise SystemExit('Some baseline tasks could not be instantiated; see report.json')


if __name__ == '__main__': main()
