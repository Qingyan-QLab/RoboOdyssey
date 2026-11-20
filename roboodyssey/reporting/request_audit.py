"""CPU-only, private v5 failure/repair review from durable logs."""
from collections import Counter
import html
import json
import os
from pathlib import Path
from urllib.parse import quote
from ..artifacts.layout import iter_sessions, session_file, reports_dir, read_json_lines


def lines(path):
    """Read audit rows, preserving malformed lines as explicit error records."""
    return [record for _, record in read_json_lines(path, errors='record', missing_ok=True)]


def write_request_audit(root, output=None):
    root = Path(root)
    output = Path(output) if output is not None else reports_dir(root)
    rows, errors, mutations = [], [], []
    for scene_id, sid, path in iter_sessions(root):
        def records(name):
            valid = []
            for item in lines(session_file(path,name)):
                if item.get('log_error'):
                    errors.append({'scene_id': scene_id, 'session_id': sid, **item})
                else:
                    valid.append(item)
            return valid
        indexes = {e.get('event_id'): e for e in records('video_index.jsonl')}
        tasks = records('runtime_tasks.jsonl')
        outcomes = {e.get('task_id'): e for e in records('request_outcomes.jsonl')}
        dispatched = {e.get('task_id'): e.get('payload', {}) for e in records('agent_io.jsonl')
                      if e.get('direction') == 'request' and e.get('request_type') == 'step'}
        for item in records('runtime_events.jsonl') + records('integration_errors.jsonl'):
            if item.get('event') == 'integration_error' or 'error' in item:
                errors.append({'scene_id': scene_id, 'session_id': sid, **item})
        for item in records('runtime_mutations.jsonl'):
            mutations.append({'scene_id': scene_id, 'session_id': sid, **item})
        for task in tasks:
            if task.get('log_error'):
                errors.append({'scene_id': scene_id, 'session_id': sid, **task}); continue
            tid = task['original']['task_id']
            repair = task['repair']
            record = repair['repair_record']
            outcome = outcomes.get(tid)
            index = indexes.get(task.get('decision_event_id'))
            video = None
            if index and session_file(path,'review.mp4').is_file():
                video = {'path': Path(os.path.relpath(session_file(path,'review.mp4'),output)).as_posix(), 'seconds': index['video_seconds']}
            rows.append({'scene_id': scene_id, 'session_id': sid, 'task_id': tid,
                'original_instruction': task['original']['instruction'],
                'resolved_instruction': (repair.get('public_task') or {}).get('instruction'),
                'issued_instruction': dispatched.get(tid, {}).get('instruction'),
                'dispatch_attempted': tid in dispatched,
                'repair_status': repair['status'], 'repair_record': record,
                'original_ground_truth': task.get('original_ground_truth'),
                'runtime_ground_truth': repair.get('runtime_ground_truth'),
                'source_context': task.get('source_context', []),
                'outcome': outcome, 'video': video, 'decision_event_id': task.get('decision_event_id')})
    by_key = {(r['scene_id'], r['session_id'], r['task_id']): r for r in rows}
    for row in rows:
        for source in row['source_context']:
            prior = by_key.get((row['scene_id'], source['session_id'], source['task_id']))
            source['actual_state_changes'] = ((prior or {}).get('outcome') or {}).get('state_changes')
    report = {'schema_version': 1, 'private': True, 'tasks': rows, 'mutations': mutations, 'integration_errors': errors,
              'counts': dict(Counter(r['repair_status'] for r in rows)),
              'incomplete_tasks': sum(r['outcome'] is None or bool(r['outcome'].get('interrupted')) for r in rows),
              'interpretation': 'Source context identifies original dependencies, not inferred causality. Proven exclusion causes are in repair_record.failure_causes. This diagnostic report does not replace official metrics.'}
    output.mkdir(parents=True, exist_ok=True)
    (output/'request_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    esc = lambda v: html.escape(str(v))
    detail = lambda label, v: '<details><summary>'+esc(label)+'</summary><pre>'+esc(json.dumps(v,ensure_ascii=False,indent=2))+'</pre></details>'
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>v5 失败与修补审阅</title>',
             '<style>body{font:16px system-ui;margin:24px;max-width:1300px}section{border:1px solid #ccd;padding:18px;margin:16px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#057}</style>',
             '<h1>v5 失败、真实状态与后续修补</h1><p>私有报告。源任务关联不等于失败因果；未完成或接入错误不能算正常有效评分。下发指调用 Agent 前记录的请求，不保证远端接收成功。视频是事件采样，真实耗时见索引。</p>',
             detail('统计及接入错误', {k: report[k] for k in ('counts','incomplete_tasks','integration_errors')})]
    for row in rows:
        ref = '/'.join(row[k] for k in ('scene_id','session_id','task_id'))
        parts.append('<section id="'+esc(ref)+'"><h2>'+esc(ref)+'</h2><p>原请求：'+esc(row['original_instruction'])+
                     '</p><p>修补后：'+esc(row['resolved_instruction'] or '无可下发请求')+
                     '</p><p>实际尝试下发：'+esc(row['issued_instruction'] or '未下发')+'</p><p>状态：'+esc(row['repair_status'])+'</p>')
        if row['video']:
            url = quote(row['video']['path'], safe='/')+'#t='+str(row['video']['seconds'])
            parts.append('<a href="'+esc(url)+'">视频 '+esc(row['video']['seconds'])+' 秒</a>')
        for source in row['source_context']:
            source_ref = '/'.join((row['scene_id'],source['session_id'],source['task_id']))
            parts.append('<p>来源：<a href="#'+esc(source_ref)+'">'+esc(source_ref)+'</a></p>')
        for label,key in [('失败来源与实际变化','source_context'),('修补依据及转换','repair_record'),
                          ('原 GT','original_ground_truth'),('实际 GT','runtime_ground_truth'),('后续结果及状态变化','outcome')]:
            parts.append(detail(label,row[key]))
        parts.append('</section>')
    parts.append(detail('系统变化与实际回读',mutations)+'</html>')
    (output/'request_audit.html').write_text(''.join(parts),encoding='utf-8')
    return report
