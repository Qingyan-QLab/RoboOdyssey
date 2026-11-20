"""CPU reports for current benchmark runs and sequence collections."""
import argparse
from datetime import datetime
import html
import json
from pathlib import Path
from urllib.parse import quote

from ..artifacts.layout import manifest, iter_sessions, local_path, write_json, read_json
from .request_audit import write_request_audit


def analyze(root, output=None, *, near_margin_m=.5, rescore=False, reference_dataset=None):
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root/'reports'/('analysis_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'source': str(root), 'output': str(output), 'ok': True, 'reports': [], 'errors': [], 'notes': []}
    def add(label, path):
        report['reports'].append({'label': label, 'path': str(path.relative_to(output))})
    try:
        layout = manifest(root)
        if layout is None:
            raise ValueError('run is missing its benchmark artifact manifest')
        if layout.get('kind') in {'collection', 'experiment'}:
            if reference_dataset:
                raise ValueError('reference-dataset reconstruction requires a single run directory')
            for i, entry in enumerate(layout['runs']):
                child = local_path(root, entry['directory'])
                try:
                    result = analyze(child, output/f'sequence_{i:03d}', near_margin_m=near_margin_m, rescore=rescore)
                    add(entry.get('identity', {}).get('sequence_id') or child.name, output/f'sequence_{i:03d}'/'index.html')
                    report['ok'] &= result['ok']
                except Exception as exc:
                    report['errors'].append({'run': str(child), 'error': str(exc)})
            if (root/'summary.json').exists():
                write_json(output/'aggregate.json', read_json(root/'summary.json'))
                add('序列汇总（保留原运行有效性）', output/'aggregate.json')
        else:
            sessions = list(iter_sessions(root))
            if not sessions:
                raise ValueError('no sessions found in this run')
            if layout.get('runtime_contract') != {'name': 'contextual_requests', 'version': 3}:
                raise ValueError('analysis requires contextual_requests v3 evidence')
            audit = write_request_audit(root, output/'repairs')
            add('任务修补、实际 GT 与视频审计', output/'repairs/request_audit.html')
            report['ok'] &= not audit['integration_errors'] and not audit['incomplete_tasks']
            try:
                from .task_diagnostics_report import write_task_scores
                diagnostic = write_task_scores(root, output/'scores', near_margin_m=near_margin_m,
                    rescore=rescore, reference_dataset=reference_dataset)
                add('正式任务分、导航进度与 reset 诊断', output/'scores/report.html')
                report['ok'] &= not diagnostic['issues']
            except Exception as exc:
                report['errors'].append({'stage': 'scores', 'error': str(exc)})
            if (root/'status.json').exists():
                write_json(output/'run_status.json', read_json(root/'status.json'))
                add('原运行状态（不被离线分析覆盖）', output/'run_status.json')
    except Exception as exc:
        report['errors'].append({'stage': 'analysis', 'error': f'{type(exc).__name__}: {exc}'})
    report['ok'] &= not report['errors']
    write_json(output/'analysis.json', report)
    esc = html.escape
    links = ''.join('<li><a href="'+esc(quote(r['path'], safe='/'))+'">'+esc(r['label'])+'</a></li>' for r in report['reports'])
    (output/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Benchmark 分析</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:32px}li{margin:16px 0}pre{white-space:pre-wrap}</style>'
        '<h1>Benchmark 分析</h1><p>来源：'+esc(str(root))+'</p><p>分析证据检查：'+('通过' if report['ok'] else '存在问题')+
        '。这不改变原运行的完成状态或官方评测资格。</p><ul>'+links+'</ul><pre>'+esc(json.dumps({'notes':report['notes'],'errors':report['errors']},ensure_ascii=False,indent=2))+'</pre></html>', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, help='new directory; defaults to a timestamp under reports/')
    parser.add_argument('--near-margin-m', type=float, default=.5)
    parser.add_argument('--rescore', action='store_true', help='explicitly rescore original evidence with the current metric; preserve original logs')
    parser.add_argument('--reference-dataset', type=Path, help='reconstruct missing maps from the verified historical dataset and actual states (single run only)')
    args = parser.parse_args(argv)
    if args.reference_dataset and not args.rescore:
        parser.error('--reference-dataset requires --rescore')
    report = analyze(args.run, args.output, near_margin_m=args.near_margin_m,
        rescore=args.rescore, reference_dataset=args.reference_dataset)
    print(Path(report['output'])/'index.html')
    return 0 if report['ok'] else 1
