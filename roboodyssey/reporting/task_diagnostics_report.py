"""Portable HTML, CSV and JSON views of task diagnostics. No external assets."""
import csv
import html
import json
import os
from pathlib import Path
from urllib.parse import quote


def esc(value):
    return html.escape(str(value))


def fmt(value, digits=2):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def pct(value):
    return 'N/A' if value is None else f'{100 * value:.2f}%'


def ref(row):
    return '/'.join(row[k] for k in ('scene_id', 'session_id', 'task_id'))


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+esc(h)+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>'+''.join('<td>'+cell+'</td>' for cell in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def detail(label, value):
    return '<details><summary>'+esc(label)+'</summary><pre>'+esc(json.dumps(value, ensure_ascii=False, indent=2))+'</pre></details>'


def video_link(video, output, seconds=None):
    if not video:
        return 'N/A'
    seconds = video['seconds'] if seconds is None else seconds
    if seconds is None:
        return 'N/A'
    url = quote(os.path.relpath(video['file'], output), safe='/') + '#t=' + str(seconds)
    return '<a href="'+esc(url)+'">视频 '+fmt(seconds, 1)+' s</a>'


def distance_chart(goal, resets):
    values = goal['distance_series_m']
    width, height, pad = 800, 130, 24
    ceiling = max(max(values), goal['tolerance_m'], .1) * 1.08
    x = lambda i: pad + i * (width - 2 * pad) / max(1, len(values) - 1)
    y = lambda v: height - pad - v * (height - 2 * pad) / ceiling
    # Keep every logged sample, including extrema and both sides of resets.
    points = ' '.join(f'{x(i):.2f},{y(v):.2f}' for i, v in enumerate(values))
    threshold = y(goal['tolerance_m'])
    markers = ''.join(f'<line x1="{x(2*r["task_step"]+1):.2f}" x2="{x(2*r["task_step"]+1):.2f}" y1="10" y2="{height-pad}" stroke="#bbb"><title>reset 动作 {r["task_step"]+1}</title></line>' for r in resets)
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="目标距离随动作变化">{markers}<line x1="{pad}" x2="{width-pad}" y1="{threshold:.2f}" y2="{threshold:.2f}" stroke="#29834b" stroke-dasharray="5 4"/><polyline points="{points}" fill="none" stroke="#2469ac" stroke-width="1.5"/><text x="{pad}" y="12">{ceiling:.2f} m</text><text x="{pad}" y="{height-4}">起点</text><text x="{width-80}" y="{height-4}">任务结束</text></svg>'


def write_csv(path, columns, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            # Guard spreadsheet formula interpretation in public task text.
            writer.writerow({k: ('N/A' if row.get(k) is None else
                "'" + row[k] if isinstance(row[k], str) and row[k].startswith(('=', '+', '-', '@')) else row[k]) for k in columns})


def write_diagnostics(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'scores.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    task_rows, goal_rows, reset_rows, transfer_rows = [], [], [], []
    for row in report['tasks']:
        nav = row['navigation'] or {}
        motion = nav.get('motion', {})
        evaluation = row.get('evaluation') or {}
        path_metric = evaluation.get('navigation') or {}
        transfer_details = [{k:v for k,v in g.items() if k != 'timeline'} for g in evaluation.get('transfers', [])]
        transfer_rows.extend({'task':ref(row), **g} for g in transfer_details)
        task_rows.append({**{k: row[k] for k in ('order', 'scene_id', 'session_id', 'task_id', 'instruction', 'formal_task_score', 'excluded', 'success', 'reason_code', 'result_explanation', 'steps')},
            'target_names': ' → '.join(row['target_names']),
            'runtime_gt': json.dumps(row['runtime_gt'], ensure_ascii=False),
            'success_conditions': '；'.join(row['success_conditions']),
            'goal_results': json.dumps(row['goal_results'], ensure_ascii=False),
            'score_max':100,'metric_version':row.get('metric_version','unavailable'),
            'completion_score':row.get('completion_score'),
            'original_task_score':row.get('original_task_score'),
            'original_metric_version':row.get('original_metric_version'),
            'transfer_metrics': json.dumps(transfer_details, ensure_ascii=False),
            **{'metric_'+k:path_metric.get(k) for k in ('reference_length_m','remaining_route_m','charged_length_m','reset_cost_m','progress_score','best_progress_score','path_efficiency')},
            **{'measured_'+k:v for k,v in evaluation.get('measurements',{}).items()},
            'step_efficiency_score':evaluation.get('step_efficiency_score'),
            'time_efficiency_score':evaluation.get('time_efficiency_score'),
            **{k: nav.get(k) for k in ('final_progress_score', 'best_progress_score', 'goals_completed_in_order', 'goals_total', 'walking_efficiency')},
            **motion, 'categories': '；'.join(nav.get('categories', []))})
        for goal in nav.get('goals', []):
            goal_rows.append({'task': ref(row), **{k: v for k, v in goal.items() if k not in {'distance_series_m', 'target_box_xy'}}})
        for reset in nav.get('reset_recoveries', []):
            reset_rows.append({'task': ref(row), **{k: v for k, v in reset.items() if k not in {'before_window', 'recovery_window'}},
                **{'pre_'+k: v for k, v in reset['before_window'].items()},
                **{'recovery_'+k: v for k, v in reset['recovery_window'].items()}})
    for name, rows in (('tasks.csv', task_rows), ('navigation_goals.csv', goal_rows),
                       ('resets.csv', reset_rows), ('transfer_goals.csv', transfer_rows)):
        columns = list(dict.fromkeys(k for row in rows for k in row)) or ['task']
        write_csv(output/name, columns, rows)
    summary = report['summary']
    session_rows = [{'session':key,**value} for key,value in summary.get('by_session',{}).items()]
    if session_rows:
        write_csv(output/'sessions.csv',list(session_rows[0]),session_rows)
    grid_summary=summary.get('navigation_metric',{})
    transfer_summary=summary.get('transfer_metric',{})
    metric_table=[['全部计分任务均分',fmt(summary['formal_rule_average'])+' / 100'],
        ['完整任务成功率',pct(summary.get('task_success_rate'))],
        ['导航任务均分',fmt(grid_summary.get('score'))+' / 100'],
        ['导航最终网格进度',fmt(grid_summary.get('progress_score'))+' / 100'],
        ['导航最佳网格进度',fmt(grid_summary.get('best_progress_score'))+' / 100'],
        ['导航路径效率均值',pct(grid_summary.get('path_efficiency'))],
        ['搬运目标均分',fmt(transfer_summary.get('score'))+' / 100'],
        ['搬运最终进展',fmt(transfer_summary.get('progress_score'))+' / 100'],
        ['搬运成本效率均值',pct(transfer_summary.get('path_efficiency'))],
        ['动作总数',str(sum(r['steps'] for r in report['tasks']))],
        ['行走／reset 传送长度',fmt(summary['walking_length_m'])+' / '+fmt(summary['reset_displacement_m'])+' m'],
        ['碰撞次数／平移动作碰撞率',str(summary['collisions'])+' / '+pct(summary['collision_rate'])],
        ['已执行 reset 次数',str(summary['resets'])]]
    prose = (
        f"共 {summary['tasks_total']} 个任务：成功 {summary['successes']}，失败 {summary['failures']}，排除 {summary['tasks_excluded']}。"
        f"任务平均分 {fmt(summary['formal_rule_average'])}/100，分母 {summary['tasks_evaluated']}；"
        f"评分版本 {', '.join(summary.get('metric_versions', ['unavailable']))}。"
        f"原运行 official={report['original_run_official']}，完成状态={str((report.get('original_status') or {}).get('execution', '未提供'))}。")
    caveat = ('本报告是现有日志的离线复核，不将原运行重新认证为官方运行。原始 GT、请求和结果均保留；'
              '已排除任务为 N/A，问答／搬运不支持为 0。导航分仅适用于 visit／establish_place／revisit_place，'
              '带 navigation 能力标签的问答不套用导航到达评分。')
    if report.get('rescored'):
        caveat += ' 本报告已按 progress_path_v3 显式复算，原任务分另存 original_task_score；原运行的观测频率、任务文本及实际成功结果均保留。'
    if report.get('reference_reconstruction'):
        caveat += ' 路径参考地图由哈希验证的数据归档和每任务真实状态离线重建，基准网格逐格一致；它不是当时保存的实时地图快照。'
    methods = [
        '评分版本以每条记录为准：progress_path_v3 的导航按有序进度×路径效率计 0–100 分；搬运按剩余交付成本进展×成本效率计分。缺失评分证据保持 N/A。明确排除项为 N/A。',
        ('本次路径参考来自验证归档及真实状态的离线重建；' if report.get('reference_reconstruction') else '路径参考来自任务开始时保存的碰撞网格；')+'以 8 邻域带权最短路计算，不穿对角墙角，目标为运行时 GT 合格区域。它是网格近似，不是 Isaac navmesh。',
        '路径效率=D0/max(D0,已计费路径+剩余有序路线)，任务导航分=有序进度×路径效率。reset 按端点网格距离、实际位移和一格长度的最大值计费，不免费传送。',
        'progress_path_v3 搬运：P=clip(1-J剩余/J初始,0,1)，E=J初始/max(J初始,C已付+J剩余)，得分=100×P×E。移动距离按标准移动步长换算，每次抓取/放置尝试计 1，reset 按等效路径换算；转向和等待不额外计费。最终状态决定剩余工作，放错或掉落会重新计入恢复成本。',
        '搬运参考保存真实对象原点、可移除占据、房间标签、地面、抓取半径和放置偏移；放置方向使用 72 个等间隔朝向的网格近似，当前姿态额外检查真实朝向。缺少参考或证据时分数为 N/A，任务保留在分母。',
        '问答仅按答案正确性计 100／0，没有观察或动作过程分。离线复算保留日志中记录的完整成功标志。',
        '步骤与真实耗时单独显示。没有已验证的 GT 参考步数／耗时时，效率比值为 N/A；2 FPS 不等于每个动作固定 0.5 秒。',
        'd 为机器人 XY 点到运行时目标包围盒的距离；r=max(d−达标容差,0)。单目标进度=100×clip(1−r/r起点,0,1)。起点已在范围内时，在范围内为 100，否则为 0。',
        '多目标进度=100×(按序已完成数＋当前目标进度)/目标数。每段起点取前一个目标实际到达位置；同一到达样本不计下一个目标。最佳分是整条有序进度曲线的最大值。',
        '下表每个目标的起始／最终／全程最近距离是几何记录；只有激活后的到达才计入有序完成。已完成访问不会因为随后离开而撤销。',
        '行走长度是非 reset 动作前后 XY 位移之和。reset 传送距离单列；图中灰线是 reset，蓝线包含传送前后跳变，绿虚线是达标容差。',
        '最终／最佳进度包含 reset 后的实际位置，表示本次运行的几何进展；是否由传送帮助须结合 reset 明细判断，不归功于正常行走。',
        '碰撞次数指 collision=true 的动作条数，主碰撞率=碰撞平移动作/全部平移动作；JSON 和 CSV 同时保留全动作碰撞率。',
        '路径直达率只用于成功且无 reset 的任务：各段到达标范围的直线距离下界之和/行走长度，最大为 1。不是导航网格最短路或 SPL；失败、含 reset 或起始下界为零时为 N/A。',
        '恢复阶段到下一次 reset、当前目标按序达标或任务结束为止，比较同一个目标。仅传送带来的接近不算后续行走改善；几何进展不能证明识别或彻底解困。',
        f'近达标带：距达标范围 ≤ {report["near_margin_m"]:g} m，仅用于分类。始终未进入该带不等于模型未识别；反复碰撞／reset 表示至少 2 次。分类可重叠。',
        '下方原有距离曲线和最终／最佳几何进度仍使用包围盒欧氏距离，与新网格路径分分别标注。缺失或无法验证的参考不补造网格评分；缺失证据不缩小分母。',
        '历史记录耗时来自 task_start 到 task_complete 的单调时钟跨度，包含记录和录像开销；Agent 往返来自请求／响应时间戳，不等于纯模型推理耗时。它们不冒充新运行的精确动作计时。']
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>逐任务评分与导航诊断</title>',
        '<style>body{font:15px system-ui;margin:24px;color:#172c3b}p{line-height:1.65}table{border-collapse:collapse;width:100%}td,th{border:1px solid #d6e0e5;padding:8px;text-align:left;vertical-align:top}th{background:#eaf1f5;position:sticky;top:0}td.num{white-space:nowrap}.scroll{overflow:auto}details{margin:12px 0}section{border-top:2px solid #abc;padding:16px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:600px;overflow:auto}a{color:#086591}input,select{padding:8px;margin:6px}svg{max-width:800px;width:100%;background:#f6f9fb}svg text{font-size:11px}li{margin:8px 0}.note{background:#fff2da;padding:12px}</style>',
        '<h1>逐任务评分、距离、路径与恢复诊断</h1><p>'+esc(prose)+'</p><p class="note">'+esc(caveat)+'</p>',
        table(['指标','结果'],[[esc(k),esc(v)] for k,v in metric_table]),
        table(['任务类型','计分／排除','成功率','新版任务分（满分100）'],[
            [esc(k),str(v['tasks_evaluated'])+' / '+str(v['tasks_excluded']),pct(v['success_rate']),fmt(v['score_100'])]
            for k,v in summary.get('by_task_kind',{}).items()]),
        detail('逐 Session 分数、网格指标及时间测量', {k:summary.get(k) for k in ('by_session','navigation_metric','recorded_timing')}),
        '<p><a href="tasks.csv">任务 CSV</a> · <a href="sessions.csv">Session CSV</a> · <a href="navigation_goals.csv">目标距离 CSV</a> · <a href="transfer_goals.csv">搬运成本 CSV</a> · <a href="resets.csv">逐次 reset CSV</a> · <a href="scores.json">完整 JSON 与来源哈希</a></p>',
        '<details><summary>计算口径</summary><ul>'+''.join('<li>'+esc(m)+'</li>' for m in methods)+'</ul></details>',
        detail('汇总、分类与恢复统计', summary), detail('原运行完成状态', report.get('original_status')),
        detail('证据问题（空列表表示本次复核无差异）', report['issues']),
        '<h2>问题归类（标签可重叠）</h2>',
        table(['类别', '全部导航任务', '失败导航任务'], [[esc(k), str(v), str(summary['failed_navigation_categories'].get(k, 0))] for k, v in summary['categories'].items()]),
        '<details><summary>reset 恢复情况</summary>'+table(['几何进展判断', '次数'], [[esc(k), str(v)] for k, v in summary['reset_categories'].items()])+f'<p>其中 {summary["resets_with_new_best_from_walking"]} 次后续行走刷新了该目标复位前的最近距离；该数字不是单凭距离断言解困成功。</p></details>',
        '<h2>按序任务对照表</h2><p>距离单位 m；多目标逐项列出。点击任务查看 GT、成功条件、距离曲线和每次 reset。</p>',
        '<label>检索 <input id="query" placeholder="任务、文本、原因"></label><label>结果 <select id="result"><option value="">全部</option><option value="success">成功</option><option value="failure">失败</option><option value="excluded">排除</option><option value="navigation">导航</option></select></label>']
    headers = ['序号／任务', '结果／原因', '任务分（满分100）', '最终几何进度（满分100）', '最佳几何进度（满分100）',
               '网格进度（满分100）', '网格路径效率', '动作数', '记录耗时 s', '起始／最终／最近距离', '行走／传送', '碰撞／平移次数／率', 'reset', '路径直达率', '问题分类']
    parts.append('<div class="scroll"><table id="tasks"><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>')
    for row in report['tasks']:
        nav = row['navigation'] or {}; motion = nav.get('motion', {})
        evaluation=row.get('evaluation') or {}; path_metric=evaluation.get('navigation') or {}
        status = 'excluded' if row['excluded'] else 'success' if row['success'] else 'failure'
        cells = [f'{row["order"]} · <a href="#{esc(ref(row))}">{esc(row["session_id"]+"/"+row["task_id"])}</a><br>'+esc(' → '.join(row['target_names']))+'<br>'+esc(' / '.join(row['goal_kinds'])),
            esc(('排除' if row['excluded'] else '成功' if row['success'] else '失败')+' / '+row['reason_code']),
            '<b>'+fmt(row['formal_task_score'])+'</b><br>'+esc(row.get('metric_version','unavailable')),
            fmt(nav.get('final_progress_score')), fmt(nav.get('best_progress_score')),
            fmt(path_metric.get('progress_score')),pct(path_metric.get('path_efficiency')),
            str(row['steps']),fmt(evaluation.get('measurements',{}).get('elapsed_seconds',evaluation.get('measurements',{}).get('recorded_task_span_seconds'))),
            '<br>'.join(esc(g['goal_id'])+': '+' / '.join(fmt(g[k]) for k in ('start_distance_m', 'final_distance_m', 'closest_distance_m')) for g in nav.get('goals', [])) or 'N/A',
            fmt(motion.get('walking_length_m'))+' / '+fmt(motion.get('reset_displacement_m')),
            fmt(motion.get('collision_count'), 0)+' / '+fmt(motion.get('move_attempts'), 0)+' / '+pct(motion.get('collision_rate')),
            fmt(motion.get('reset_count'), 0), pct(nav.get('walking_efficiency')), esc('；'.join(nav.get('categories', []))) or '—']
        search = ref(row)+' '+str(row['instruction'])+' '+str(row['reason_code'])+' '+' '.join(nav.get('categories', []))
        parts.append('<tr data-result="'+status+'" data-nav="'+str(bool(nav)).lower()+'" data-search="'+esc(search.lower())+'">'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
    parts.append('</tbody></table></div>')
    for row in report['tasks']:
        nav = row['navigation']
        parts.append('<section id="'+esc(ref(row))+'"><h2>'+str(row['order'])+' · '+esc(ref(row))+'</h2><p>本次实际下发文本（保留原日志）：'+esc(row['instruction'] or '未下发')+'</p><p>任务分：'+fmt(row['formal_task_score'])+' / 100；结果：'+esc(row['reason_code'])+'</p><p>'+esc(row['result_explanation'])+'</p><ol>'+''.join('<li>'+esc(c)+'</li>' for c in row['success_conditions'])+'</ol>')
        if row.get('evaluation'):
            parts.append(detail('评分版本、目标部分分、路径计费及时间测量',row['evaluation']))
            transfers = row['evaluation'].get('transfers', [])
            if transfers:
                parts.append(table(['搬运目标', '初始成本 J0', '剩余成本 J', '已付成本 C', '最终进展', '成本效率', '得分'],
                    [[esc(g['goal_id']), fmt(g['reference_cost']), fmt(g['remaining_cost']), fmt(g['charged_cost']),
                      fmt(g['progress_score']), pct(g['path_efficiency']), fmt(g['score'])] for g in transfers]))
        parts.append('<p>'+video_link(row['video'], output)+' · <a href="#tasks">返回对照表</a></p>')
        parts.append(detail('执行结果与失败证据', {'prediction': row['prediction'], 'goal_results': row['goal_results'], 'failure_causes': row['failure_causes']}))
        parts.append(detail('实际运行时 GT（评分字段）', row['runtime_gt']))
        parts.append(detail('离线 GT（仅对照，不用于评分）', row['original_gt']))
        parts.append(detail('修补依据及日志位置', {k: row[k] for k in ('repair_record', 'runtime_gt_source', 'outcome_source')}))
        if row.get('navigation_error'):
            parts.append('<p class="note">导航指标不可用：'+esc(row['navigation_error'])+'</p>')
        if nav:
            parts.append('<p>按序完成 '+str(nav['goals_completed_in_order'])+'/'+str(nav['goals_total'])+'；最终进度 '+fmt(nav['final_progress_score'])+'；最佳进度 '+fmt(nav['best_progress_score'])+'。</p>')
            for g in nav['goals']:
                parts.append('<details><summary>'+esc(g['goal_id']+' · '+g['name'])+'；起始／最终／最近 '+ ' / '.join(fmt(g[k]) for k in ('start_distance_m', 'final_distance_m', 'closest_distance_m'))+' m；按序完成 '+str(g['ordered_success'])+'</summary>'+distance_chart(g, nav['reset_recoveries'])+detail('目标明细（激活前路过不计完成）', {k: v for k, v in g.items() if k != 'distance_series_m'})+'</details>')
            if nav['reset_recoveries']:
                reset_table = []
                for r in nav['reset_recoveries']:
                    reset_table.append([str(r['reset_number'])+' / 动作 '+str(r['task_step']+1), esc(r['goal_id']),
                        ' / '.join(fmt(r[k]) for k in ('before_distance_m', 'after_distance_m', 'recovery_best_distance_m', 'recovery_end_distance_m')),
                        fmt(r['reset_displacement_m'])+' / '+fmt(r['recovery_window']['walking_length_m']),
                        fmt(r['teleport_progress_m'])+' / '+fmt(r['walking_progress_m'])+' / '+fmt(r['net_progress_m']),
                        pct(r['before_window']['collision_rate'])+' / '+pct(r['recovery_window']['collision_rate']),
                        esc(r['category'])+('；行走刷新此前最近距离' if r['new_best_from_walking'] else ''),
                        esc(r['recovery_end_reason'])+' / 动作 '+str(r['recovery_end_task_step']+1), video_link(row['video'], output, r['video_seconds']) if r['video_seconds'] is not None else 'N/A'])
                parts.append('<details><summary>'+str(len(reset_table))+' 次 reset 的恢复对照</summary>'+table(['次序／动作', '比较目标', '复位前／刚复位／后续最佳／阶段末距离', '传送／后续行走 m', '传送接近／行走接近／净接近 m', '前阶段／后阶段碰撞率', '几何进展判断', '恢复窗口终点', '视频'], reset_table)+'</details>')
        parts.append('</section>')
    parts.append('<script>const q=document.querySelector("#query"),s=document.querySelector("#result");function filter(){for(const r of document.querySelectorAll("#tasks tbody tr")){r.hidden=!(r.dataset.search.includes(q.value.toLowerCase())&&(!s.value||(s.value==="navigation"?r.dataset.nav==="true":r.dataset.result===s.value)));}}q.addEventListener("input",filter);s.addEventListener("change",filter);</script></html>')
    (output/'report.html').write_text(''.join(parts), encoding='utf-8')
    md = ['# 逐任务评分与导航诊断', '', prose, '', caveat, '',
          '[交互报告](report.html) · [任务 CSV](tasks.csv) · [目标距离 CSV](navigation_goals.csv) · [搬运成本 CSV](transfer_goals.csv) · [reset CSV](resets.csv)', '',
          '| 指标 | 结果 |','|---|---:|',*['| '+k+' | '+v+' |' for k,v in metric_table], '',
          f"导航任务 {summary['navigation_tasks']} 个；最终／最佳进度均值 {fmt(summary['navigation_final_progress_mean'])} / {fmt(summary['navigation_best_progress_mean'])}。", '',
          f"行走 {fmt(summary['walking_length_m'])} m；reset 传送 {fmt(summary['reset_displacement_m'])} m；碰撞 {summary['collisions']} 次，平移动作碰撞率 {pct(summary['collision_rate'])}；reset {summary['resets']} 次。", '',
          '失败导航任务分类（可重叠）：'+ '；'.join(f'{k} {v} 个' for k,v in summary['failed_navigation_categories'].items())+'。', '',
          'reset 分类：'+ '；'.join(f'{k} {v} 次' for k,v in summary['reset_categories'].items())+'。', '',
          '| 序号 | 任务／目标 | 任务分（满分100） | 原因 | 最终几何进度（满分100） | 最佳几何进度（满分100） |', '|---|---|---:|---|---:|---:|']
    for row in report['tasks']:
        nav = row['navigation'] or {}
        md.append(f"| {row['order']} | {row['session_id']}/{row['task_id']} · {' → '.join(row['target_names'])} | {fmt(row['formal_task_score'])} | {row['reason_code']} | {fmt(nav.get('final_progress_score'))} | {fmt(nav.get('best_progress_score'))} |")
    md.extend(['', '## 口径', '']+['- '+m for m in methods])
    (output/'summary.md').write_text('\n'.join(md)+'\n', encoding='utf-8')


def write_task_scores(root, output, *, near_margin_m=.5, rescore=False, reference_dataset=None):
    """Compute and render task scores and evidence diagnostics."""
    from ..evaluation.task_diagnostics import analyze_run
    output = Path(output)
    if output.exists():
        raise FileExistsError('analysis output must be a new directory: ' + str(output))
    report = analyze_run(root, near_margin_m=near_margin_m, rescore=rescore,
        reference_dataset=reference_dataset, reference_output=output)
    write_diagnostics(report, output)
    return report
