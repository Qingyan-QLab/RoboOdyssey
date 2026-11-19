"""Legacy task reports and runtime repair review from durable logs."""

from __future__ import annotations

from collections import Counter, defaultdict
import html
import json
from pathlib import Path
from statistics import mean


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sample(points: list[list[float]], limit: int = 260) -> list[list[float]]:
    if len(points) <= limit:
        return points
    stride = max(1, (len(points) - 1) // (limit - 1))
    result = points[::stride]
    if result[-1] != points[-1]:
        result.append(points[-1])
    return result[:limit]


def collect(root: Path) -> dict:
    summary = _read(root / "summary.json")
    sessions = []
    family = defaultdict(
        lambda: {"tasks": 0, "success": 0, "steps": 0, "collisions": 0, "distance": [], "spl": []}
    )
    action_totals = Counter()
    trajectory_by_session = {}
    for session_metrics in summary.get("session_metrics", []):
        session_id = str(session_metrics["session_id"])
        session_dir = root / "scenes" / str(session_metrics["scene_id"]) / "sessions" / session_id
        actions = Counter()
        trajectory = []
        if (session_dir / "action_changelog.jsonl").is_file():
            with (session_dir / "action_changelog.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    event = json.loads(line)
                    action = event.get("action")
                    name = action.get("action") if isinstance(action, dict) else action
                    actions[str(name)] += 1
                    action_totals[str(name)] += 1
                    position = (event.get("after") or {}).get("position")
                    if isinstance(position, list) and len(position) >= 2:
                        trajectory.append([float(position[0]), float(position[1])])
        trajectory_by_session[session_id] = _sample(trajectory)
        nav = [
            item
            for item in session_metrics.get("task_results", [])
            if item.get("task_type") == "navigation" and item.get("scored", True)
        ]
        sessions.append(
            {
                "id": session_id,
                "success": session_metrics.get("success"),
                "score": session_metrics.get("task_score"),
                "reset_count": int(session_metrics.get("reset_count", 0)),
                "tasks": session_metrics.get(
                    "tasks_evaluated", len(session_metrics.get("task_results", []))
                ),
                "excluded_tasks": session_metrics.get("tasks_excluded", 0),
                "nav_success": sum(bool(item.get("success")) for item in nav),
                "nav_tasks": len(nav),
                "steps": sum(int(item.get("steps", 0)) for item in nav),
                "collisions": sum(int(item.get("collisions", 0)) for item in nav),
                "actions": dict(actions),
                "nav": [
                    {
                        "id": item.get("task_id"),
                        "family": item.get("task_family"),
                        "success": bool(item.get("success")),
                        "distance": item.get("final_distance_m"),
                        "steps": item.get("steps", 0),
                        "collisions": item.get("collisions", 0),
                        "spl": item.get("spl", 0.0),
                        "reason": item.get("completion_reason"),
                        "required_order": item.get("required_site_order", []),
                        "visited": item.get("visited_sites", []),
                    }
                    for item in nav
                ],
            }
        )
        for item in session_metrics.get("task_results", []):
            if item.get("scored", True) is False:
                continue
            key = str(item.get("task_family", item.get("task_type", "unknown")))
            value = family[key]
            value["tasks"] += 1
            value["success"] += int(bool(item.get("success")))
            value["steps"] += int(item.get("steps", 0))
            value["collisions"] += int(item.get("collisions", 0))
            if item.get("final_distance_m") is not None:
                value["distance"].append(float(item["final_distance_m"]))
            if item.get("spl") is not None:
                value["spl"].append(float(item["spl"]))
    family_out = {}
    for key, value in sorted(family.items()):
        family_out[key] = {
            "tasks": value["tasks"],
            "success": value["success"],
            "rate": value["success"] / value["tasks"] if value["tasks"] else 0.0,
            "steps": value["steps"],
            "collisions": value["collisions"],
            "avg_distance": mean(value["distance"]) if value["distance"] else None,
            "avg_spl": mean(value["spl"]) if value["spl"] else None,
        }
    return {
        "summary": summary,
        "sessions": sessions,
        "families": family_out,
        "actions": dict(action_totals),
        "trajectories": trajectory_by_session,
        "run_config": _read(root / "run_config.json")
        if (root / "run_config.json").exists()
        else {},
    }


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Benchmark result report</title>
<style>
body{margin:0;background:#f5f7fb;color:#1f2937;font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1480px;margin:auto;padding:28px}h1{margin:0 0 6px;font-size:27px}h2{margin:26px 0 12px;font-size:19px}.muted{color:#6b7280}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px}.card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:15px;box-shadow:0 1px 2px #0000000b}.value{font-size:25px;font-weight:700}.label{color:#6b7280;font-size:12px}.good{color:#087f5b}.bad{color:#c92a2a}.warn{color:#b26a00}.grid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:16px}.panel{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px;overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px 9px;border-bottom:1px solid #edf0f4;text-align:left;white-space:nowrap}th{background:#f8fafc;position:sticky;top:0}.bar{height:10px;border-radius:5px;background:#e5e7eb;overflow:hidden;min-width:90px}.bar i{display:block;height:100%;background:#2f80ed}.pill{display:inline-block;padding:2px 7px;border-radius:99px;font-size:12px;background:#fee2e2;color:#991b1b}.pill.ok{background:#dcfce7;color:#166534}svg{width:100%;height:360px;background:#fbfcfe;border:1px solid #edf0f4;border-radius:8px}.legend{display:flex;gap:14px;flex-wrap:wrap;color:#64748b;font-size:12px;margin:7px 0}.legend span:before{content:"";display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--c);margin-right:5px}.note{border-left:4px solid #f59e0b;background:#fffbeb;padding:10px 12px;margin:12px 0}.small{font-size:12px}
@media(max-width:900px){.grid{grid-template-columns:1fr}main{padding:16px}}
</style></head><body><main>
<h1>Benchmark 评测结果诊断</h1><div class="muted" id="subtitle"></div><div id="diagnosis"></div><div class="cards" id="cards"></div>
<h2>Session 概览</h2><div class="panel"><table id="sessions"><thead><tr><th>Session</th><th>任务成功率</th><th>导航</th><th>导航步数</th><th>碰撞</th><th>reset</th><th>状态</th></tr></thead><tbody></tbody></table></div>
<div class="grid"><section><h2>任务族表现</h2><div class="panel"><table id="families"><thead><tr><th>Task family</th><th>成功率</th><th>样本</th><th>平均距离</th><th>平均 SPL</th><th>碰撞</th></tr></thead><tbody></tbody></table></div></section><section><h2>动作分布</h2><div class="panel"><table id="actions"><thead><tr><th>Action</th><th>次数</th><th>占比</th></tr></thead><tbody></tbody></table></div></section></div>
<h2>轨迹与任务细节</h2><div class="panel"><label>选择 session： <select id="selector"></select></label><div class="legend"><span style="--c:#2f80ed">agent path</span><span style="--c:#2ca25f">task success</span><span style="--c:#de2d26">task failure</span></div><svg id="plot" viewBox="0 0 900 360" preserveAspectRatio="none"></svg><div id="taskdetail"></div></div>
<h2>判读</h2><div class="panel small"><ul><li>“completed”只表示 benchmark 正常跑完，不表示任务成功。</li><li>SimpleNav 适配器对 QA 和 manipulation 主动返回失败，因此总体分数不应解读为完整 benchmark 能力。</li><li>导航任务存在大量碰撞、reset 和达到 2000 步上限的情况；这些是当前导航闭环的主要瓶颈。</li></ul></div>
</main><script>const DATA=__DATA__;
const pct=x=>x==null?'—':(100*x).toFixed(1)+'%'; const esc=x=>String(x??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
const s=DATA.summary; document.getElementById('subtitle').textContent=`${s.dataset_id||''} · ${s.sessions_evaluated} sessions · ${s.tasks_evaluated} scored tasks · ${s.tasks_excluded||0} excluded`;
const diagnosis=document.getElementById('diagnosis'); diagnosis.innerHTML=`<div class="note">计分任务 <b>${s.tasks_evaluated}</b>，排除 <b>${s.tasks_excluded||0}</b>；总体成功率 <b>${pct(s.task_success_rate)}</b>，中断 Session <b>${s.sessions_interrupted||0}</b>。排除原因：${esc(JSON.stringify(s.exclusions_by_reason||{}))}。无计分任务时显示“—”，比较成功率时应同时查看排除数量。</div>`;
const cards=[['Task success',pct(s.task_success_rate),s.task_success_rate>.5?'good':'bad'],['Navigation success',pct(s.task_type_success_rate.navigation??null),(s.task_type_success_rate.navigation??null)>.5?'good':'warn'],['Session success',pct(s.session_success_rate),s.session_success_rate?'good':'bad'],['Resets',s.reset_count,'warn'],['QA success',pct(s.task_type_success_rate.qa??null),'bad'],['Manipulation success',pct(s.task_type_success_rate.manipulation??null),'bad']]; document.getElementById('cards').innerHTML=cards.map(x=>`<div class="card"><div class="label">${x[0]}</div><div class="value ${x[2]}">${x[1]}</div></div>`).join('');
document.querySelector('#sessions tbody').innerHTML=DATA.sessions.map(x=>`<tr><td>${x.id}</td><td>${pct(x.score)}</td><td>${x.nav_success}/${x.nav_tasks}</td><td>${x.steps}</td><td>${x.collisions}</td><td>${x.reset_count}</td><td><span class="pill ${x.success?'ok':''}">${x.success==null?'not scored':x.success?'success':'failed'}</span></td></tr>`).join('');
document.querySelector('#families tbody').innerHTML=Object.entries(DATA.families).map(([k,x])=>`<tr><td>${esc(k)}</td><td>${pct(x.rate)}</td><td>${x.success}/${x.tasks}</td><td>${x.avg_distance==null?'—':x.avg_distance.toFixed(2)+' m'}</td><td>${x.avg_spl==null?'—':x.avg_spl.toFixed(3)}</td><td>${x.collisions}</td></tr>`).join('');
const totalActions=Object.values(DATA.actions).reduce((a,b)=>a+b,0); document.querySelector('#actions tbody').innerHTML=Object.entries(DATA.actions).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${v}</td><td>${pct(v/totalActions)}</td></tr>`).join('');
const selector=document.getElementById('selector'); selector.innerHTML=DATA.sessions.map(x=>`<option>${x.id}</option>`).join(''); function draw(){const id=selector.value, item=DATA.sessions.find(x=>x.id===id), pts=DATA.trajectories[id]||[], svg=document.getElementById('plot'); svg.innerHTML=''; if(!pts.length){svg.innerHTML='<text x="30" y="50">无轨迹数据</text>';return} const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]), minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys), sx=820/Math.max(1e-6,maxx-minx),sy=300/Math.max(1e-6,maxy-miny), f=p=>[40+(p[0]-minx)*sx,330-(p[1]-miny)*sy]; const d=pts.map((p,i)=>(i?'L':'M')+f(p).map(v=>v.toFixed(1)).join(' ')).join(' '); svg.innerHTML=`<path d="${d}" fill="none" stroke="#2f80ed" stroke-width="2"/><circle cx="${f(pts[0])[0]}" cy="${f(pts[0])[1]}" r="4" fill="#2ca25f"/><circle cx="${f(pts.at(-1))[0]}" cy="${f(pts.at(-1))[1]}" r="4" fill="#de2d26"/><text x="42" y="20" fill="#64748b" font-size="12">x ${minx.toFixed(1)}…${maxx.toFixed(1)} · y ${miny.toFixed(1)}…${maxy.toFixed(1)}</text>`; document.getElementById('taskdetail').innerHTML=`<table><thead><tr><th>Task</th><th>Family</th><th>结果</th><th>距离</th><th>步数</th><th>碰撞</th><th>SPL</th></tr></thead><tbody>${item.nav.map(t=>`<tr><td>${esc(t.id)}</td><td>${esc(t.family)}</td><td><span class="pill ${t.success?'ok':''}">${t.success?'success':'failed'}</span></td><td>${t.distance==null?'—':Number(t.distance).toFixed(2)+' m'}</td><td>${t.steps}</td><td>${t.collisions}</td><td>${Number(t.spl).toFixed(3)}</td></tr>`).join('')}</tbody></table>`} selector.onchange=draw; draw();</script></body></html>"""


def write_report(root, output):
    """Render the legacy HTML view for either command entry."""
    data = json.dumps(collect(Path(root)), ensure_ascii=False).replace("<", r"\u003c")
    Path(output).write_text(HTML.replace("__DATA__", data), encoding="utf-8")


def write_review(output):
    path = output / "runtime_tasks.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = []
    for record in records:
        cells = [
            record["task_id"],
            record["original_text"],
            record.get("rendered_text") or "未下发",
            record["status"],
            record.get("reason_code", ""),
            ", ".join(record["reasons"]),
            json.dumps(record.get("dependency") or {}, ensure_ascii=False),
            "不计分" if record.get("scored") is False else "计分",
        ]
        rows.append(
            "<tr>" + "".join("<td>" + html.escape(str(c)) + "</td>" for c in cells) + "</tr>"
        )
    (output / "runtime_review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>运行时任务修补</title>'
        "<style>body{font:15px sans-serif;margin:24px}table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ccc;padding:10px;text-align:left;vertical-align:top}</style>"
        "<h1>运行时任务修补</h1><p>私有审阅；下发前已满足目标及无法下发的任务分别记录并排除计分。缺失评测结果仍按失败计。</p>"
        "<table><tr><th>任务</th><th>原始文本</th><th>实际下发</th><th>状态</th><th>原因代码</th><th>说明</th><th>来源任务</th><th>计分</th></tr>"
        + "".join(rows)
        + "</table>",
        encoding="utf-8",
    )
