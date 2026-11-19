"""Offline task scores and navigation diagnostics from immutable v3 run logs.

No simulator, model, offline reference path, or regenerated runtime request is
used here. Distances describe geometry; they do not establish recognition.
"""
from ..agents.protocol import required_action_name as action_name
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
from ..artifacts.layout import iter_sessions, session_file, sha256_file, read_json_lines as read_lines
from .navigation import box_distance
from .task_metrics import METRIC_VERSION, SUPPORTED_METRIC_VERSIONS, evaluate_metrics, load_reference
from .rescore import final_state, historical_measurements, metric_breakdown, ReconstructedReference


NAVIGATION = {'visit', 'establish_place', 'revisit_place'}
MOVES = {'move_forward', 'move_backward', 'move_left', 'move_right'}
EPS = 1e-6


def xy(pose):
    value = pose['position'][:2]
    if len(value) != 2 or any(not math.isfinite(v) for v in value):
        raise ValueError('missing or nonfinite XY pose')
    return value


def target_box(goal):
    if goal['goal_kind'] == 'revisit_place':
        box = goal['fixed_target_geometry']['world_aabb']
    else:
        box = goal['expected_outcome']['target_geometry']['world_aabb_xy']
    if any(len(box[k]) < 2 or any(not math.isfinite(v) for v in box[k][:2]) for k in ('min', 'max')):
        raise ValueError('invalid runtime target box')
    if any(box['min'][i] > box['max'][i] for i in (0, 1)):
        raise ValueError('inverted runtime target box')
    return {k: box[k][:2] for k in ('min', 'max')}


def is_applied_reset(action):
    return action_name(action) == 'reset' and action['result'].get('reset_applied') is True


def motion_metrics(actions):
    """Action endpoint displacements, with every reset excluded from walking."""
    walk = teleport = 0.0
    moves = collisions = move_collisions = collision_missing = move_flags_missing = 0
    resets = reset_attempts = 0
    for event in actions:
        name = action_name(event)
        displacement = math.dist(xy(event['before']), xy(event['after']))
        if name == 'reset':
            reset_attempts += 1
            if 'reset_applied' not in event['result']:
                raise ValueError('reset result does not prove whether teleport was applied')
            resets += is_applied_reset(event)
            if not is_applied_reset(event) and displacement > EPS:
                raise ValueError('reset moved the robot without reset_applied evidence')
            teleport += displacement
        else:
            walk += displacement
        flag = event['result'].get('collision')
        collision_missing += type(flag) is not bool
        collisions += flag is True
        if name in MOVES:
            moves += 1
            move_collisions += flag is True
            move_flags_missing += type(flag) is not bool
    return {'action_count': len(actions), 'walking_length_m': walk, 'reset_displacement_m': teleport,
            'reset_count': resets, 'reset_attempts': reset_attempts, 'move_attempts': moves,
            'collision_count': collisions if not collision_missing else None,
            'move_collision_count': move_collisions if not move_flags_missing else None,
            'collision_rate': move_collisions / moves if moves and not move_flags_missing else None,
            'collision_rate_all_actions': collisions / len(actions) if actions and not collision_missing else None,
            'missing_collision_flags': collision_missing}


def progress_fraction(distance, tolerance, baseline):
    remaining = max(0.0, distance - tolerance)
    if baseline == 0:
        return 1.0 if remaining == 0 else 0.0
    return min(1.0, max(0.0, 1.0 - remaining / baseline))


def navigation_diagnostics(truth, state, actions, outcome, *, near_margin_m=.5):
    goals = truth['goals']
    if not goals or any(g['goal_kind'] not in NAVIGATION for g in goals):
        return None
    if truth.get('state_assumption') != 'actual_predispatch_state':
        raise ValueError('navigation requires the recorded runtime GT, not planned-success GT')
    if truth['success_criteria']['goal_order'] != [g['goal_id'] for g in goals]:
        raise ValueError('goal order differs from runtime GT goal list')
    target_ids = {g['target']['object_ids'][0] for g in goals if g['goal_kind'] != 'revisit_place'}
    if any(change['object_id'] in target_ids for a in actions
           for change in a.get('state_changes', {}).get('objects', [])):
        raise ValueError('target moved during task; static-target distance series is not applicable')
    start = xy(state.get('robot_pose') or state['agent_pose'])
    # Match the scorer's before/after sample indices, including repeated poses
    # between actions. Never interpolate through a teleport or credit a later
    # target before its predecessor has a strictly earlier observation index.
    poses = [xy(a[phase]) for a in actions for phase in ('before', 'after')] or [start]
    if math.dist(start, poses[0]) > 1e-4:
        raise ValueError('trajectory start differs from predispatch state')
    if any(math.dist(xy(a['after']), xy(b['before'])) > 1e-4 for a, b in zip(actions, actions[1:])):
        raise ValueError('unlogged discontinuity between actions')
    motion = motion_metrics(actions)
    boxes = [target_box(g) for g in goals]
    tolerances = [g['success_criteria'].get('position_tolerance_m', .8) for g in goals]
    distances = [[box_distance(p, b) for p in poses] for b in boxes]
    activations, reached = [0], []
    stage = 0
    timeline, completed = [], []
    for sample in range(len(poses)):
        was_reached = False
        if stage < len(goals) and distances[stage][sample] <= tolerances[stage]:
            reached.append(sample)
            stage += 1
            was_reached = True
            if stage < len(goals):
                activations.append(sample)
        if stage == len(goals):
            score = 100.0
        elif was_reached:
            # The newly activated goal cannot receive credit at the same sample.
            score = 100.0 * stage / len(goals)
        else:
            baseline = max(0.0, distances[stage][activations[stage]] - tolerances[stage])
            score = 100.0 * (stage + progress_fraction(distances[stage][sample], tolerances[stage], baseline)) / len(goals)
        timeline.append(score)
        completed.append(stage)
    goal_rows = []
    for index, goal in enumerate(goals):
        active = activations[index] if index < len(activations) else None
        hit = reached[index] if index < len(reached) else None
        # Start at the first strictly later sample for later goals. The entry
        # point supplies the denominator but alone cannot prove the next visit.
        eligible = (active + (index > 0)) if active is not None else None
        segment = distances[index][eligible:(hit + 1 if hit is not None else len(poses))] if eligible is not None else []
        goal_rows.append({'goal_id': goal['goal_id'], 'goal_kind': goal['goal_kind'],
            'object_id': goal['target']['object_ids'][0], 'target_box_xy': boxes[index],
            'tolerance_m': tolerances[index], 'start_distance_m': distances[index][0],
            'final_distance_m': distances[index][-1], 'closest_distance_m': min(distances[index]),
            'active_start_distance_m': distances[index][active] if active is not None else None,
            'active_closest_distance_m': min(segment) if segment else None,
            'activation_pose_index': active, 'reached_pose_index': hit,
            'ordered_success': hit is not None,
            'distance_series_m': distances[index]})
    logged = outcome.get('goal_results', [])
    if len(logged) != len(goals) or any(
            row['goal_id'] != evidence['goal_id'] or row['ordered_success'] != evidence['success']
            or row['reached_pose_index'] != evidence.get('pose_index')
            for row, evidence in zip(goal_rows, logged)):
        raise ValueError('reconstructed navigation visits disagree with recorded goal evidence')
    final_goal = min(stage, len(goals) - 1)
    last = goal_rows[final_goal]
    tags = []
    if outcome.get('success') is not True and stage < len(goals):
        if len(goals) > 1 and stage:
            tags.append('多目标只完成部分')
        closest = last['active_closest_distance_m']
        if closest is not None:
            if closest - last['tolerance_m'] > near_margin_m:
                tags.append('始终未进入近达标带')
            else:
                tags.append('接近但未达标')
                if last['final_distance_m'] - closest > near_margin_m:
                    tags.append('接近后离开')
    if (motion['collision_count'] or 0) >= 2:
        tags.append('反复碰撞')
    if motion['reset_count'] >= 2:
        tags.append('反复 reset')
    # A leg's straight-line distance to its allowed region is a lower bound,
    # not a navigable shortest path. Teleport-assisted or incomplete routes do
    # not have a comparable walking efficiency and are deliberately N/A.
    lower_bound = sum(max(0.0, r['active_start_distance_m'] - r['tolerance_m']) for r in goal_rows) if stage == len(goals) else None
    efficiency = (lower_bound / max(lower_bound, motion['walking_length_m'], EPS)
                  if outcome.get('success') is True and lower_bound is not None and lower_bound > EPS and not motion['reset_count'] else None)
    resets = reset_diagnostics(actions, goal_rows, completed)
    return {'distance_type': 'aabb_2d', 'goals_completed_in_order': stage, 'goals_total': len(goals),
            'final_progress_score': timeline[-1], 'best_progress_score': max(timeline),
            'active_goal_at_end': last['goal_id'], 'goals': goal_rows, 'motion': motion,
            'straight_line_leg_lower_bound_m': lower_bound, 'walking_efficiency': efficiency,
            'walking_efficiency_basis': 'completed ordered legs; straight-line lower bound; no resets; not SPL',
            'reset_recoveries': resets, 'categories': tags}


def reset_diagnostics(actions, goals, completed):
    reset_indices = [i for i, a in enumerate(actions) if is_applied_reset(a)]
    rows = []
    for ordinal, index in enumerate(reset_indices):
        event = actions[index]
        before, after = 2 * index, 2 * index + 1
        target = min(completed[before], len(goals) - 1)
        goal = goals[target]
        series = goal['distance_series_m']
        following = reset_indices[ordinal + 1] if ordinal + 1 < len(reset_indices) else len(actions)
        end = 2 * following - 1
        hit = goal['reached_pose_index']
        if hit is not None and after <= hit <= end:
            end = hit
        earlier = reset_indices[ordinal - 1] + 1 if ordinal else 0
        recovery_actions = [a for j, a in enumerate(actions[index + 1:following], index + 1) if 2 * j + 1 <= end]
        pre_motion = motion_metrics(actions[earlier:index])
        recovery_motion = motion_metrics(recovery_actions)
        best = min(series[after:end + 1])
        prior_best = min(series[goal['activation_pose_index']:before + 1])
        walking_gain = series[after] - best
        net_gain = series[before] - best
        if hit == after:
            category = '复位直接进入达标范围'
        elif hit is not None and after < hit <= end:
            category = '复位后行走达标'
        elif walking_gain > EPS and net_gain > EPS:
            category = '后续行走优于复位前位置'
        elif walking_gain > EPS:
            category = '复位后有接近但未优于复位前'
        elif net_gain > EPS:
            category = '仅传送带来接近'
        else:
            category = '未获得更近位置'
        rows.append({'reset_number': ordinal + 1, 'task_step': event['task_step'],
            'event_id': event.get('event_id'), 'action_source': event.get('action_source'),
            'goal_id': goal['goal_id'], 'object_id': goal['object_id'],
            'before_distance_m': series[before], 'after_distance_m': series[after],
            'prior_best_distance_m': prior_best, 'recovery_best_distance_m': best,
            'recovery_end_distance_m': series[end], 'teleport_progress_m': series[before] - series[after],
            'walking_progress_m': walking_gain, 'net_progress_m': net_gain,
            'new_best_from_walking': best < min(prior_best, series[after]) - EPS,
            'reset_displacement_m': math.dist(xy(event['before']), xy(event['after'])),
            'recovery_end_pose_index': end, 'recovery_end_task_step': actions[end // 2]['task_step'],
            'recovery_end_reason': 'goal_reached' if hit == end else 'next_reset' if following < len(actions) else 'task_end',
            'before_window': pre_motion, 'recovery_window': recovery_motion, 'category': category})
    return rows


def compact_gt(truth):
    if truth is None:
        return None
    keys = ('goal_id', 'goal_kind', 'target', 'query', 'preconditions', 'expected_outcome',
            'success_criteria', 'inspection_contract', 'fixed_target_geometry', 'named_place', 'state_assumption')
    value = {k: deepcopy(truth[k]) for k in ('task_id', 'state_assumption', 'success_criteria', 'scoring_contract') if k in truth}
    value['goals'] = [{k: deepcopy(g[k]) for k in keys if k in g} for g in truth['goals']]
    for original, goal in zip(truth['goals'], value['goals']):
        if original.get('memory_evidence'):
            goal['history_binding'] = deepcopy(original['memory_evidence'].get('binding'))
    return value


def task_score(outcome):
    if outcome.get('scored') is False:
        reason = outcome.get('reason_code')
        if reason not in {'already_satisfied', 'unavailable'} or outcome.get('dispatched') is not False:
            raise ValueError('unproved exclusion')
        if reason == 'unavailable' and not outcome.get('failure_causes'):
            raise ValueError('exclusion has no failure causality')
        return None
    if outcome.get('interrupted') or type(outcome.get('success')) is not bool:
        raise ValueError('incomplete or invalid task outcome')
    if outcome.get('agent_status') == 'unsupported' and outcome['success']:
        raise ValueError('unsupported task cannot be successful')
    if outcome['success'] and not all(g['success'] for g in outcome['goal_results']):
        raise ValueError('task success disagrees with required goals')
    if 'evaluation' in outcome:
        metric=outcome['evaluation']
        if metric.get('version') not in SUPPORTED_METRIC_VERSIONS or metric.get('score_max') != 100:
            raise ValueError('unsupported evaluation metric version/scale')
        score=metric.get('score')
        if score is not None and (type(score) not in (int,float) or not math.isfinite(score) or not 0 <= score <= 100):
            raise ValueError('invalid evaluation score')
        return score
    return None


def conditions(truth, names):
    if not truth:
        return ['未形成可下发运行时 GT；排除，不用离线位置代替。']
    result = []
    for goal in truth['goals']:
        oid = goal['target']['object_ids'][0]
        name = names.get('objects', {}).get(oid, {}).get('en', oid)
        kind = goal['goal_kind']
        if kind in NAVIGATION:
            destination = '实际建立的地点范围' if kind == 'revisit_place' else '运行时目标包围盒'
            text = f'{name}：距{destination} ≤ {goal["success_criteria"].get("position_tolerance_m", .8):g} m'
            if kind == 'establish_place':
                text += '，以实际到达位置建立地点'
        elif kind == 'transfer':
            room_id = goal['success_criteria']['room_id']
            room = names.get('areas', {}).get(room_id, {}).get('en', room_id)
            text = f'{name}：在 {room} 内成功放置、释放，底部距实际地面 ≤ 0.04 m；须有成功 place 事件'
        else:
            expected = goal['expected_outcome']
            accepted = expected.get('accepted_answers', [expected.get('answer')])
            text = f'{name}：{goal.get("query")}；答案 {accepted}'
            text += '；仅按答案正确性计 100／0 分，无观察或动作过程分'
        result.append(text)
    if len(result) > 1:
        result.append('须依次完成全部目标，后一个到达样本必须晚于前一个。')
    return result


def explain_result(row):
    reason = row['reason_code']
    messages = {
        'success': '完整任务及全部必要目标按运行时 GT 达标。',
        'simplenav_unsupported_qa': 'SimpleNav 返回不支持问答，未提供合格回答；任务已下发，计 0 分。',
        'simplenav_unsupported_manipulation': 'SimpleNav 返回不支持搬运，未完成规定的放置与释放；任务已下发，计 0 分。',
        'already_satisfied': '下发前整个任务已经满足，未下发，N/A，不进入分母。',
        'unavailable': '前序失败使所需真实经历无法成立，修补后仍不可评测；未下发，N/A，不进入分母。',
        'step_limit': f'达到本次动作上限（{row["steps"]} 步），未完成完整任务。',
    }
    text = messages.get(reason, str(reason))
    nav = row.get('navigation')
    if nav and row['success'] is False:
        active = next((g for g in nav['goals'] if not g['ordered_success']), None)
        if active is None:
            return text + ' 几何到达证据不能覆盖运行日志中记录的任务失败原因。'
        text += f' 按序完成 {nav["goals_completed_in_order"]}/{nav["goals_total"]}；当前未完成目标 {active["name"]}。'
        if active['active_closest_distance_m'] is not None:
            text += f' 激活后最近距离 {active["active_closest_distance_m"]:.3f} m，达标容差 {active["tolerance_m"]:g} m。'
    return text


def _index(records, label):
    result = {}
    for number, item in records:
        key = item['task_id']
        if key in result:
            raise ValueError(f'duplicate {label}: {key}')
        result[key] = (number, item)
    return result


def analyze_run(root, *, near_margin_m=.5, rescore=False, reference_dataset=None, reference_output=None):
    root = Path(root).resolve()
    if not math.isfinite(near_margin_m) or near_margin_m < 0:
        raise ValueError('near margin must be finite and nonnegative')
    sources = set()
    config = json.loads((root / 'run_config.json').read_text())
    sources.add(root / 'run_config.json')
    if reference_dataset and (not rescore or not reference_output):
        raise ValueError('reference reconstruction requires explicit rescore and a new output directory')
    reconstruction = ReconstructedReference(reference_dataset, config, reference_output) if reference_dataset else None
    if reconstruction:
        sources.update(reconstruction.sources)
    status_path = root/'status.json'
    status = json.loads(status_path.read_text()) if status_path.is_file() else None
    if status is not None:
        sources.add(status_path)
    names = {}
    cache_path = Path(config['runtime_scene_cache']) if config.get('runtime_scene_cache') else None
    if cache_path and cache_path.is_file():
        names = json.loads(cache_path.read_text()).get('names', {})
        sources.add(cache_path)
    rows, issues, session_checks = [], [], []
    paths = list(iter_sessions(root))
    if not paths:
        raise ValueError('run contains no sessions')
    for scene_id, sid, path in paths:
        def records(filename, *, required=True):
            file = session_file(path, filename)
            if not file.exists() and not required:
                return []
            sources.add(file)
            return read_lines(file)
        outcomes = _index(records('request_outcomes.jsonl'), 'outcome')
        actions, issued, videos = defaultdict(list), {}, {}
        requests, boundaries = defaultdict(list), defaultdict(list)
        for _, event in records('action_changelog.jsonl', required=False):
            actions[event['task_id']].append(event)
        for _, event in records('agent_io.jsonl'):
            if rescore and event.get('request_type') == 'step' and event.get('direction') in {'request','response'}:
                requests[event['task_id']].append({k:event[k] for k in ('direction','timestamp') if k in event})
            if event.get('direction') == 'request' and event.get('request_type') == 'step':
                issued.setdefault(event['task_id'], event['payload']['instruction'])
        if rescore:
            for _, event in records('runtime_events.jsonl', required=False):
                if event.get('event') in {'task_start','task_complete'}:
                    boundaries[event['task_id']].append({k:event[k] for k in ('event','monotonic_ns')})
        for _, event in records('video_index.jsonl', required=False):
            videos[event['event_id']] = event['video_seconds']
        metrics_file = path / 'metrics.json'
        sources.add(metrics_file)
        metrics = json.loads(metrics_file.read_text())
        metrics_tasks = {t['task_id']: (t.get('success'), t.get('scored'), t.get('reason_code')) for t in metrics['task_results']}
        metrics_evaluations = {t['task_id']:t.get('evaluation') for t in metrics['task_results']}
        session_checks.append({'session_id': sid, 'run_status': metrics.get('run_status'),
                               'evaluation_valid': metrics.get('evaluation_valid')})
        if metrics.get('run_status') != 'completed' or metrics.get('evaluation_valid') is not True:
            issues.append({'session_id': sid, 'scope': 'task_score', 'error': 'session is not complete and valid'})
        task_ids = []
        for line, task in records('runtime_tasks.jsonl'):
            public, repair = task['original'], task['repair']
            tid = public['task_id']; task_ids.append(tid)
            outcome_line, outcome = outcomes[tid]
            truth = repair.get('runtime_ground_truth')
            task_actions = actions[tid]
            identity = {'scene_id': scene_id, 'session_id': sid, 'task_id': tid}
            score = None
            try:
                score = task_score(outcome)
                if metrics_tasks.get(tid) != (outcome.get('success'), outcome.get('scored'), outcome.get('reason_code')):
                    raise ValueError('outcome disagrees with session metrics')
                if outcome.get('scored') is not False and (not truth or truth.get('state_assumption') != 'actual_predispatch_state'):
                    raise ValueError('scored task missing actual runtime GT')
                if (tid in issued) != (outcome.get('dispatched') is True):
                    raise ValueError('dispatch log disagrees with outcome')
                if score is not None and issued[tid] != repair['public_task']['instruction']:
                    raise ValueError('dispatched instruction differs from runtime request')
                if outcome.get('evaluation') != metrics_evaluations[tid]:
                    raise ValueError('outcome evaluation disagrees with session metrics')
            except (KeyError, ValueError) as exc:
                score = None
                issues.append({**identity, 'scope': 'task_score', 'error': str(exc)})
            evidence = [{k: v for k, v in g.items() if k not in {'actions', 'observations'}} for g in outcome.get('goal_results', [])]
            row = {**identity, 'order': len(rows) + 1, 'instruction': issued.get(tid),
                'original_instruction': public['instruction'], 'issued_at': public.get('issued_at'),
                'goal_kinds': [g['goal_kind'] for g in (truth or task['original_ground_truth'])['goals']],
                'target_names': [names.get('objects', {}).get(g['target']['object_ids'][0], {}).get('en', g['target']['object_ids'][0])
                                 for g in (truth or task['original_ground_truth'])['goals']],
                'formal_task_score': None if rescore else score, 'excluded': outcome.get('scored') is False,
                'original_task_score': score,
                'original_metric_version':outcome.get('evaluation',{}).get('version',None),
                'success': outcome.get('success'), 'reason_code': outcome.get('reason_code'),
                'agent_status': outcome.get('agent_status'), 'steps': outcome.get('steps', 0),
                'metric_version':METRIC_VERSION if rescore else outcome.get('evaluation',{}).get('version',None),
                'evaluation':None if rescore else outcome.get('evaluation'), 'completion_score':100 if outcome.get('success') is True else 0 if not outcome.get('scored') is False else None,
                'prediction': outcome.get('prediction'), 'goal_results': evidence,
                'runtime_gt': compact_gt(truth), 'original_gt': compact_gt(task['original_ground_truth']),
                'success_conditions': conditions(truth, names), 'repair_record': repair['repair_record'],
                'failure_causes': outcome.get('failure_causes', []), 'navigation': None,
                'runtime_gt_source': {'file': str(session_file(path, 'runtime_tasks.jsonl')), 'line': line},
                'outcome_source': {'file': str(session_file(path, 'request_outcomes.jsonl')), 'line': outcome_line},
                'video': {'file': str(session_file(path, 'review.mp4')), 'seconds': videos.get(task.get('decision_event_id'), 0)} if (session_file(path, 'review.mp4')).is_file() else None}
            if rescore and row['excluded']:
                row['evaluation']=evaluate_metrics(None,task['state_before'],task['state_before'],[],[],outcome)
            if truth and not row['excluded']:
                try:
                    if len(task_actions) != outcome['steps'] or [a['task_step'] for a in task_actions] != list(range(outcome['steps'])):
                        raise ValueError('action log is incomplete or out of order')
                    if rescore or outcome.get('evaluation'):
                        reference=task.get('metric_reference',{'status':'unavailable','reason':'reference_not_recorded'})
                        grid=load_reference(reference,path)
                        if reference.get('status')=='ready':
                            sources.add(path/reference['file'])
                        elif (rescore and reconstruction and outcome.get('agent_status')!='unsupported'
                              and any(g['goal_kind'] in NAVIGATION | {'transfer'} for g in truth['goals'])):
                            reference,grid=reconstruction.prepare(scene_id,sid,truth,task['state_before'])
                        poses=[a[phase] for a in task_actions for phase in ('before','after')] or [task['state_before']['robot_pose']]
                        # Reconstruct the final state from the saved initial
                        # snapshot and actual state changes, never offline GT.
                        after=final_state(task['state_before'],task_actions)
                        recomputed=evaluate_metrics(truth,task['state_before'],after,poses,task_actions,outcome,
                            reference=reference,grid=grid,timing=outcome.get('task_timing'),metric_version=row['metric_version'])
                        if not rescore and recomputed!=outcome['evaluation']:
                            raise ValueError('recomputed evaluation differs from logged metric')
                        if rescore:
                            if not outcome.get('task_timing'):
                                recomputed['measurements'].update(historical_measurements(boundaries[tid],requests[tid]))
                            row['formal_task_score']=recomputed['score']
                            if recomputed['score'] is None:
                                issues.append({**identity,'scope':'task_score','error':recomputed.get('reason','metric unavailable')})
                        row['evaluation']=recomputed
                    nav = navigation_diagnostics(truth, task['state_before'], task_actions, outcome, near_margin_m=near_margin_m)
                    row['navigation'] = nav
                    if nav:
                        for goal in nav['goals']:
                            goal['name'] = names.get('objects', {}).get(goal['object_id'], {}).get('en', goal['object_id'])
                        for reset in nav['reset_recoveries']:
                            reset['video_seconds'] = videos.get(reset.get('event_id'))
                except (KeyError, ValueError, OSError) as exc:
                    row['navigation_error'] = str(exc)
                    issues.append({**identity, 'scope': 'navigation', 'error': str(exc)})
                    if rescore or outcome.get('evaluation'):
                        row['formal_task_score']=None
                        issues.append({**identity,'scope':'task_score','error':str(exc)})
            row['result_explanation'] = explain_result(row)
            rows.append(row)
        if len(task_ids) != len(set(task_ids)) or set(task_ids) != set(outcomes) or set(task_ids) != set(metrics_tasks):
            raise ValueError(f'{sid}: task/outcome/metrics sets differ')
        for _, error in records('integration_errors.jsonl', required=False):
            issues.append({'session_id': sid, 'scope': 'task_score', 'error': error})
    totals = summarize(rows)
    totals.update(metric_breakdown(rows))
    totals['task_score_evidence_valid'] = not any(i['scope'] == 'task_score' for i in issues)
    if not totals['task_score_evidence_valid']:
        totals['formal_rule_average'] = None
    provenance = []
    for file in sorted(sources):
        provenance.append({'file': str(file), 'bytes': file.stat().st_size, 'sha256': sha256_file(file)})
    return {'schema_version': 1, 'private': True, 'run_directory': str(root),
        'rescored':rescore, 'reference_reconstruction':reconstruction.provenance if reconstruction else None,
        'original_run_official': config.get('official'), 'original_status': status, 'near_margin_m': near_margin_m,
        'summary': totals, 'session_checks': session_checks, 'issues': issues,
        'tasks': rows, 'source_files': provenance,
        'method': {
            'formal': '0-100 progress_path_v3; exclusions N/A; missing evidence remains unscored',
            'distance': 'Euclidean XY point-to-runtime-AABB distance, same as v3 scorer; not navmesh distance',
            'remaining': 'r=max(distance-tolerance,0)',
            'progress': '100*clip(1-r/r_start,0,1); r_start=0: 100 inside allowed range, otherwise 0',
            'ordered_progress': '100*(completed_prefix+active_goal_progress)/goal_count; leg baseline at previous ordered arrival; inactive goals contribute zero; completed visits remain completed',
            'best_progress': 'maximum of the ordered progress series, not the sum of independent per-target maxima',
            'reset_progress': 'progress includes actual positions after resets; teleport assistance is separately measured, not attributed to normal walking',
            'walking': 'sum of logged non-reset action endpoint XY displacements; reset displacements listed separately',
            'collision': 'actions with collision=true; rate=colliding translation commands/translation commands; also report all-action rate',
            'recovery': 'same active target at reset; window ends at next reset, task end, or ordered completion of this target; teleport gain separated from subsequent walking gain',
            'efficiency': 'successful reset-free routes only: sum of leg straight-line remaining distances / walking length, capped at 1; geometric lower bound, not SPL or navigable shortest-path efficiency',
            'recognition': 'geometric proximity does not establish recognition',
            'near_band': f'remaining distance <= {near_margin_m:g} m; diagnostic label only, no change to success threshold',
            'repetition': 'at least two collision events / applied resets; labels may overlap'}}


def summarize(rows):
    scored = [r for r in rows if not r['excluded']]
    nav = [r['navigation'] for r in rows if r['navigation']]
    resets = [reset for n in nav for reset in n['reset_recoveries']]
    move_attempts = sum(n['motion']['move_attempts'] for n in nav)
    return {'tasks_total': len(rows), 'tasks_evaluated': len(scored),
        'successes': sum(r['success'] is True for r in scored),
        'failures': sum(r['success'] is False for r in scored),
        'metric_versions':sorted({r['metric_version'] for r in scored if r.get('metric_version')}),
        'tasks_with_metric':sum(r['formal_task_score'] is not None for r in scored),
        'metrics_complete':all(r['formal_task_score'] is not None for r in scored),
        'task_success_rate':sum(r['success'] is True for r in scored)/len(scored) if scored else None,
        'tasks_missing_metric':[{k:r[k] for k in ('scene_id','session_id','task_id')} for r in scored if r['formal_task_score'] is None],
        'tasks_excluded': sum(r['excluded'] for r in rows),
        'exclusions': dict(Counter(r['reason_code'] for r in rows if r['excluded'])),
        'formal_rule_average': sum(r['formal_task_score'] for r in scored) / len(scored)
            if scored and all(r['formal_task_score'] is not None for r in scored)
            and len(set(r.get('metric_version',None) for r in scored))==1 else None,
        'outcomes': dict(Counter(r['reason_code'] for r in rows)),
        'navigation_tasks': len(nav),
        'navigation_successes': sum(r['success'] is True for r in rows if r['navigation']),
        'navigation_final_progress_mean': sum(n['final_progress_score'] for n in nav) / len(nav) if nav else None,
        'navigation_best_progress_mean': sum(n['best_progress_score'] for n in nav) / len(nav) if nav else None,
        'walking_length_m': sum(n['motion']['walking_length_m'] for n in nav),
        'reset_displacement_m': sum(n['motion']['reset_displacement_m'] for n in nav),
        'collisions': sum(n['motion']['collision_count'] for n in nav) if all(n['motion']['collision_count'] is not None for n in nav) else None,
        'move_attempts': move_attempts,
        'collision_rate': (sum(n['motion']['move_collision_count'] or 0 for n in nav) / move_attempts
                           if move_attempts and all(n['motion']['move_collision_count'] is not None for n in nav) else None),
        'resets': len(resets), 'reset_categories': dict(Counter(r['category'] for r in resets)),
        'resets_with_new_best_from_walking': sum(r['new_best_from_walking'] for r in resets),
        'categories': dict(Counter(c for n in nav for c in n['categories'])),
        'failed_navigation_categories': dict(Counter(c for r in rows if r['navigation'] and r['success'] is False for c in r['navigation']['categories']))}
