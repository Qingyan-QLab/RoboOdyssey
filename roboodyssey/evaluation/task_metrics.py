"""Versioned partial task scores; success remains a separate strict predicate."""
from ..interaction.protocol import required_action_name as _name
from copy import deepcopy
import math
from pathlib import Path

from .navigation import GridDistance, OrderedRoute
from .metrics.qa import contextual_answer_correct
from .transfer import TransferReference, TRANSFER_POLICY, transfer_metric
from ..artifacts.layout import local_path, safe_id, sha256_file


METRIC_VERSION = 'progress_path_v3'
SUPPORTED_METRIC_VERSIONS = {METRIC_VERSION}
NAVIGATION = {'visit', 'establish_place', 'revisit_place'}
METRIC_POLICY = {
    'version': METRIC_VERSION, 'range': [0, 100],
    'navigation': 'ordered_geodesic_progress_times_path_efficiency',
    'navigation_reference': 'actual_task_start_grid_with_avatar_and_carried_door_clearance',
    'reset_cost': 'max(grid_geodesic_between_endpoints, displacement, grid_resolution)',
    'transfer': 'remaining_delivery_cost_progress_times_efficiency',
    'transfer_policy': TRANSFER_POLICY,
    'answer': 'binary_answer_correctness_only',
    'task_aggregation': 'mean_of_required_goal_scores',
    'missing_reference': 'null_without_removing_task_from_denominator',
}


def prepare_reference(env, truth, session):
    """Freeze real task-start geometry. A missing reference is explicit, not zero."""
    transfer = any(g['goal_kind'] == 'transfer' for g in truth['goals'])
    if not transfer and not any(g['goal_kind'] in NAVIGATION for g in truth['goals']):
        return {'status': 'not_required'}, None
    if transfer:
        if not hasattr(env, 'evaluation_transfer'):
            return {'status': 'unavailable', 'reason': 'environment_has_no_transfer_snapshot'}, None
        try:
            grid = TransferReference(**env.evaluation_transfer(truth['goals']))
        except (KeyError, ValueError) as exc:
            return {'status': 'unavailable', 'reason': str(exc)}, None
    else:
        if not hasattr(env, 'evaluation_navigation'):
            return {'status': 'unavailable', 'reason': 'environment_has_no_navigation_snapshot'}, None
        grid = GridDistance(**env.evaluation_navigation())
    folder = Path(session)/'evaluation'
    folder.mkdir(exist_ok=True)
    file = folder/(safe_id(truth['task_id'])+'.npz')
    grid.save(file)
    reference = {'status': 'ready', 'file': file.relative_to(session).as_posix(),
        'sha256': sha256_file(file), 'distance_type': 'grid_geodesic_8',
        'resolution_m': grid.resolution, 'robot_radius_m': grid.radius,
        'carried_door_radius_m': grid.carried_radius}
    if transfer:
        reference.update(kind='transfer_cost', cost_unit=TRANSFER_POLICY['cost_unit'],
                         move_step_m=grid.step)
    return reference, grid


def load_reference(reference, session):
    if reference.get('status') != 'ready':
        return None
    file = local_path(session, reference['file'])
    if sha256_file(file) != reference['sha256']:
        raise ValueError('navigation reference checksum mismatch')
    return TransferReference.load(file) if reference.get('kind') == 'transfer_cost' else GridDistance.load(file)


def _finite(value):
    return value if math.isfinite(value) else None


def navigation_metric(truth, grid, poses, actions):
    """Use only the ordered completed prefix; never sum independent best passes."""
    goals = [g for g in truth['goals'] if g['goal_kind'] in NAVIGATION]
    route = OrderedRoute(grid, goals)
    if not poses:
        raise ValueError('navigation metric requires actual poses')
    points = [p['position'][:2] for p in poses]
    for event in actions:
        if _name(event) in {'pick','place','open','close'} and event['result'].get('success'):
            raise ValueError('navigation topology changed during task; frozen path reference is not comparable')
        if event.get('state_changes', {}).get('objects'):
            raise ValueError('navigation world changed during task; frozen path reference is not comparable')
    if len(points) != 2*len(actions) and actions:
        raise ValueError('navigation poses do not align with action before/after samples')
    if actions and any(math.dist(points[2*i+phase], event[key]['position'][:2]) > 1e-5
                       for i,event in enumerate(actions) for phase,key in enumerate(('before','after'))):
        raise ValueError('navigation pose/action evidence mismatch')
    if any(math.dist(a['after']['position'][:2], b['before']['position'][:2]) > 1e-5
           for a,b in zip(actions, actions[1:])):
        raise ValueError('unlogged trajectory discontinuity')
    reference = route.distance(points[0], 0, remaining_route=True)
    if not math.isfinite(reference):
        raise ValueError('runtime ordered route is unreachable on the task-start grid')
    baseline = route.distance(points[0], 0)
    stage, walk, resets, reset_cost = 0, 0.0, 0, 0.0
    records, hits = [], []
    for index, point in enumerate(points):
        if index % 2 and actions:
            event = actions[index//2]
            displacement = math.dist(event['before']['position'][:2], point)
            if _name(event) == 'reset':
                if type(event['result'].get('reset_applied')) is not bool:
                    raise ValueError('reset result lacks reset_applied evidence')
                if not event['result']['reset_applied'] and displacement > 1e-6:
                    raise ValueError('reset moved without reset_applied evidence')
                if event['result']['reset_applied']:
                    resets += 1
                    charge = grid.between(event['before']['position'][:2], point)
                    if not math.isfinite(charge):
                        raise ValueError('reset endpoints are disconnected on task-start grid')
                    reset_cost += max(charge, displacement, grid.resolution)
            else:
                if displacement > 1e-6 and not grid.connected_segment(event['before']['position'][:2], point):
                    raise ValueError('logged movement crosses the frozen collision grid')
                walk += displacement
        arrived = stage < len(goals) and route.reached(point, stage)
        if arrived:
            hits.append(index)
            stage += 1
            baseline = route.distance(point, stage) if stage < len(goals) else 0.0
        current = route.distance(point, stage) if stage < len(goals) else 0.0
        if not math.isfinite(current):
            fraction = 0.0
        elif stage == len(goals):
            fraction = 0.0
        elif arrived:
            fraction = 0.0  # A second visit requires a strictly later sample.
        elif baseline == 0:
            fraction = float(current == 0)
        else:
            fraction = max(0.0, min(1.0, 1-current/baseline))
        progress = (stage+fraction)/len(goals)
        remaining = route.distance(point, stage, remaining_route=True)
        charged = walk+reset_cost
        if not math.isfinite(remaining):
            efficiency = 0.0
        elif reference <= 1e-9:
            efficiency = float(charged <= 1e-9 and remaining <= 1e-9)
        else:
            efficiency = reference/max(reference, charged+remaining)
        records.append({'pose_index': index, 'completed': stage, 'progress': progress,
            'efficiency': efficiency, 'score': 100*progress*efficiency,
            'active_distance_m': _finite(current), 'remaining_route_m': _finite(remaining)})
    final = records[-1]
    return {'distance_type': 'grid_geodesic_8', 'reference_length_m': reference,
        'remaining_route_m': final['remaining_route_m'], 'walking_length_m': walk,
        'reset_count': resets, 'reset_cost_m': reset_cost, 'charged_length_m': walk+reset_cost,
        'progress_score': 100*final['progress'], 'path_efficiency': final['efficiency'],
        'score': final['score'], 'best_score': max(r['score'] for r in records),
        'best_progress_score': 100*max(r['progress'] for r in records),
        'goals_completed': stage, 'goal_hits': hits, 'goals_total': len(goals),
        'timeline': records}


def evaluate_metrics(truth, before, after, poses, actions, outcome, *, reference=None, grid=None,
                     timing=None, metric_version=METRIC_VERSION):
    if metric_version not in SUPPORTED_METRIC_VERSIONS:
        raise ValueError('unsupported evaluation metric version: '+str(metric_version))
    reference = reference or {'status': 'unavailable', 'reason': 'reference_not_recorded'}
    value = {'version': metric_version, 'score_max': 100, 'score': None,
        'status': 'ready', 'reference': deepcopy(reference), 'navigation': None,
        'goal_scores': [], 'measurements': {'steps': len(actions), **(timing or {})},
        'step_efficiency_score': None, 'time_efficiency_score': None,
        'cost_reference_status': 'no_verified_gt_steps_or_wall_time'}
    value['transfers'] = []
    if timing:
        for key in ('elapsed_seconds','agent_wait_seconds','action_seconds'):
            if not math.isfinite(timing[key]) or timing[key] < 0:
                raise ValueError('invalid task timing: '+key)
        if abs(timing['finished_monotonic_s']-timing['started_monotonic_s']-timing['elapsed_seconds']) > 1e-6:
            raise ValueError('task timing boundary mismatch')
        if timing['agent_wait_seconds']+timing['action_seconds'] > timing['elapsed_seconds']+1e-6:
            raise ValueError('task timing component exceeds elapsed time')
    if outcome.get('scored') is False:
        return {**value, 'status': 'excluded'}
    if outcome.get('agent_status') == 'unsupported':
        return {**value, 'score': 0.0}
    evidence = {g['goal_id']: g for g in outcome['goal_results']}
    nav_goals = [g for g in truth['goals'] if g['goal_kind'] in NAVIGATION]
    nav = None
    if nav_goals:
        if grid is None:
            value.update(status='unavailable', reason=reference.get('reason', 'navigation_reference_missing'))
        else:
            try:
                if math.dist(before['robot_pose']['position'][:2],poses[0]['position'][:2]) > 1e-4:
                    raise ValueError('trajectory start differs from predispatch state')
                nav = navigation_metric(truth, grid, poses, actions)
                expected = [evidence[g['goal_id']].get('pose_index') for g in nav_goals if evidence[g['goal_id']]['success']]
                if nav['goal_hits'] != expected:
                    raise ValueError('metric ordered visits disagree with success evidence')
                value['navigation'] = nav
            except ValueError as exc:
                nav = None
                value.update(status='unavailable', reason=str(exc))
    nav_index = 0
    for goal in truth['goals']:
        item = evidence[goal['goal_id']]
        kind, oid = goal['goal_kind'], goal['target']['object_ids'][0]
        components = {}
        if kind in NAVIGATION:
            if nav:
                completed = nav['goals_completed']
                partial = max(0.0, nav['progress_score']/100*len(nav_goals)-completed)
                progress = 1.0 if nav_index < completed else partial if nav_index == completed else 0.0
                score = 100*progress*nav['path_efficiency']
            else:
                score = None
            nav_index += 1
        elif kind == 'transfer':
            acquired = (before.get('held_object') == oid or before['objects'][oid].get('held') is True or any(
                _name(a) == 'pick' and a['result'].get('success') is True
                and a['result'].get('object_id') == oid for a in actions))
            destination = acquired and after['objects'][oid]['room_id'] == goal['success_criteria']['room_id']
            components = {'acquired': bool(acquired or item['success']),
                          'at_destination': bool(destination or item['success']),
                          'valid_place_and_release': item['success'] is True}
            score = None
            try:
                if not isinstance(grid, TransferReference):
                    raise ValueError(reference.get('reason', 'transfer_cost_reference_missing'))
                metric = transfer_metric(goal, grid, before, after, actions, item['success'])
                score = metric['score']
                value['transfers'].append(metric)
            except (KeyError, ValueError) as exc:
                value.update(status='unavailable', reason=str(exc))
        elif kind == 'answer':
            if 'prediction' in outcome:
                correct = contextual_answer_correct(outcome['prediction'], goal['expected_outcome'])
            else:
                correct = item.get('answer_correct')
            components = {'answer_correct': correct}
            if type(correct) is bool:
                score = 100*float(correct)
            else:
                score = None
                value.update(status='unavailable', reason='answer_correctness_evidence_missing')
        else:
            raise ValueError('unsupported metric goal kind: '+str(kind))
        value['goal_scores'].append({'goal_id': goal['goal_id'], 'kind': kind,
            'score': score, 'components': components})
    scores = [g['score'] for g in value['goal_scores']]
    if scores and all(score is not None for score in scores):
        value['score'] = sum(scores)/len(scores)
    return value


def metric_totals(results, *, valid=True):
    """Never drop a task with missing metric evidence from the score denominator."""
    scored = [r for r in results if r.get('scored', True)]
    present = [r.get('evaluation', {}) for r in scored]
    versions = {v.get('version') for v in present}
    if not versions:
        versions = {r['evaluation']['version'] for r in results if r.get('evaluation')} or {METRIC_VERSION}
    version = next(iter(versions)) if len(versions) == 1 else 'mixed_incomparable'
    def complete(value):
        return (value.get('version') in SUPPORTED_METRIC_VERSIONS and value.get('score_max') == 100
                and type(value.get('score')) in (int,float) and math.isfinite(value['score'])
                and 0 <= value['score'] <= 100)
    available = version in SUPPORTED_METRIC_VERSIONS and all(complete(v) for v in present)
    return {'metric_version': version, 'score_max': 100,
        'score_100': sum(v['score'] for v in present)/len(scored) if valid and scored and available else None,
        'metrics_complete': bool(available and valid),
        'tasks_with_metric': sum(complete(v) for v in present),
        'tasks_missing_metric': [r['task_id'] for r,v in zip(scored,present) if not complete(v)]}
