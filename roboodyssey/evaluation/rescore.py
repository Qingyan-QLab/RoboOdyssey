"""Explicit CPU rescoring of historical evidence, without rewriting a run."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from ..artifacts.layout import sha256_file, safe_id
from ..contracts.state import apply_object_deltas

from .task_metrics import prepare_reference


def final_state(before, actions):
    after = deepcopy(before)
    for action in actions:
        after['robot_pose'] = deepcopy(action['after'])
        apply_object_deltas(after.get('objects', {}), action.get('state_changes', {}).get('objects', []))
        held = action.get('state_changes', {}).get('held_object')
        if held:
            after['held_object'] = held['after']
    return after


class ReconstructedReference:
    """Rebuild the all-open-doors runtime map from verified inputs and real state.

    This is explicitly reconstructed evidence, never an original live snapshot.
    Other scene policies are refused rather than guessed.
    """
    def __init__(self, dataset, config, output):
        from ..data.archive import dataset_root
        from ..data.frozen import SequenceData, select_sequences
        self.root, self.output = dataset_root(dataset), Path(output).resolve()
        if (self.root/'scene_manifest.json').is_file():
            sid = (config.get('sequence_identity') or {}).get('sequence_id')
            choices = select_sequences(self.root, [sid] if sid else None)
            if len(choices) != 1:
                raise ValueError('reference archive has multiple sequences; historical sequence identity required')
            self.root = choices[0]
        manifest_file = self.root/'dataset_manifest.json'
        expected = config.get('dataset_manifest_sha256')
        if not expected or sha256_file(manifest_file) != expected:
            raise ValueError('reference dataset manifest does not match the historical run')
        manifest = json.loads(manifest_file.read_text())
        frozen = SequenceData(self.root)
        self.scene_id = manifest['scene_id']
        if manifest.get('runtime_contract', {}).get('version') != 3:
            raise ValueError('reference reconstruction requires contextual_requests v3')
        self.sources = {manifest_file}
        def artifact(key):
            file = frozen.path(key)
            if not isinstance(manifest[key], dict):
                if sha256_file(file) != manifest.get('files', {}).get(manifest[key]):
                    raise ValueError('reference dataset artifact checksum mismatch: '+manifest[key])
            self.sources.add(file)
            return file
        self.metadata = json.loads(artifact('scene_metainfo').read_text())
        self.setup = json.loads(artifact('environment_setup').read_text())
        self.cache = json.loads(artifact('runtime_scene_cache').read_text())
        with np.load(artifact('navigation'), allow_pickle=False) as archive:
            self.navigation = {key: archive[key].copy() for key in archive.files}
        if json.loads(str(self.navigation.get('environment_setup_json'))) != self.setup:
            raise ValueError('navigation and environment setup disagree')

        from ..contracts import raster as scene_package
        from roboodyssey.contracts.doors import POLICY
        from ..scenes.isaac_env import IsaacVLNEnv
        if self.setup.get('policy') != POLICY:
            raise ValueError('unsupported reference reconstruction door policy')
        self.sources.update((Path(scene_package.__file__).resolve(), Path(__file__).resolve(),
                             Path(__import__(IsaacVLNEnv.__module__, fromlist=['']).__file__).resolve()))
        self.radius = config.get('motion', {}).get('robot_radius_m')
        if self.radius is None:
            raise ValueError('historical run has no verified robot radius')
        # Prove the current CPU rasterizer reproduces the frozen original map
        # before applying each recorded actual task state, including mutations.
        original_objects = {o['object_id']: {**o, **self.cache['objects'][o['object_id']]['initial']}
                            for o in self.metadata['objects']}
        baseline = self.environment({'objects': original_objects, 'held_object': None})
        for key in ('static_occupancy', 'static_base_occupancy', 'dynamic_occupancy'):
            if not np.array_equal(baseline._navigation[key], self.navigation[key]):
                raise ValueError('rasterizer differs from frozen navigation: '+key)
        self.provenance = {'kind': 'reconstructed_from_verified_dataset_and_actual_task_state',
            'original_live_snapshot': False, 'baseline_grid_exact_match': True,
            'dataset_root': str(self.root), 'dataset_manifest_sha256': expected,
            'robot_radius_m': self.radius}

    def environment(self, before):
        from ..scenes.isaac_env import IsaacVLNEnv, EnvConfig
        env = IsaacVLNEnv.__new__(IsaacVLNEnv)  # No constructor, SDK, stage or GPU.
        env._np = np
        env.config = EnvConfig(robot_body=None, robot_radius=float(self.radius))
        env._environment_setup, env._metadata = self.setup, self.metadata
        env._objects = deepcopy(before['objects'])
        env._held_object = before.get('held_object')
        env._navigation = dict(self.navigation)
        env._resolution = float(self.navigation['resolution'])
        env._origin = self.navigation['origin_xy']
        env._door_open = {door['object_id'] for door in self.setup['doors']}
        env._door_cells = {}  # all-open-doors geometry is baked into the map
        # CPU reconstruction uses captured evidence; it must never fabricate
        # a live USD readback or initialize a simulator.
        env._refresh_vlm_navigation(measured_state=before)
        return env

    def prepare(self, scene_id, session_id, truth, before):
        if scene_id != self.scene_id:
            raise ValueError('reference scene does not match historical task')
        doors = {door['object_id'] for door in self.setup['doors']}
        if set(before.get('door_open', [])) != doors:
            raise ValueError('actual door state differs from frozen all-open-doors policy')
        if set(before['objects']) != set(self.cache['objects']):
            raise ValueError('actual object inventory differs from reference dataset')
        if any(g['goal_kind'] == 'transfer' for g in truth['goals']):
            # Raster provenance cannot recover missing prim origins or certify
            # the old simulator's interaction mechanics. Never infer them.
            return {'status': 'unavailable', 'reason': 'historical_transfer_interaction_reference_missing',
                    'source': self.provenance['kind'], 'original_live_snapshot': False}, None
        env = self.environment(before)
        directory = self.output/'references'/safe_id(scene_id)/safe_id(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        reference, grid = prepare_reference(env, truth, directory)
        reference['file'] = (directory/reference['file']).relative_to(self.output).as_posix()
        reference.update(source=self.provenance['kind'], original_live_snapshot=False,
            state_before_sha256=hashlib.sha256(json.dumps(before, sort_keys=True,
                separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest())
        return reference, grid


def historical_measurements(boundaries, requests):
    """Report recorded spans; never invent new-run timing components or GT cost."""
    result = {'timing_source': 'historical_event_and_agent_io_logs',
              'recorded_task_span_seconds': None, 'recorded_agent_roundtrip_seconds': None,
              'request_count': len([r for r in requests if r['direction'] == 'request'])}
    starts = [e['monotonic_ns'] for e in boundaries if e['event'] == 'task_start']
    ends = [e['monotonic_ns'] for e in boundaries if e['event'] == 'task_complete']
    if len(starts) == len(ends) == 1 and ends[0] >= starts[0]:
        result['recorded_task_span_seconds'] = (ends[0]-starts[0])/1e9
    pending, total, valid = None, 0.0, True
    for record in requests:
        if record['direction'] == 'request':
            if pending is not None:
                valid = False
            pending = record.get('timestamp')
        elif record['direction'] == 'response':
            end = record.get('timestamp')
            if pending is None or end is None or end < pending:
                valid = False
            else:
                total += end-pending
            pending = None
    if valid and pending is None and requests:
        result['recorded_agent_roundtrip_seconds'] = total
    return result


def metric_breakdown(rows):
    """Group task means without silently dropping missing scores."""
    def group(items):
        scored = [r for r in items if not r['excluded']]
        scores = [r['formal_task_score'] for r in scored]
        return {'tasks_total': len(items), 'tasks_evaluated': len(scored),
            'tasks_excluded': len(items)-len(scored),
            'successes': sum(r['success'] is True for r in scored),
            'success_rate': sum(r['success'] is True for r in scored)/len(scored) if scored else None,
            'score_100': sum(scores)/len(scored) if scored and all(v is not None for v in scores) else None}
    def by(key):
        keys = sorted({key(r) for r in rows})
        return {k: group([r for r in rows if key(r) == k]) for k in keys}
    nav = [(r.get('evaluation') or {}).get('navigation') for r in rows]
    nav = [n for n in nav if n]
    transfers = [g for r in rows for g in (r.get('evaluation') or {}).get('transfers', [])]
    measured = [(r.get('evaluation') or {}).get('measurements', {}) for r in rows if not r['excluded']]
    def measurement(key):
        values = [m[key] for m in measured if m.get(key) is not None]
        return {'tasks_with_measurement': len(values), 'sum_seconds': sum(values) if values else None}
    return {'by_session': by(lambda r:r['scene_id']+'/'+r['session_id']),
        'by_task_kind': by(lambda r:'+'.join(r['goal_kinds'])),
        'navigation_metric': {'tasks_with_reference': len(nav),
            **{key:sum(n[key] for n in nav)/len(nav) if nav else None for key in
               ('score','progress_score','best_progress_score','best_score','path_efficiency')},
            **{key:sum(n[key] for n in nav) if nav else None for key in
               ('reference_length_m','walking_length_m','reset_cost_m','charged_length_m')}},
        'transfer_metric': {'goals_with_reference': len(transfers),
            **{key:sum(g[key] for g in transfers)/len(transfers) if transfers else None
               for key in ('score', 'progress_score', 'path_efficiency', 'reference_cost', 'charged_cost')},
            'remaining_cost': sum(g['remaining_cost'] for g in transfers)/len(transfers)
                if transfers and all(g['remaining_cost'] is not None for g in transfers) else None},
        'recorded_timing': {key:measurement(key) for key in
            ('elapsed_seconds','agent_wait_seconds','action_seconds',
             'recorded_task_span_seconds','recorded_agent_roundtrip_seconds')}}
