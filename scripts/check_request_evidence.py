#!/usr/bin/env python3
"""Check actual USD state and fresh RGB/instance frames from published data."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--sequence')
    args = parser.parse_args(argv)
    from roboodyssey.scenes.dataset_loader import BenchmarkDataset
    dataset = BenchmarkDataset(args.dataset, sequence_id=args.sequence)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'active_gpu': args.gpu,
                         'physics_gpu': args.gpu, 'multi_gpu': False})
    env = None
    error = None
    try:
        from roboodyssey.scenes.isaac_backend import make_env_factory
        from roboodyssey.runtime.request_runner import actual_state, public_observation
        scene = dataset.scene_ids()[0]
        sessions = dataset.sessions(scene)
        env = make_env_factory(dataset, app)(scene, dataset.canonical_state(scene), sessions[0].session_id)
        cache = json.loads(dataset._requests.path('runtime_scene_cache').read_text())
        before = actual_state(env, cache)
        # Visibility is diagnostic evidence, never a public model input.
        oid = next(g['target']['object_ids'][0] for s in sessions for t in s.ground_truth() for g in t['goals'])
        requirement = {'object_id': oid, 'max_distance_m': 2, 'min_visible_pixels': 8}
        _, first = env.observe_with_evidence([requirement])
        observation, second = env.observe_with_evidence([requirement])
        if second[0]['frame_id'] <= first[0]['frame_id']:
            raise RuntimeError('instance evidence did not advance to a fresh frame')
        after = actual_state(env, cache)
        if before != after:
            raise RuntimeError('frame capture changed authoritative world state')
        public = public_observation(observation, cache, {})
        if any(k in public for k in ('room_id', 'nearby_manipulable_objects', 'nearby_doors')):
            raise RuntimeError('private affordances leaked into public observation')
        print('REQUEST_ARCHIVE_ENV_SMOKE '+json.dumps({
            'scene': scene, 'sequence': dataset.sequence_identity(), 'actual_objects': len(after['objects']),
            'frame_id': second[0]['frame_id'], 'visible_pixels_at_initial_pose': second[0]['visible_pixels'],
            'capture_preserved_world': True, 'public_fields': list(public)}), flush=True)
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        error = exc
    finally:
        if env is not None:
            env.close()
        app.close(exit_code=1 if error else 0)
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
