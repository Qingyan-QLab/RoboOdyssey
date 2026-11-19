"""Frozen synthetic regression cases; selecting a case never generates tasks."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from roboodyssey.artifacts.layout import sha256_file, write_json as write
from roboodyssey.contracts.navigation_io import write_navigation_npz
from roboodyssey.contracts.state import build_runtime_cache, digest

_DATA = json.loads((Path(__file__).parent/'fixtures/contextual.json').read_text())
SOURCE = {'session_id': 'session_000', 'task_id': 'move', 'goal_id': 'transfer_book_red'}


def key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def goal(oid, kind='visit', **kwargs):
    return {'goal_id': kind+'_'+oid, 'kind': kind, 'object_id': oid, **kwargs}


def task(tid, *objectives):
    return {'task_id': tid, 'objectives': list(objectives)}


def fixture_v3(tasks=None, *, second=None):
    inputs = deepcopy(_DATA['inputs'])
    plan = deepcopy(_DATA['fixture_plans'][key([tasks, second])])
    context = SimpleNamespace(inputs=inputs, metadata=inputs['metadata'])
    return context, inputs, plan


def select_scenario(context, plan):
    selection = deepcopy(plan)
    for name in ('scene_reference', 'sequence_id', 'mutation_sampling'):
        selection.pop(name, None)
    result = deepcopy(_DATA['scenarios'][key(selection)])
    result['language_catalog'] = deepcopy(_DATA['language_catalog'])
    result['plan'] = deepcopy(plan)
    return result


def runtime_cache(inputs, plan):
    cache = build_runtime_cache(inputs['metadata'], inputs['understanding'], inputs['language_profile'])
    cache['names'] = deepcopy(plan['names'])
    cache['original_area_descriptions'] = {rid: area['text'] for rid, area in cache['areas'].items()}
    for rid, entry in plan['names']['areas'].items():
        cache['areas'][rid]['text'] = entry['en']
        cache['areas'][rid]['anchor_object_ids'] = []
    cache.pop('cache_key', None)
    cache['cache_key'] = digest(cache)
    return cache


def write_dataset(root, result, context, inputs):
    """Materialize a previously frozen test case, including authentic inventories."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root/'private/scene').mkdir(parents=True)
    manifest = {'schema_version': 5, 'dataset_schema': 'cl_task_sequence_v1', 'layout_version': 1,
        'scene_id': inputs['metadata']['scene']['scene_id'], 'scene_asset': inputs.get('scene_asset'),
        'runtime_contract': {'name': 'contextual_requests', 'version': 3}, 'sessions': []}
    shared = inputs.get('scene_version')
    for name in ('sequence_id', 'scene_reference'):
        if name in result['plan']:
            manifest[name] = deepcopy(result['plan'][name])
    if shared:
        manifest.update(layout_version=2, scene_asset={'scope': 'scene_version'})
    artifacts = {'scene_metainfo': inputs['metadata'], 'scene_understanding': inputs['understanding'],
        'manipulation': inputs['manipulation'], 'public_language_profile': inputs['language_profile'],
        'runtime_scene_cache': runtime_cache(inputs, result['plan']), 'plan': result['plan'],
        'language_catalog': result['language_catalog'], 'coverage_report': result['coverage'],
        'audit_report': {'status': 'frozen_synthetic_fixture'}}
    for name, value in artifacts.items():
        if shared and name in shared.manifest['artifacts']:
            manifest[name] = {'scope': 'scene_version', 'key': name}
        else:
            manifest[name] = f'private/{name}.json' if name != 'scene_metainfo' else 'scene_metainfo.json'
            write(root/manifest[name], value)
    if shared:
        for name in ('navigation', 'environment_setup'):
            manifest[name] = {'scope': 'scene_version', 'key': name}
    else:
        manifest['navigation'] = 'private/navigation.npz'
        write_navigation_npz(root/manifest['navigation'], inputs['navigation'])
        if inputs['navigation'].get('environment_setup'):
            manifest['environment_setup'] = 'private/environment_setup.json'
            write(root/manifest['environment_setup'], inputs['navigation']['environment_setup'])
    for session in result['sessions']:
        sid = session['public']['session_id']
        entry = {'session_id': sid, 'session': f'sessions/{sid}/session.json'}
        write(root/entry['session'], session['public'])
        for name in ('ground_truth', 'templates', 'mutations'):
            entry[name] = f'private/sessions/{sid}/{name}.json'
            write(root/entry[name], session[name])
        manifest['sessions'].append(entry)
    manifest['files'] = {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob('*') if p.is_file()}
    write(root/'dataset_manifest.json', manifest)
    if shared:
        index_path = shared.root/'scene_manifest.json'
        index = json.loads(index_path.read_text())
        index['sequences'][manifest['sequence_id']] = {
            'manifest': f"sequences/{manifest['sequence_id']}/dataset_manifest.json",
            'sha256': sha256_file(root/'dataset_manifest.json'), 'revision': shared.revision}
        write(index_path, index)
    return manifest


def prepare_scene(root, context, inputs, names):
    """Assemble a shared-layout fixture using existing frozen scene values."""
    root = Path(root); revision = 'r_fixture'
    directory = root/'scene_versions'/revision
    directory.mkdir(parents=True)
    artifacts = {'scene_metainfo': inputs['metadata'], 'scene_understanding': inputs['understanding'],
        'manipulation': inputs['manipulation'], 'public_language_profile': inputs['language_profile'],
        'names': names, 'environment_setup': inputs['navigation'].get('environment_setup'),
        'initial_state': {'robot_base_pose': {'position': [1.1, 1.1, 0], 'orientation': [0,0,0,1]}}}
    manifest = {'schema_version': 1, 'format': 'cl_scene_version_v1',
        'scene_id': inputs['metadata']['scene']['scene_id'], 'revision': revision,
        'scene_asset': None, 'artifacts': {k: k+'.json' for k in artifacts}}
    for name, value in artifacts.items():
        write(directory/manifest['artifacts'][name], value)
    manifest['artifacts']['navigation'] = 'navigation.npz'
    write_navigation_npz(directory/'navigation.npz', inputs['navigation'])
    manifest['initial_state_sha256'] = digest(artifacts['initial_state'])
    manifest['files'] = {p.name: sha256_file(p) for p in directory.iterdir()}
    write(directory/'scene_version.json', manifest)
    version = SimpleNamespace(root=root, revision=revision, manifest=manifest)
    reference = {'revision': revision, 'manifest_sha256': sha256_file(directory/'scene_version.json')}
    version.reference = lambda: deepcopy(reference)
    shared = deepcopy(inputs); shared['scene_version'] = version
    version.inputs = lambda: (context, shared)
    write(root/'scene_manifest.json', {'schema_version': 1, 'format': 'cl_scene_collection_v1',
        'scene_id': manifest['scene_id'], 'default_revision': revision,
        'versions': {revision: {'manifest': f'scene_versions/{revision}/scene_version.json',
                              'sha256': reference['manifest_sha256']}}, 'sequences': {}})
    return version


def shared_plan(plan, version, sequence_id):
    result = deepcopy(plan)
    result.update(sequence_id=sequence_id, scene_reference=version.reference())
    return result
