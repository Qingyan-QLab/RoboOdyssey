"""Read frozen contracts directly; never compile tasks or replay planned success."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from types import SimpleNamespace

from .archive import dataset_root
from ..artifacts.layout import sha256_file
from ..contracts.language import render_request, validate_catalog
from ..contracts.navigation_io import read_navigation_npz
from ..contracts.state import digest


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'invalid dataset identity: {value!r}')
    return value


def inside(root, reference):
    if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
        raise ValueError(f'dataset reference must be relative: {reference!r}')
    root = Path(root).resolve()
    path = (root/reference).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'dataset reference escapes root: {reference}')
    return path


def check_file(path, expected):
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError(f'invalid checksum for {path}')
    if sha256_file(path) != expected:
        raise ValueError(f'dataset file checksum mismatch: {path}')


def inventory(root, files):
    if not isinstance(files, dict) or not files:
        raise ValueError('dataset missing file inventory')
    for relative, expected in files.items():
        check_file(inside(root, relative), expected)


def collection(root):
    root = dataset_root(root)
    value = read(root/'scene_manifest.json')
    if value.get('schema_version') != 1 or value.get('format') != 'cl_scene_collection_v1':
        raise ValueError('unsupported scene collection version')
    identity(value['scene_id'])
    for group, directory, filename in (('versions', 'scene_versions', 'scene_version.json'),
                                        ('sequences', 'sequences', 'dataset_manifest.json')):
        if not isinstance(value.get(group), dict) or not value[group]:
            raise ValueError(f'collection has no {group}')
        for key, entry in value[group].items():
            identity(key)
            if entry.get('manifest') != f'{directory}/{key}/{filename}':
                raise ValueError(f'noncanonical collection reference: {key}')
            check_file(inside(root, entry['manifest']), entry['sha256'])
    if value.get('default_revision') not in value['versions']:
        raise ValueError('unknown default scene revision')
    return value


def select_sequences(source, ids=None):
    root = dataset_root(source)
    if not (root/'scene_manifest.json').is_file():
        manifest = read(root/'dataset_manifest.json')
        if ids and list(ids) != [manifest.get('sequence_id')]:
            raise ValueError('selected sequence differs from dataset')
        return [root]
    index = collection(root)
    selected = list(ids) if ids is not None else list(index['sequences'])
    if not selected or len(selected) != len(set(selected)):
        raise ValueError('select one or more distinct sequences')
    unknown = set(selected)-set(index['sequences'])
    if unknown:
        raise ValueError('unknown sequences: '+', '.join(sorted(unknown)))
    return [inside(root, index['sequences'][sid]['manifest']).parent for sid in selected]


class SceneVersion:
    def __init__(self, root, revision, *, expected_hash):
        self.root = dataset_root(root)
        index = collection(self.root)
        self.revision = identity(revision)
        entry = index['versions'].get(revision)
        if entry is None:
            raise ValueError('unknown scene revision: '+revision)
        path = inside(self.root, entry['manifest'])
        check_file(path, expected_hash)
        self.manifest_hash = expected_hash
        self.directory, self.manifest = path.parent, read(path)
        m = self.manifest
        if (m.get('schema_version') != 1 or m.get('format') != 'cl_scene_version_v1'
                or m.get('scene_id') != index['scene_id'] or m.get('revision') != revision):
            raise ValueError('scene revision identity/version mismatch')
        inventory(self.directory, m.get('files'))
        for key, relative in m['artifacts'].items():
            if relative not in m['files']:
                raise ValueError('shared artifact missing from inventory: '+key)
            inside(self.directory, relative)
        asset = m.get('scene_asset')
        if asset:
            check_file(self.asset_path(), asset['sha256'])

    def path(self, key):
        return inside(self.directory, self.manifest['artifacts'][key])

    def asset_path(self):
        asset = self.manifest.get('scene_asset')
        if not asset:
            return None
        if asset.get('kind') != 'embedded':
            raise ValueError('published scene asset must be embedded with a relative path')
        return inside(self.directory, asset['path'])

    def reference(self):
        return {'revision': self.revision, 'manifest_sha256': self.manifest_hash}


class SequenceData:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = read(self.root/'dataset_manifest.json')
        m = self.manifest
        if (m.get('schema_version') != 5 or m.get('dataset_schema') != 'cl_task_sequence_v1'
                or m.get('layout_version') not in (1, 2)
                or m.get('runtime_contract') not in [dict(name='contextual_requests', version=v) for v in (1, 2, 3)]):
            raise ValueError('unsupported frozen dataset layout/runtime contract')
        self.version = None
        if m['layout_version'] == 2:
            sid = identity(m.get('sequence_id'))
            if self.root.name != sid or self.root.parent.name != 'sequences':
                raise ValueError('shared sequence must be under collection/sequences/sequence_id')
            ref = m['scene_reference']
            self.version = SceneVersion(self.root.parent.parent, ref['revision'], expected_hash=ref['manifest_sha256'])
            if self.version.manifest['scene_id'] != m['scene_id']:
                raise ValueError('sequence and shared scene identities differ')
            index = collection(self.root.parent.parent)['sequences'].get(sid)
            if not index or index['revision'] != ref['revision']:
                raise ValueError('sequence differs from collection revision')
            for key in ('scene_metainfo', 'scene_understanding', 'navigation', 'manipulation',
                        'public_language_profile', 'environment_setup'):
                if m.get(key) != {'scope': 'scene_version', 'key': key}:
                    raise ValueError('invalid shared scene reference: '+key)
            if m.get('scene_asset') != {'scope': 'scene_version'}:
                raise ValueError('invalid shared scene asset reference')

    def path(self, key):
        ref = self.manifest[key]
        if isinstance(ref, dict):
            if not self.version or ref != {'scope': 'scene_version', 'key': key}:
                raise ValueError('invalid scoped reference: '+key)
            return self.version.path(key)
        path = inside(self.root, ref)
        if ref not in self.manifest['files']:
            raise ValueError('artifact missing from inventory: '+key)
        return path

    def asset(self):
        if self.version:
            asset = self.version.manifest.get('scene_asset')
            return {'path': str(self.version.asset_path()), 'sha256': asset['sha256']} if asset else None
        asset = self.manifest.get('scene_asset')
        if asset:
            ref = asset['path']
            # Layout 1 historical archives may refer to a separately supplied asset.
            path = Path(ref).expanduser() if Path(ref).is_absolute() else inside(self.root, ref)
            check_file(path, asset['sha256'])
            return {**asset, 'path': str(path)}
        return None

    def load(self):
        m = self.manifest
        inventory(self.root, m.get('files'))
        self.asset()
        cache = read(self.path('runtime_scene_cache'))
        metadata = read(self.path('scene_metainfo'))
        if cache.get('scene_id') != m['scene_id'] or metadata['scene']['scene_id'] != m['scene_id']:
            raise ValueError('runtime cache/metadata scene identity mismatch')
        if set(cache['objects']) != {o['object_id'] for o in metadata['objects']}:
            raise ValueError('runtime cache object inventory mismatch')
        navigation = read_navigation_npz(self.path('navigation'))
        if navigation.get('environment_setup'):
            if 'environment_setup' not in m or read(self.path('environment_setup')) != navigation['environment_setup']:
                raise ValueError('environment setup differs from navigation')
        if m['runtime_contract']['version'] == 3:
            catalog = validate_catalog(read(self.path('language_catalog')))
        else:
            catalog = None
        plan = read(self.path('plan'))
        if m.get('sequence_id') != plan.get('sequence_id'):
            raise ValueError('sequence manifest and frozen plan identities differ')
        if self.version:
            if plan.get('scene_reference') != m['scene_reference']:
                raise ValueError('frozen plan scene revision mismatch')
            initial = read(self.version.path('initial_state'))
            if digest(initial) != self.version.manifest['initial_state_sha256']:
                raise ValueError('initial robot state checksum mismatch')
            if plan.get('names') != read(self.version.path('names')):
                raise ValueError('frozen names differ from scene revision')
        sessions = m.get('sessions')
        if not isinstance(sessions, list) or not sessions:
            raise ValueError('sequence has no sessions')
        ids = [s['session_id'] for s in sessions]
        if len(set(ids)) != len(ids) or ids != [s['session_id'] for s in plan['sessions']]:
            raise ValueError('manifest session order differs from frozen plan')
        for entry in sessions:
            identity(entry['session_id'])
            values = {}
            for key in ('session', 'ground_truth', 'templates', 'mutations'):
                ref = entry[key]
                if ref not in m['files']:
                    raise ValueError('session artifact missing from inventory: '+ref)
                values[key] = read(inside(self.root, ref))
            public, truths, templates = (values[k] for k in ('session', 'ground_truth', 'templates'))
            if public['scene_id'] != m['scene_id'] or public['session_id'] != entry['session_id']:
                raise ValueError('public session identity mismatch')
            tasks = public['tasks']
            tids = [t['task_id'] for t in tasks]
            if (len(set(tids)) != len(tids) or tids != [g['task_id'] for g in truths]
                    or tids != [t['task_id'] for t in templates]):
                raise ValueError('task/GT/template order mismatch')
            for task, truth, template in zip(tasks, truths, templates):
                if not truth.get('goals') or len(truth['goals']) != len(template['goals']):
                    raise ValueError('frozen goal/template count mismatch')
                if catalog is not None:
                    if template.get('schema_version') != 3 or render_request(template) != task['instruction']:
                        raise ValueError('public instruction differs from frozen template')
                    snapshots = [template['main'], *(intro['snapshot'] for intro in template['introductions'])]
                    entries = {c['template_id']: c for c in catalog['templates']}
                    for snapshot in snapshots:
                        approved = entries.get(snapshot['template_id'])
                        if not approved or snapshot['expression'] != approved['variants'].get(snapshot['variant_id']):
                            raise ValueError('template expression differs from frozen catalog')
                if any(oid not in cache['objects'] for goal in truth['goals'] for oid in goal['target']['object_ids']):
                    raise ValueError('ground truth references an unknown object')
            if not isinstance(values['mutations'], list):
                raise ValueError('mutations must be a frozen list')
        return self

    def mutation_context(self):
        """Minimal immutable geometry needed to validate real boundary changes."""
        metadata = read(self.path('scene_metainfo'))
        navigation = read_navigation_npz(self.path('navigation'))
        manipulation = read(self.path('manipulation'))
        pick_ids = {c['object_id'] for c in manipulation.get('pick_place_candidates', [])}
        return SimpleNamespace(metadata=metadata, engine=SimpleNamespace(
            rooms=deepcopy(metadata['rooms']), navigation=navigation, pick_ids=pick_ids))


class LegacyData:
    """Read archived schema 4 contracts without the generation/replay toolchain."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = read(self.root/'dataset_manifest.json')

    def path(self, key):
        defaults = {'runtime_scene_cache': 'private/scene/runtime_scene_cache.json',
                    'scene_metainfo': 'scene_metainfo.json'}
        return inside(self.root, self.manifest.get(key, defaults.get(key, key)))

    def session_path(self, entry, key):
        return inside(self.root, entry[key])

    def load_result(self):
        if self.manifest.get('files'):
            inventory(self.root, self.manifest['files'])
        for entry in self.manifest['sessions']:
            public = read(self.session_path(entry, 'session'))
            gt = read(self.session_path(entry, 'ground_truth'))
            templates = read(self.session_path(entry, 'task_templates'))
            if ([t['task_id'] for t in public['tasks']] != [t['task_id'] for t in gt['tasks']]
                    or [t['task_id'] for t in public['tasks']] != [t['task_id'] for t in templates['tasks']]):
                raise ValueError('legacy task/GT/template identity mismatch')
