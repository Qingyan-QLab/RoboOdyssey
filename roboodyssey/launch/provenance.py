"""CPU provenance records: source hashes, local revisions, Python packages."""
from importlib.metadata import distributions
import platform
from pathlib import Path
from ..artifacts.layout import sha256_file as sha256
import subprocess
import sys

from ..artifacts.layout import write_json


ROOT = Path(__file__).resolve().parents[2]


def repository(root):
    root = Path(root)
    def git(*args):
        result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, timeout=15)
        return result.stdout.strip() if result.returncode == 0 else None
    files = {}
    for folder in ('roboodyssey', 'scripts'):
        for path in sorted((root/folder).rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.sh', '.json', '.toml'} and '__pycache__' not in path.parts and '.local.' not in path.name:
                files[path.relative_to(root).as_posix()] = sha256(path)
    return {'root': str(root), 'revision': git('rev-parse', 'HEAD'),
            'status': git('status', '--porcelain'), 'source_sha256': files}


def capture_provenance(path, *, dataset_root=None, scene_source=None, session_sources=(),
                       model_root=None, checkpoint=None, python_executables=(), source_roots=()):
    if dataset_root is not None and scene_source is not None:
        raise ValueError('provenance requires either frozen dataset or standalone scene inputs')
    if session_sources and scene_source is None:
        raise ValueError('standalone Session provenance requires scene_source')
    value = {'schema_version': 1, 'python': sys.executable, 'python_version': sys.version,
             'platform': platform.platform(), 'packages': {d.metadata['Name']: d.version for d in distributions() if d.metadata['Name']},
             'repositories': [repository(ROOT)], 'external_environments': []}
    if dataset_root:
        from ..data.archive import dataset_root as resolve_dataset
        source = Path(dataset_root).expanduser().resolve()
        root = resolve_dataset(source)
        value['dataset'] = {'root': str(root), 'manifests': {
            p.name: sha256(p) for p in (root/'dataset_manifest.json', root/'scene_manifest.json') if p.is_file()}}
        if source.is_file():
            value['dataset']['archive'] = {'path': str(source), 'sha256': sha256(source)}
    if scene_source is not None:
        from ..data.sources import BenchmarkSuite, load_scene, load_session

        scene = load_scene(scene_source)
        sessions = [load_session(source) for source in session_sources]
        if sessions:
            BenchmarkSuite(scene, sessions)  # Preserve identity/order checks across inputs.

        def resource_record(spec, source):
            record = {'root': str(spec.manifest_path.parent),
                      'manifest': {'name': spec.manifest_path.name, 'sha256': sha256(spec.manifest_path)},
                      'resources': {name: sha256(file) for name, file in spec.resources.items()}}
            source = Path(source).expanduser().resolve()
            if source.is_file():
                record['archive'] = {'path': str(source), 'sha256': sha256(source)}
            return record

        scene_record = resource_record(scene, scene_source)
        scene_record.update(scene_id=scene.scene_id, asset_sha256=sha256(scene.asset))
        session_records = []
        for session, source in zip(sessions, session_sources):
            record = resource_record(session, source)
            record.update(scene_id=session.scene_id, sequence_id=session.sequence_id,
                          session_id=session.session_id, session_sha256=sha256(session.path))
            session_records.append(record)
        value['dataset'] = {'kind': 'external_scene_session', 'scene': scene_record,
                            'sessions': session_records}
    if model_root:
        value['repositories'].append(repository(model_root))
    for source_root in source_roots:
        value['repositories'].append(repository(source_root))
    if checkpoint:
        checkpoint = Path(checkpoint)
        files = [checkpoint] if checkpoint.is_file() else sorted(p for p in checkpoint.rglob('*') if p.is_file())
        value['checkpoint'] = {'path': str(checkpoint), 'files': {str(p): sha256(p) for p in files}}
        base = checkpoint.parent if checkpoint.is_file() else checkpoint
        value['action_statistics'] = {str(p): sha256(p) for p in (base/'dataset_statistics.json', base.parent/'dataset_statistics.json') if p.is_file()}
    for executable in dict.fromkeys(python_executables):
        # Host/model Python only; never launch an Isaac executable here.
        result = subprocess.run([str(executable), '-c',
            "import sys,json,importlib.metadata as m; print(json.dumps({'version':sys.version,'packages':{d.metadata['Name']:d.version for d in m.distributions() if d.metadata['Name']}}))"],
            capture_output=True, text=True, timeout=30)
        value['external_environments'].append({'python': str(executable), 'returncode': result.returncode,
                                              'metadata': result.stdout, 'error': result.stderr})
    write_json(path, value)
    return value
