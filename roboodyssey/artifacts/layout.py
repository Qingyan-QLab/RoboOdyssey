"""Current result paths, evidence references and shared JSON readers."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


TRACE_FILES = {'runtime_tasks.jsonl', 'runtime_history.jsonl', 'agent_io.jsonl',
               'action_changelog.jsonl', 'runtime_events.jsonl', 'runtime_mutations.jsonl',
               'request_outcomes.jsonl', 'integration_errors.jsonl'}
VIDEO_FILES = {'review.mp4', 'review.partial.mp4', 'video_index.jsonl', 'video_manifest.json', 'video_encoder.log'}


def sha256_file(path):
    """Hash file bytes in bounded memory; shared by validation and provenance."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value, *, allow_nan=True):
    """Hash canonical JSON; callers can require finite numbers explicitly."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=allow_nan).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def read_json_lines(path, *, errors='raise', missing_ok=False):
    """Yield numbered JSON objects, raising or retaining corrupt audit rows."""
    if errors not in {'raise', 'record'}:
        raise ValueError('JSONL errors must be raise or record')
    path = Path(path)
    if missing_ok and not path.exists():
        return
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    if errors == 'raise':
                        raise ValueError(f'{path}:{number}: expected an object')
                    raise ValueError('expected a JSON object')
            except ValueError as exc:
                if errors == 'raise':
                    raise
                record = {'log_error': str(exc), 'file': path.name, 'line': number}
            yield number, record


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def local_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('artifact path escapes run: ' + str(relative))
    return path


def safe_id(value):
    if not re.fullmatch(r'[\w.-]+', value) or value in {'.', '..'}:
        raise ValueError('unsafe result identity: ' + value)
    return value


def manifest(root):
    path = Path(root) / 'manifest.json'
    if not path.exists():
        return None
    value = read_json(path)
    if value.get('artifact_schema') != 'benchmark_results' or value.get('schema_version') != 2:
        raise ValueError('unsupported benchmark artifact manifest')
    return value


def initialize_run(root, dataset):
    root = Path(root)
    identity = dataset.sequence_identity()
    parent = manifest(root.parent.parent)
    inline = parent and parent.get('kind') in {'experiment', 'collection'}
    scene = identity['scene_id']
    sequence = safe_id(identity['sequence_id'])
    entries = [{'scene_id': scene, 'sequence_id': sequence,
                'sessions_root': 'sessions' if inline else f'sequences/{sequence}/sessions',
                'session_ids': [s.session_id for s in dataset.sessions(scene)]}]
    write_json(root/'manifest.json', {'artifact_schema': 'benchmark_results', 'schema_version': 2,
        'kind': 'run', 'created_at': datetime.now(timezone.utc).isoformat(),
        'sequence_identity': identity, 'dataset_schema': dataset.manifest['schema_version'],
        'runtime_contract': dataset.manifest.get('runtime_contract'), 'sequences': entries,
        'files': {'config': 'run_config.json', 'launch_config': 'launch_config.json', 'provenance': 'provenance.json', 'status': 'status.json', 'summary': 'summary.json',
                  'run_log': 'logs/run.log', 'reports': 'reports'},
        'trace_schema': 2})
    (root/'logs').mkdir(exist_ok=True)
    update_status(root, phase='initializing', execution='pending', summary='pending', cleanup='pending')


def update_status(root, **values):
    path = Path(root)/'status.json'
    state = read_json(path) if path.exists() else {'schema_version': 1}
    state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
    write_json(path, state)
    return state


def session_dir(root, scene, session):
    root = Path(root)
    data = manifest(root)
    if data and data.get('kind') == 'run':
        entry = next(e for e in data['sequences'] if e['scene_id'] == scene)
        return local_path(root, entry['sessions_root']) / safe_id(session)
    raise ValueError('session paths require a benchmark run manifest')


def iter_sessions(root):
    root = Path(root)
    data = manifest(root)
    if data:
        if data.get('kind') != 'run':
            raise ValueError('expected a single run; use the analysis dispatcher for collections')
        for entry in data['sequences']:
            for sid in entry['session_ids']:
                yield entry['scene_id'], sid, local_path(root, entry['sessions_root']) / safe_id(sid)
    else:
        raise ValueError('run is missing its benchmark artifact manifest')


def create_session(root, scene, sid):
    path = session_dir(root, scene, sid)
    path.mkdir(parents=True, exist_ok=False)
    for name in ('traces', 'video'):
        (path/name).mkdir(exist_ok=True)
    return path


def session_file(session, name):
    session = Path(session)
    if name in TRACE_FILES:
        return session/'traces'/name
    if name in VIDEO_FILES:
        return session/'video'/name
    return session/name


def video_dir(session):
    return session_file(session, 'review.mp4').parent


def reports_dir(root):
    return Path(root)/'reports'


def compact_outcome(value):
    result = deepcopy(value)
    for goal in result.get('goal_results', []):
        actions = goal.pop('actions', None)
        if actions is not None:
            goal['action_evidence'] = {'scope': 'session', 'file': 'traces/action_changelog.jsonl', 'task_id': value['task_id'], 'count': len(actions)}
    return result


def compact_history(value):
    # Keep authoritative before/after snapshots. Only duplicated action arrays
    # are replaced by a reference; the live runtime history is untouched.
    result = deepcopy(value)
    actions = result.pop('actions', None)
    if actions is not None:
        result['action_evidence'] = {'scope': 'session', 'file': 'traces/action_changelog.jsonl', 'task_id': value['task_id'], 'count': len(actions)}
    return result
