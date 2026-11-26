"""Public CPU-only readers for the current shared scene/sequence contract."""
from pathlib import Path

from .archive import dataset_root
from .frozen import SceneVersion, SequenceData, collection, inside, read, select_sequences

CURRENT_FORMAT = {'schema_version': 5, 'layout_version': 2,
                  'runtime_contract': {'name': 'contextual_requests', 'version': 3}}
API_VERSION = 1


def load_scene(source, *, revision=None):
    """Load a collection, sequence, or independent scene revision directory.

    A scene-only collection may have no sequences. Assets must be embedded;
    published data never resolves private machine paths.
    """
    if isinstance(source, SceneVersion):
        scene = source
    else:
        root = dataset_root(source)
        if (root/'scene_version.json').is_file():
            scene = SceneVersion.from_directory(root)
        elif (root/'dataset_manifest.json').is_file():
            scene = load_sequence(root).version
        else:
            index = collection(root, require_sequences=False)
            selected = revision or index['default_revision']
            if selected not in index['versions']:
                raise ValueError('unknown scene revision: '+str(selected))
            scene = SceneVersion(root, selected, expected_hash=index['versions'][selected]['sha256'])
    if revision is not None and scene.revision != revision:
        raise ValueError('selected scene revision differs from input')
    return scene


def load_sequence(source, *, sequence_id=None, scene=None):
    """Read frozen data without invoking a generator or replaying planned success.

    For a detached sequence directory provide its separately loaded scene.
    A collection with several sequences requires an explicit sequence_id.
    """
    if isinstance(source, SequenceData):
        result = source.load()
    else:
        root = dataset_root(source)
        if (root/'scene_manifest.json').is_file():
            paths = select_sequences(root, [sequence_id] if sequence_id else None)
            if len(paths) != 1:
                raise ValueError('collection contains multiple sequences; select sequence_id')
            root = paths[0]
        result = SequenceData(root, scene=load_scene(scene) if scene is not None else None).load()
    if sequence_id is not None and result.manifest['sequence_id'] != sequence_id:
        raise ValueError('selected sequence_id differs from input')
    if scene is not None:
        validate_binding(load_scene(scene), result)
    return result


def validate_binding(scene, sequence):
    """Reject identity, revision or content mismatches; return the pinned identity."""
    scene = load_scene(scene)
    if not isinstance(sequence, SequenceData):
        sequence = load_sequence(sequence, scene=scene)
    manifest = sequence.manifest
    if scene.manifest['scene_id'] != manifest['scene_id'] or scene.reference() != manifest['scene_reference']:
        raise ValueError('scene/sequence binding mismatch')
    return {key: manifest[key] for key in ('scene_id', 'sequence_id', 'scene_reference')}
