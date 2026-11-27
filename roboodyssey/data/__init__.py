"""Public readers for standalone resource bundles and frozen scene revisions."""
from .api import API_VERSION, CURRENT_FORMAT, load_scene as load_scene_version, load_sequence, validate_binding
from .archive import scene_source_format
from .frozen import SceneVersion, SequenceData
from .sources import (
    BenchmarkSuite, SceneSpec, SessionSpec,
    load_scene as load_scene_spec, load_session, load_published_dataset,
)


def load_scene(source, *, revision=None):
    """Load a SceneSpec resource bundle or a frozen SceneVersion.

    The manifest selects the contract; this does not convert between formats.
    Use load_scene_spec or load_scene_version when a fixed return type is needed.
    Revision selection applies only to the frozen scene-version contract.
    """
    if isinstance(source, SceneVersion):
        return load_scene_version(source, revision=revision)
    resource = isinstance(source, SceneSpec)
    if resource or scene_source_format(source) == "resource":
        if revision is not None:
            raise ValueError("revision is only supported for frozen SceneVersion inputs")
        return source if resource else load_scene_spec(source)
    return load_scene_version(source, revision=revision)


__all__ = [
    "API_VERSION", "CURRENT_FORMAT", "BenchmarkSuite", "SceneSpec", "SessionSpec",
    "SceneVersion", "SequenceData", "load_scene", "load_scene_spec", "load_scene_version",
    "load_session", "load_published_dataset", "load_sequence", "validate_binding",
]
