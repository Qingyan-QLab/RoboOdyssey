"""Resolve published session.tar files into a separate, content-addressed cache."""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

from ..artifacts.layout import sha256_file


def discover_root(folder: Path) -> Path:
    """Accept a collection, a sequence, or the published wrapper directory."""
    for candidate in (folder, folder/'session/preparation/collection', folder/'preparation/collection'):
        if any((candidate/name).is_file() for name in ('scene_manifest.json', 'dataset_manifest.json', 'scene_version.json')):
            return candidate
    raise ValueError(f'no scene_manifest.json or dataset_manifest.json under {folder}')


def dataset_root(source: str | Path) -> Path:
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        return discover_root(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not tarfile.is_tarfile(source):
        raise ValueError(f'dataset must be a directory or a tar archive: {source}')
    digest = sha256_file(source)
    cache_base = Path(os.environ.get('BENCHMARK_DATA_CACHE',
        str(Path(os.environ.get('XDG_CACHE_HOME', str(Path.home()/'.cache')))/'longterm-indoor-benchmark/datasets'))).expanduser().resolve()
    cache_base.mkdir(parents=True, exist_ok=True)
    target = cache_base/digest
    if target.exists():
        return discover_root(target)
    staging = Path(tempfile.mkdtemp(prefix=f'.{digest[:12]}-', dir=cache_base))
    try:
        seen = set()
        with tarfile.open(source, 'r:*') as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts or '\\' in member.name:
                    raise ValueError(f'archive path escapes dataset: {member.name}')
                if not member.isdir() and not member.isfile():
                    raise ValueError(f'archive links/special files are unsupported: {member.name}')
                destination = staging.joinpath(*name.parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if name.as_posix() in seen:
                    raise ValueError(f'duplicate archive member: {member.name}')
                seen.add(name.as_posix())
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as stream, destination.open('xb') as output:
                    shutil.copyfileobj(stream, output)
        discover_root(staging)
        (staging/'.benchmark_archive.json').write_text(json.dumps(
            {'schema_version': 1, 'sha256': digest, 'source_name': source.name})+'\n')
        try:
            staging.rename(target)
        except OSError:
            if not target.is_dir():
                raise
        return discover_root(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def resource_root(source: str | Path, manifest_name: str) -> Path:
    """Resolve a standalone Scene/Session directory or safe tar into the cache."""
    if manifest_name not in {"scene_resources.json", "session_resources.json"}:
        raise ValueError("unsupported resource manifest")
    source = Path(source).expanduser().resolve()

    def locate(folder: Path) -> Path:
        candidates = [folder, *(p for p in folder.iterdir() if p.is_dir())]
        matches = [p for p in candidates if (p / manifest_name).is_file()]
        if len(matches) != 1:
            raise ValueError(f"expected one {manifest_name} under {folder}")
        return matches[0]

    if source.is_dir():
        return locate(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not tarfile.is_tarfile(source):
        raise ValueError(f"resource source must be a directory or tar archive: {source}")
    digest = sha256_file(source)
    cache_base = Path(os.environ.get("BENCHMARK_DATA_CACHE",
        str(Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) /
            "longterm-indoor-benchmark/datasets"))).expanduser().resolve() / "resources"
    cache_base.mkdir(parents=True, exist_ok=True)
    target = cache_base / digest
    if target.exists():
        return locate(target)
    staging = Path(tempfile.mkdtemp(prefix=f".{digest[:12]}-", dir=cache_base))
    try:
        seen = set()
        with tarfile.open(source, "r:*") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts or "\\" in member.name:
                    raise ValueError(f"archive path escapes resource root: {member.name}")
                if not member.isdir() and not member.isfile():
                    raise ValueError(f"archive links/special files are unsupported: {member.name}")
                destination = staging.joinpath(*name.parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if name.as_posix() in seen:
                    raise ValueError(f"duplicate archive member: {member.name}")
                seen.add(name.as_posix())
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as stream, destination.open("xb") as output:
                    shutil.copyfileobj(stream, output)
        locate(staging)
        try:
            staging.rename(target)
        except OSError:
            if not target.is_dir():
                raise
        return locate(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def scene_source_format(source: str | Path) -> str:
    """Identify one scene contract without falling back after a validation error.

    Match the same directory layouts as the two explicit loaders. Mixed inputs
    are ambiguous and require the caller to select a loader explicitly.
    """
    source = Path(source).expanduser().resolve()
    frozen_names = {"scene_manifest.json", "dataset_manifest.json", "scene_version.json"}
    frozen_parents = {(), ("session", "preparation", "collection"), ("preparation", "collection")}
    resource = frozen = False
    if source.is_dir():
        resource_roots = [source, *(path for path in source.iterdir() if path.is_dir())]
        resource = any((path / "scene_resources.json").is_file() for path in resource_roots)
        frozen = any((source.joinpath(*parent) / name).is_file()
                     for parent in frozen_parents for name in frozen_names)
    elif source.is_file():
        if not tarfile.is_tarfile(source):
            raise ValueError(f"scene source must be a directory or tar archive: {source}")
        with tarfile.open(source, "r:*") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts or "\\" in member.name:
                    raise ValueError(f"archive path escapes scene source: {member.name}")
                if not member.isfile():
                    continue
                if name.name == "scene_resources.json" and len(name.parts) in {1, 2}:
                    resource = True
                if name.name in frozen_names and name.parts[:-1] in frozen_parents:
                    frozen = True
    else:
        raise FileNotFoundError(source)
    if resource and frozen:
        raise ValueError("ambiguous scene source: use load_scene_spec or load_scene_version explicitly")
    if not resource and not frozen:
        raise ValueError(f"no scene resource or published scene manifest under {source}")
    return "resource" if resource else "frozen"
