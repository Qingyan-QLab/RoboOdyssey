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
