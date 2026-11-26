"""Unified command line for dataset setup, diagnostics, and evaluation."""
from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from .artifacts.layout import sha256_file as _sha256
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def _doctor(dataset: Path, require_isaac: bool) -> int:
    dataset = dataset.expanduser().resolve()
    from .validation.dataset import validate
    errors = validate(dataset)
    if errors:
        raise ValueError('\n'.join(errors))
    try:
        import numpy
    except ImportError as exc:
        raise SystemExit("numpy is required; run ./scripts/bootstrap.sh") from exc
    print(f"numpy {numpy.__version__}")
    if require_isaac:
        try:
            import isaacsim  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "Isaac Sim is unavailable in this Python runtime; run through its python.sh"
            ) from exc
        print("Isaac Sim Python is available")
    print("Environment check passed.")
    return 0


def _download(url: str, destination: Path, expected_sha256: str | None) -> int:
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="longterm_benchmark_") as directory:
        suffix = "".join(Path(urlparse(url).path).suffixes)
        archive = Path(directory) / f"dataset_archive{suffix}"
        with urlopen(url) as response, archive.open("wb") as stream:  # noqa: S310
            shutil.copyfileobj(response, stream)
        actual = _sha256(archive)
        if expected_sha256 and actual.lower() != expected_sha256.lower():
            raise ValueError(f"dataset SHA-256 mismatch: expected {expected_sha256}, got {actual}")
        from .data.archive import dataset_root
        source = dataset_root(archive)
        from .validation.dataset import validate
        errors = validate(source)
        if errors:
            raise ValueError('\n'.join(errors))
        shutil.copytree(source, destination)
    print(f"Dataset ready: {destination}")
    return _doctor(destination, require_isaac=False)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    commands = {
        'run': ('run', {}, 'run the benchmark'),
        'validate': ('validation.dataset', {}, 'CPU dataset/archive/collection validation'),
        'preflight': ('launch.simplenav', {'mode': 'preflight'}, 'CPU-only SimpleNav experiment checks'),
        'experiment': ('launch.simplenav', {'mode': 'experiment'}, 'prepare or explicitly execute an experiment'),
        'analyze': ('reporting.analyze', {}, 'CPU repair, score and reset reports'),
        'compare': ('reporting.compare', {}, 'compare explicitly paired runs with matching evaluation conditions'),
        'batch': ('launch.batch', {}, 'plan or run generic serial sequence attempts with bounded retries'),
        'inventory': ('validation.inventory', {}, 'generate/check the complete source file map'),
        'robots': ('scenes.robot_body', {}, 'list or verify cylinder and local robot bodies (CPU only)'),
    }
    if raw and raw[0] in commands:
        module, options, _ = commands[raw[0]]
        return import_module('.' + module, __package__).main(raw[1:], **options)
    parser = argparse.ArgumentParser(prog="python -m roboodyssey")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (_, _, help_text) in commands.items():
        subparsers.add_parser(name, help=help_text)
    doctor = subparsers.add_parser("doctor", help="validate data and runtime dependencies")
    doctor.add_argument("--dataset", type=Path, required=True)
    doctor.add_argument("--require-isaac", action="store_true")

    download = subparsers.add_parser("download", help="download and verify a dataset archive")
    download.add_argument("--url", required=True)
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--sha256")

    check = subparsers.add_parser('check', help='CPU regression suite; simulator/model imports forbidden')
    check.add_argument('--core', action='store_true', help='fast subset; use the default for complete integration checks')
    args = parser.parse_args(raw)
    if args.command == 'check':
        return subprocess.run([sys.executable, '-m', 'roboodyssey.validation.check', *(['--core'] if args.core else [])]).returncode
    if args.command == "doctor":
        return _doctor(args.dataset, args.require_isaac)
    if args.command == "download":
        return _download(args.url, args.output, args.sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
