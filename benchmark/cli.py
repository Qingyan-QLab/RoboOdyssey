"""Unified command line for dataset setup, diagnostics, and evaluation."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen
from urllib.parse import urlparse

from .run import DEFAULT_DATASET
from .scenes.dataset_loader import BenchmarkDataset


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _doctor(dataset: Path, require_isaac: bool) -> int:
    dataset = dataset.expanduser().resolve()
    BenchmarkDataset(dataset)
    validator = ROOT / "scripts" / "validate_dataset.py"
    subprocess.run([sys.executable, str(validator), str(dataset)], check=True)
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
        unpacked = Path(directory) / "unpacked"
        unpacked.mkdir()
        shutil.unpack_archive(str(archive), str(unpacked))
        roots = [item for item in unpacked.iterdir()]
        source = roots[0] if len(roots) == 1 and roots[0].is_dir() else unpacked
        if not (source / "dataset_manifest.json").is_file():
            raise ValueError("downloaded archive does not contain dataset_manifest.json")
        shutil.copytree(source, destination)
    print(f"Dataset ready: {destination}")
    return _doctor(destination, require_isaac=False)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "run":
        from .run import main as run_main
        return run_main(raw[1:])
    parser = argparse.ArgumentParser(prog="python -m benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="validate data and runtime dependencies")
    doctor.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    doctor.add_argument("--require-isaac", action="store_true")

    download = subparsers.add_parser("download", help="download and verify a dataset archive")
    download.add_argument("--url", required=True)
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--sha256")

    subparsers.add_parser("run", help="run the benchmark")
    args = parser.parse_args(raw)
    if args.command == "doctor":
        return _doctor(args.dataset, args.require_isaac)
    if args.command == "download":
        return _download(args.url, args.output, args.sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
