#!/usr/bin/env python3
"""Generate deterministic SHA-256 checksums for a release dataset."""
from __future__ import annotations
import argparse, hashlib
from pathlib import Path

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.dataset.resolve()
    output = args.output or root / "checksums.sha256"
    files = [path for path in root.rglob("*") if path.is_file() and path.resolve() != output.resolve()]
    text = "".join(f"{digest(path)}  {path.relative_to(root).as_posix()}\n" for path in sorted(files))
    output.write_text(text, encoding="utf-8")
    print(f"Wrote {len(files)} checksums to {output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
