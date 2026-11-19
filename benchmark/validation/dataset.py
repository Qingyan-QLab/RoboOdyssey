"""Validate published frozen scene collections and task sequences."""
from __future__ import annotations

import argparse
from pathlib import Path
from ..data.archive import dataset_root
from ..data.frozen import SequenceData, select_sequences


def validate(root: Path) -> list[str]:
    try:
        root = dataset_root(root)
        paths = select_sequences(root)
        errors = []
        for path in paths:
            try:
                SequenceData(path).load()
            except Exception as exc:
                errors.append(f'{path.name}: {exc}')
        return errors
    except Exception as exc:
        return [str(exc)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    errors = validate(args.dataset.resolve())
    if errors:
        print("Dataset validation failed:\n" + "\n".join(f"- {error}" for error in errors))
        return 1
    print(f"Dataset validation passed: {args.dataset.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
