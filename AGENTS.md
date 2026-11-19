# Repository Guidelines

## Project Structure & Module Organization

- `benchmark/` contains the evaluation runtime: Protocol v1 handling, dataset and scene loading, Isaac Sim integration, session execution, metrics, and reports.
- `examples/` contains agent adapters and the random smoke agent; keep external-agent integration here unless a protocol capability is shared by all agents.
- `generation/` is an optional Actor SDG entry point and is separate from benchmark execution.
- `scripts/` provides environment checks, dataset validation, checksums, smoke runs, and result analysis. `run_benchmark.sh` is the Isaac Sim launcher.
- `tests/` contains CPU-only unit tests, `docs/` documents protocols and metrics, and `dataset/` holds example scene packages. Runtime output belongs in ignored `results/` directories.

## Build, Test, and Development Commands

Set up a local environment with `./scripts/bootstrap.sh` and activate `.venv`. Run these checks before submitting changes:

```bash
./scripts/check_env.sh
python3 -m unittest discover -s tests -v
python3 -m benchmark.run --dry-run
```

For an Isaac Sim smoke run, set `ISAACSIM_ROOT` and run `./scripts/run_smoke.sh`; use `--viewer` only with a graphical environment. Validate datasets with `python3 scripts/validate_dataset.py <path>` and create checksums with `python3 scripts/make_checksums.py <path>`.

## Coding Style & Naming Conventions

Use Python 3.10+ with four-space indentation, `snake_case` for functions, variables, and modules, `PascalCase` for classes, and `UPPER_SNAKE_CASE` constants. Keep protocol payloads and dataset schemas explicit and JSON-serializable. No formatter or linter is configured, so review imports and changed lines manually. Shell scripts should remain Bash-compatible and use strict mode (`set -euo pipefail`).

## Testing Guidelines

Tests use `unittest` and should run without GPU or Isaac Sim. Add focused `test_*` methods in `tests/` for loader, protocol, metric, or session changes. For dataset or runner changes, also run a dry run; use the Isaac smoke test when available.

## Commit & Pull Request Guidelines

The history currently contains only an initial commit, so no established message format is available. Use short, imperative subjects (for example, `Validate session dependencies`) and keep unrelated changes separate. Pull requests should explain behavior and schema changes, link issues, list validation commands and results, and include screenshots or logs when viewer or output behavior changes. Do not commit checkpoints, caches, `results/`, or unlicensed large scene assets; update `docs/` and checksums when contracts change.

## Data and Environment Notes

Isaac Sim, ROS, external agent services, model checkpoints, and Actor SDG dependencies are installed separately. Preserve frozen dataset inputs, respect `NOTICE`, and publish large USDZ files through the documented release or LFS workflow.
