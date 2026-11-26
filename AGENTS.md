# Repository Guidelines

## Structure and Dependency Boundaries

- `roboodyssey/` is the reusable core: Scene/Session loading, environment, interaction protocol, scoring references, evaluation, artifacts, and reports. It must not import `agent_integrations` or `synthesis`.
- Evaluated Agent integrations live in the separate `https://github.com/Qingyan-QLab/RoboOdyssey-Agent-Integrations` repository.
- Data synthesis and executable Oracle policies live in `https://github.com/Qingyan-QLab/RoboOdyssey-DataSynth`.
- `docs/SCENE_SESSION.md` defines standalone Scene/Session directory or tar inputs; `docs/DATA_FORMAT.md` defines existing published archives. `tests/` holds CPU-only regression cases.
- Runtime output belongs in ignored `results/`; external data, model sources, and weights are supplied separately.

## Development and Verification

Use Python 3.10+ and work from the repository root. Run these checks before submitting a change:

```bash
./scripts/bootstrap.sh
python3 -m roboodyssey check
python3 -m roboodyssey inventory --check
python3 -m roboodyssey run --dataset /path/to/session.tar --dry-run
```

The final dry-run needs a real external dataset. For a standalone bundle use `--scene-source <directory-or-tar> --session-source <directory-or-tar> --dry-run`. Isaac Sim execution uses its `python.sh` or `run_benchmark.sh`. Model environments are separate.

## Coding and Testing

Use four-space Python indentation, `snake_case` names, explicit JSON-serializable contracts, and Bash strict mode. Add focused `unittest` coverage for loaders, protocol, metric, and session changes without importing Isaac Sim or a model stack. Keep `docs/FILE_MAP.md` current through `python3 -m roboodyssey inventory`; add purposes for new files in `roboodyssey/validation/inventory.py`.

Preserve result schema, input hashes, public/private Agent boundaries, and sequence continuity when changing runtime code. Check CPU behavior first, then run an Isaac smoke case when that environment is available. Document any change to Scene/Session or Agent contracts in `docs/` and the relevant subdirectory README.

## Commits, Data, and Security

Use short imperative commit subjects. Pull requests should describe schema and behavior changes and list validation results. Do not commit generated results, caches, checkpoints, credentials, private conversation data, or unlicensed large scene assets. Preserve frozen dataset inputs and respect `NOTICE`.
