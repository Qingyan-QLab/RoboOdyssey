#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_STEPS="${SMOKE_MAX_STEPS:-20}"
DATASET="${DATASET_DIR:-${ROOT}/dataset/scene000430_new}"

"${ROOT}/scripts/check_env.sh" --require-isaac --dataset "${DATASET}"

exec "${ROOT}/run_benchmark.sh" \
  --dataset "${DATASET}" \
  --agent-cmd "python3 ${ROOT}/examples/random_agent.py" \
  --max-steps "${MAX_STEPS}" \
  "$@"
