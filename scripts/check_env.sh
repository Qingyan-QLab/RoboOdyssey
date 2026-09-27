#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET_DIR:-${ROOT}/dataset/scene000430_new}"
REQUIRE_ISAAC=0

usage() {
  cat <<'EOF'
Usage: scripts/check_env.sh [--dataset PATH] [--require-isaac]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --require-isaac) REQUIRE_ISAAC=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "${BENCHMARK_PYTHON:-}" ]]; then
  PYTHON_BIN="${BENCHMARK_PYTHON}"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi
if [[ "${PYTHON_BIN}" == */* ]]; then
  [[ -x "${PYTHON_BIN}" ]] || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 2; }
else
  command -v "${PYTHON_BIN}" >/dev/null || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 2; }
fi
[[ -d "${DATASET}" ]] || { echo "Dataset directory not found: ${DATASET}" >&2; exit 2; }

"${PYTHON_BIN}" "${ROOT}/scripts/validate_dataset.py" "${DATASET}"
"${PYTHON_BIN}" - <<'PY'
try:
    import numpy
except ImportError as exc:
    raise SystemExit("numpy is required; run scripts/bootstrap.sh") from exc
print(f"numpy {numpy.__version__}")
PY

if [[ "${REQUIRE_ISAAC}" == 1 ]]; then
  if [[ -n "${ISAACSIM_PYTHON:-}" ]]; then
    ISAAC_PY="${ISAACSIM_PYTHON}"
  elif [[ -n "${ISAACSIM_ROOT:-}" ]]; then
    ISAAC_PY="${ISAACSIM_ROOT}/python.sh"
  else
    echo "Set ISAACSIM_ROOT or ISAACSIM_PYTHON for an Isaac Sim run" >&2
    exit 2
  fi
  [[ -x "${ISAAC_PY}" ]] || { echo "Isaac Sim launcher not executable: ${ISAAC_PY}" >&2; exit 2; }
  "${ISAAC_PY}" -c 'import isaacsim; print("Isaac Sim Python is available")'
fi

echo "Environment check passed."
