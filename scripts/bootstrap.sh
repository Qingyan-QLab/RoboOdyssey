#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${BENCHMARK_VENV:-${ROOT}/.venv}"
INSTALL=1

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap.sh [--skip-install]

Create a small host-Python environment for validation and dry-runs. Isaac Sim
is installed separately and is selected with ISAACSIM_ROOT or ISAACSIM_PYTHON.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --skip-install) INSTALL=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 2; }
if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
fi

if [[ "${INSTALL}" == 1 ]]; then
  "${VENV}/bin/python" -m pip install -r "${ROOT}/requirements.txt"
fi

echo "Host environment ready: ${VENV}/bin/python"
echo "For Isaac Sim runs, export ISAACSIM_ROOT=/path/to/isaacsim."
