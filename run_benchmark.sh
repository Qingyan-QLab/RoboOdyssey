#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HAS_SCENE_SOURCE=0
for arg in "$@"; do
  case "${arg}" in --scene-source|--scene-source=*) HAS_SCENE_SOURCE=1 ;; esac
done
if [[ -n "${DATASET_DIR:-}" && "${HAS_SCENE_SOURCE}" == 0 ]]; then
  set -- --dataset "${DATASET_DIR}" "$@"
fi
HAS_INPUT=0
for arg in "$@"; do
  case "${arg}" in --dataset|--dataset=*|--scene-source|--scene-source=*|-h|--help) HAS_INPUT=1 ;; esac
done
if [[ "${HAS_INPUT}" == 0 ]]; then
  echo "Provide --dataset PATH, DATASET_DIR, or --scene-source PATH with --session-source PATH." >&2
  exit 2
fi
if [[ -z "${ISAACSIM_ROOT:-}" ]]; then
  for candidate in "${BENCHMARK_ROOT}/../.." "${BENCHMARK_ROOT}/.."; do
    if [[ -x "${candidate}/python.sh" ]]; then
      ISAACSIM_ROOT="$(cd "${candidate}" && pwd)"
      break
    fi
  done
fi
ISAACSIM_ROOT="${ISAACSIM_ROOT:-}"
PYTHON_BIN="${ISAACSIM_PYTHON:-${ISAACSIM_ROOT}/python.sh}"

# Dataset inspection is intentionally usable on CPU-only hosts. Do not make
# users install Isaac Sim merely to validate a clone or inspect its sessions.
DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" || "$arg" == "--help" || "$arg" == "-h" ]]; then
    DRY_RUN=1
    break
  fi
done
if [[ "${DRY_RUN}" == 1 && -z "${ISAACSIM_PYTHON:-}" ]]; then
  if [[ -n "${BENCHMARK_PYTHON:-}" ]]; then
    PYTHON_BIN="${BENCHMARK_PYTHON}"
  elif [[ -x "${BENCHMARK_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${BENCHMARK_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ "${PYTHON_BIN}" == */* ]]; then
  PYTHON_AVAILABLE=1
  [[ -x "${PYTHON_BIN}" ]] || PYTHON_AVAILABLE=0
else
  PYTHON_AVAILABLE=1
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || PYTHON_AVAILABLE=0
fi
if [[ "${PYTHON_AVAILABLE}" != 1 ]]; then
  echo "Isaac Sim Python launcher not found: ${PYTHON_BIN}" >&2
  echo "Set ISAACSIM_ROOT to an Isaac Sim installation, or ISAACSIM_PYTHON to its python.sh." >&2
  exit 2
fi

# Isaac Sim ships its own ROS 2 Jazzy runtime.  The ROS bridge extension is
# loaded inside the Isaac Sim Python process, so these variables must be set
# before Kit starts; setting them only in the HoloAgent terminal is not enough.
if [[ "${BENCHMARK_ROS_BRIDGE:-0}" == 1 ]]; then
    export ROS_DISTRO="${ROS_DISTRO:-jazzy}"
    export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
    ROS2_CORE_LIB="${ISAACSIM_ROOT}/exts/isaacsim.ros2.core/${ROS_DISTRO}/lib"
    if [[ -d "${ROS2_CORE_LIB}" ]]; then
      export LD_LIBRARY_PATH="${ROS2_CORE_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
fi

cd "${BENCHMARK_ROOT}"
export PYTHONPATH="${ISAACSIM_ROOT}${BENCHMARK_PLUGIN_PATH:+:${BENCHMARK_PLUGIN_PATH}}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${AGENT_CMD:-}" ]]; then set -- --agent-cmd "$AGENT_CMD" "$@"; fi
exec "${PYTHON_BIN}" -m roboodyssey.run \
  --output "${OUTPUT_DIR:-${BENCHMARK_ROOT}/results}" \
  "${@}"
