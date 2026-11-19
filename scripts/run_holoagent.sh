#!/usr/bin/env bash
set -euo pipefail

# Do not mix packages from the caller's active conda environment with the
# dedicated VLM or Isaac Sim Python runtimes.
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Validate inputs on the host before starting any model services.
ISAACSIM_PYTHON="${BENCHMARK_PYTHON:-python3}" "${ROOT}/run_benchmark.sh" --dry-run "$@"
for arg in "$@"; do
  case "${arg}" in --dry-run|-h|--help) exit 0 ;; esac
done
HOLOAGENT_ROOT="${HOLOAGENT_ROOT:-${ROOT}/../../../HoloAgent}"
FSR_ROOT="${HOLOAGENT_ROOT}/agentic_robot/fsr_vln"
# CUDA_VISIBLE_DEVICES below exposes only the compatible GPU, so it is cuda:0
# inside the HoloAgent process.
MAPPING_DEVICE="${HOLOAGENT_MAPPING_DEVICE:-cuda:0}"
pick_free_port() {
  /usr/bin/python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}
SERVER_PORT="${HOLOAGENT_SERVER_PORT:-$(pick_free_port)}"
VLM_PORT="${HOLOAGENT_VLM_PORT:-$(pick_free_port)}"
CONDA_ENV="${HOLOAGENT_CONDA_ENV:-holoagent_semantic_mapping_py39}"
VLM_ENV="${HOLOAGENT_VLM_CONDA_ENV:-simplenav}"
VLM_MODEL="${HOLOAGENT_VLM_MODEL:-${ROOT}/../../../SimpleNav/local/models/Qwen3.5-4B}"
SERVER_LOG="${HOLOAGENT_SERVER_LOG:-${ROOT}/results/holoagent_server.log}"

if [[ -n "${HOLOAGENT_CONDA_BASE:-}" ]]; then
  CONDA_BASE="${HOLOAGENT_CONDA_BASE}"
elif [[ -n "${CONDA_EXE:-}" && "${CONDA_EXE}" == */bin/conda ]]; then
  CONDA_BASE="${CONDA_EXE%/bin/conda}"
elif [[ -d "${HOME}/miniconda3/envs" ]]; then
  CONDA_BASE="${HOME}/miniconda3"
elif [[ -d "${HOME}/anaconda3/envs" ]]; then
  CONDA_BASE="${HOME}/anaconda3"
else
  echo "Conda installation not found; set HOLOAGENT_CONDA_BASE" >&2
  exit 2
fi
HOLO_PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
VLM_PYTHON="${CONDA_BASE}/envs/${VLM_ENV}/bin/python"
[[ -x "${HOLO_PYTHON}" ]] || { echo "Conda environment not found: ${CONDA_ENV}" >&2; exit 2; }
[[ -x "${VLM_PYTHON}" ]] || { echo "VLM environment not found: ${VLM_ENV}" >&2; exit 2; }
[[ -f "${VLM_MODEL}/config.json" ]] || { echo "VLM model not found: ${VLM_MODEL}" >&2; exit 2; }
[[ -d "${HOLOAGENT_ROOT}" ]] || { echo "HoloAgent not found: ${HOLOAGENT_ROOT}" >&2; exit 2; }
if [[ -n "${HOLOAGENT_MAPPING_GPU:-}" ]]; then
  MAPPING_GPU="${HOLOAGENT_MAPPING_GPU}"
else
  MAPPING_GPU="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader 2>/dev/null | awk -F, 'tolower($2) ~ /rtx 4090/ {gsub(/ /, "", $1); print $1; exit}')"
fi
[[ -n "${MAPPING_GPU}" ]] || {
  echo "Compatible RTX 4090 not found; set HOLOAGENT_MAPPING_GPU to its UUID" >&2
  exit 2
}
mkdir -p "$(dirname "${SERVER_LOG}")"
echo "HoloAgent OVO GPU=${MAPPING_GPU} (cuda:0), port=${SERVER_PORT}; VLM port=${VLM_PORT}"

"${VLM_PYTHON}" "${ROOT}/examples/holoagent_benchmark_server.py" \
  --port "${VLM_PORT}" --local-vlm-model "${VLM_MODEL}" \
  --local-vlm-device auto \
  >"${ROOT}/results/holoagent_vlm_server.log" 2>&1 &
VLM_PID=$!

HOLOAGENT_DATA_ROOT="${HOLOAGENT_ROOT}" \
CUDA_VISIBLE_DEVICES="${MAPPING_GPU}" \
AMENT_PREFIX_PATH="${CONDA_BASE}/envs/${CONDA_ENV}" \
LD_LIBRARY_PATH="${CONDA_BASE}/envs/${CONDA_ENV}/lib" \
RMW_IMPLEMENTATION="rmw_cyclonedds_cpp" \
"${HOLO_PYTHON}" \
  "${ROOT}/examples/holoagent_benchmark_server.py" \
  --port "${SERVER_PORT}" \
  --direct-ovo --mapping-root "${FSR_ROOT}" \
  --mapping-config configs/ovo_zed.yaml \
  --mapping-device "${MAPPING_DEVICE}" \
  --vlm-planner-url "http://127.0.0.1:${VLM_PORT}/benchmark/step" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  kill "${VLM_PID}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null && ! kill -0 "${VLM_PID}" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  kill -KILL "${SERVER_PID}" 2>/dev/null || true
  kill -KILL "${VLM_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  wait "${VLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

server_ready() {
  "${HOLO_PYTHON}" -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${SERVER_PORT}/health', timeout=1).read()" \
    >/dev/null 2>&1
}

vlm_ready() {
  "${VLM_PYTHON}" -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${VLM_PORT}/health', timeout=1).read()" \
    >/dev/null 2>&1
}

for _ in $(seq 1 180); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "HoloAgent server failed; see ${SERVER_LOG}" >&2
    tail -80 "${SERVER_LOG}" >&2
    exit 1
  fi
  if ! kill -0 "${VLM_PID}" 2>/dev/null; then
    echo "HoloAgent VLM server failed; see results/holoagent_vlm_server.log" >&2
    tail -80 "${ROOT}/results/holoagent_vlm_server.log" >&2
    exit 1
  fi
  if server_ready && vlm_ready; then
    break
  fi
  sleep 1
done
server_ready || {
  echo "Timed out waiting for HoloAgent server; see ${SERVER_LOG}" >&2
  exit 1
}
vlm_ready || { echo "Timed out waiting for HoloAgent VLM server" >&2; exit 1; }

export AGENT_CMD="python3 ${ROOT}/examples/holoagent_agent.py --endpoint http://127.0.0.1:${SERVER_PORT}/benchmark/step --timeout 300"
HAS_MAX_STEPS=0
for arg in "$@"; do
  [[ "${arg}" == "--max-steps" || "${arg}" == --max-steps=* ]] && HAS_MAX_STEPS=1
done
if [[ "${HAS_MAX_STEPS}" == 0 ]]; then
  set -- --max-steps "${HOLOAGENT_MAX_STEPS:-100}" "$@"
fi
"${ROOT}/run_benchmark.sh" "$@"
