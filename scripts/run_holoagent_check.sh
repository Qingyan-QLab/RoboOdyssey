#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
[[ $# -gt 0 ]] && shift

usage() {
  echo "Usage: $0 {unit|qa|pick|transfer} [benchmark options]" >&2
  exit 2
}

latest_run() {
  find "${ROOT}/results" -maxdepth 1 -type d -name '*_python3' -printf '%T@ %p\n' \
    | sort -nr | head -1 | cut -d' ' -f2-
}

require_gpu_headroom() {
  local minimum="${HOLOAGENT_CHECK_MIN_FREE_VRAM_MB:-14000}"
  local insufficient
  insufficient="$(nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits \
    | awk -F, -v minimum="${minimum}" '{gsub(/ /,"",$3); if (($3 + 0) < (minimum + 0)) print $1 ":" $2 " free=" $3 "MB"}')"
  if [[ -n "${insufficient}" ]]; then
    echo "Insufficient free GPU memory for Isaac Sim + OVO + VLM (need >=${minimum} MB per GPU):" >&2
    echo "${insufficient}" >&2
    echo "Finish or stop other GPU workloads, then retry. Override only if intentional with HOLOAGENT_CHECK_MIN_FREE_VRAM_MB." >&2
    exit 3
  fi
}

require_metrics() {
  local path="$1"
  [[ -f "${path}" ]] || {
    echo "Diagnostic run did not produce metrics: ${path}" >&2
    echo "Inspect the newest Isaac Sim Kit log and results run.log." >&2
    exit 1
  }
}

case "${MODE}" in
  unit)
    cd "${ROOT}"
    exec env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE -u LD_LIBRARY_PATH \
      PYTHONNOUSERSITE=1 ../../python.sh -m unittest discover -s tests -v
    ;;
  qa)
    require_gpu_headroom
    "${ROOT}/scripts/run_holoagent.sh" --session session_001 --task qa_003 --max-steps 1 "$@"
    RUN="$(latest_run)"
    require_metrics "${RUN}/scenes/scene000430/sessions/session_001/metrics.json"
    jq '.task_results[] | select(.task_id=="qa_003")' \
      "${RUN}/scenes/scene000430/sessions/session_001/metrics.json"
    ;;
  pick)
    require_gpu_headroom
    "${ROOT}/scripts/run_holoagent.sh" --session session_000 --task manipulation_004 --max-steps 1 "$@"
    RUN="$(latest_run)"
    require_metrics "${RUN}/scenes/scene000430/sessions/session_000/metrics.json"
    jq -s 'map(select(.task_id=="manipulation_004")) | .[0] |
      {action, result, action_source}' \
      "${RUN}/scenes/scene000430/sessions/session_000/action_changelog.jsonl"
    echo "Expected diagnostic result: action=pick and result.success=true (task success remains false until place)."
    ;;
  transfer)
    require_gpu_headroom
    "${ROOT}/scripts/run_holoagent.sh" --session session_000 --task manipulation_004 --max-steps 120 "$@"
    RUN="$(latest_run)"
    require_metrics "${RUN}/scenes/scene000430/sessions/session_000/metrics.json"
    jq '.task_results[] | select(.task_id=="manipulation_004")' \
      "${RUN}/scenes/scene000430/sessions/session_000/metrics.json"
    ;;
  *) usage ;;
esac
