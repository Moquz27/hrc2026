#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

isaac_python="${ISAAC_SIM_PYTHON:-}"
if [[ -z "${isaac_python}" && -n "${ISAAC_SIM_ROOT:-}" ]]; then
  isaac_python="${ISAAC_SIM_ROOT%/}/python.sh"
fi

if [[ -z "${isaac_python}" || ! -x "${isaac_python}" ]]; then
  cat >&2 <<'EOF'
Set ISAAC_SIM_PYTHON to Isaac Sim python.sh, or set ISAAC_SIM_ROOT.

Example:
  ISAAC_SIM_ROOT=/path/to/isaac-sim scripts/watch_task1_gui.sh
EOF
  exit 2
fi

seed="${SEED:-1}"
target_index="${TARGET_INDEX:-2}"
pregrasp_pullback_m="${PREGRASP_PULLBACK_M:-0.02}"
grasp_depth_offset="${GRASP_DEPTH_OFFSET:--0.020}"
log_suffix="${LOG_SUFFIX:-gui_watch_seed${seed}_target${target_index}_depth_watch}"

exec "${isaac_python}" \
  "${repo_root}/scripts/task1_smooth_autoseed_multi_object_baseline.py" \
  --seed "${seed}" \
  --target-index "${target_index}" \
  --pregrasp-pullback-m "${pregrasp_pullback_m}" \
  --grasp-depth-offset "${grasp_depth_offset}" \
  --gui \
  --hold-open \
  --log-suffix "${log_suffix}" \
  "$@"
