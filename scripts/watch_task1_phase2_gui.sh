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

Examples:
  ISAAC_SIM_ROOT=/path/to/isaac-sim scripts/watch_task1_phase2_gui.sh
  ISAAC_SIM_PYTHON=/path/to/isaac-sim/python.sh scripts/watch_task1_phase2_gui.sh
EOF
  exit 2
fi

seed="${SEED:-1}"
target_selection_policy="${TARGET_SELECTION_POLICY:-index}"
target_index="${TARGET_INDEX:-2}"
arm="${ARM:-auto}"
log_suffix="${LOG_SUFFIX:-phase2_gui_seed${seed}_target${target_index}_${arm}}"

extra_args=()
case "${CONTINUE_AFTER_LIFT:-0}" in
  1|true|TRUE|yes|YES)
    extra_args+=(--continue-after-lift)
    ;;
esac

case "${NO_HOLD_OPEN:-0}" in
  1|true|TRUE|yes|YES)
    ;;
  *)
    extra_args+=(--hold-open)
    ;;
esac

exec "${isaac_python}" \
  "${repo_root}/scripts/task1_hybrid_geometric_phase2.py" \
  --seed "${seed}" \
  --target-selection-policy "${target_selection_policy}" \
  --target-index "${target_index}" \
  --arm "${arm}" \
  --gui \
  --log-suffix "${log_suffix}" \
  "${extra_args[@]}" \
  "$@"
