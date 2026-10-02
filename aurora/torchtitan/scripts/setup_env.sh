#!/bin/bash
# Load the software stack and activate the venv built by create_env.sh.
# Source this, do not execute it:  source scripts/setup_env.sh

if [ -n "${BASH_SOURCE[0]:-}" ]; then
    SCRIPT_PATH="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    SCRIPT_PATH="${(%):-%x}"
else
    SCRIPT_PATH="$0"
fi
_SETUP_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"

export BASE_DIR="${BASE_DIR:-$(cd -- "${_SETUP_DIR}/.." && pwd)}"
export REPOS_DIR="${REPOS_DIR:-${BASE_DIR}/venv_repos}"

case "${HOSTNAME}" in
    aurora*|x4*)       export SYSTEM=aurora  FSYS=flare ;;
    sunspot*|uan*|x1*) export SYSTEM=sunspot FSYS=tegu  ;;
    *)
        echo "setup_env.sh: unrecognized host '${HOSTNAME}'" >&2
        return 1 2>/dev/null || exit 1 ;;
esac

# Both systems resolve `frameworks` to the same stack
# (aurora_frameworks-2026.1.0: torch 2.13.0a0, vllm 0.26.1), so they share one
# line. Sunspot needed scripts/frameworks_sunspot.sh while its default was
# 2025.3.1, whose module had a broken dependency on intel_gpu_umd_aicoe; that
# is no longer the default. Set FRAMEWORKS_SCRIPT to go back to the conda path.
if [ -n "${FRAMEWORKS_SCRIPT:-}" ]; then
    source "${FRAMEWORKS_SCRIPT}"
else
    module load frameworks
fi

export PBS_QUEUE="${PBS_QUEUE:-$([ "$SYSTEM" = aurora ] && echo next-eval || echo workq)}"
export PBS_FILESYSTEMS="${PBS_FILESYSTEMS:-home:${FSYS}}"

export TORCHTITAN_DIR="${REPOS_DIR}/torchtitan"
export MONARCH_DIR="${REPOS_DIR}/monarch"
export TORCHSTORE_DIR="${REPOS_DIR}/torchstore"

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

source "${BASE_DIR}/venv/bin/activate"

# Sourced scripts must not leak this: callers have their own notion of it.
unset _SETUP_DIR SCRIPT_PATH
