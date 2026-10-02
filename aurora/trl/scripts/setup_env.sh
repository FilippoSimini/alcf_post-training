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
case "${HOSTNAME}" in
    aurora*|x4*)       export SYSTEM=aurora  FSYS=flare ;;
    sunspot*|uan*|x1*) export SYSTEM=sunspot FSYS=tegu  ;;
    *)
        echo "setup_env.sh: unrecognized host '${HOSTNAME}'" >&2
        return 1 2>/dev/null || exit 1 ;;
esac

# Both systems resolve `frameworks` to aurora_frameworks-2026.1.0. Set
# FRAMEWORKS_SCRIPT to use scripts/frameworks_sunspot.sh (the 2025.3.1 conda
# path) instead.
if [ -n "${FRAMEWORKS_SCRIPT:-}" ]; then
    source "${FRAMEWORKS_SCRIPT}"
else
    module load frameworks
fi

export PBS_QUEUE="${PBS_QUEUE:-$([ "$SYSTEM" = aurora ] && echo next-eval || echo workq)}"
export PBS_FILESYSTEMS="${PBS_FILESYSTEMS:-home:${FSYS}}"

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

source "${BASE_DIR}/venv/bin/activate"

# Sourced scripts must not leak this: callers have their own notion of it.
unset _SETUP_DIR SCRIPT_PATH
