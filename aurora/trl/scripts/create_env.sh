#!/bin/bash
# One-time: build the venv for GRPO with TRL. Run once, then use setup_env.sh.
#   bash scripts/create_env.sh
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export BASE_DIR="${BASE_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export FRAMEWORKS_SCRIPT="${FRAMEWORKS_SCRIPT:-${SCRIPT_DIR}/frameworks_sunspot.sh}"

case "${HOSTNAME}" in
    aurora*|x4*)       module load frameworks ;;
    sunspot*|uan*|x1*) source "${FRAMEWORKS_SCRIPT}" ;;
    *) echo "create_env.sh: unrecognized host '${HOSTNAME}'" >&2; exit 1 ;;
esac

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-$HTTP_PROXY}"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"

echo "===== Creating venv at ${BASE_DIR}/venv"
python3 -m venv --system-site-packages "${BASE_DIR}/venv"
source "${BASE_DIR}/venv/bin/activate"

# The frameworks stack sets PYTHONUSERBASE, so a pip that escapes the venv
# installs into ~/.local instead of failing.
export PIP_REQUIRE_VIRTUALENV=1

# --no-deps throughout: torch, transformers, datasets and vllm come from the
# system stack and must not be replaced by PyPI (CUDA) builds.
pip install 'trl>=1.6.0' --no-deps
pip install peft --no-deps
pip install accelerate --no-deps
pip install wandb --no-deps

echo "===== Environment ready. Activate with: source ${SCRIPT_DIR}/setup_env.sh"
