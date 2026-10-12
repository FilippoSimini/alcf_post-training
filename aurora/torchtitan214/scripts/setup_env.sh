#!/bin/bash
# Per-shell: load the oneAPI 2026.1 toolchain, activate the conda environment
# built by create_env.sh, and set the site paths the upstream launchers do not
# default correctly for this system.
#
#   source scripts/setup_env.sh
#
# Safe to source more than once; the launchers source the toolchain again.

_SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"

export BASE_DIR="${BASE_DIR:-$(cd -- "${_SETUP_DIR}/.." && pwd)}"
export REPOS_DIR="${REPOS_DIR:-${BASE_DIR}/venv_repos}"
export TORCHTITAN_DIR="${TORCHTITAN_DIR:-${REPOS_DIR}/torchtitan}"

# Held in plain, non-CONDA-named shell variables across the toolchain setup
# below, which runs `module reset`. A conda module's unload hook does
# `unset $(env | grep -o "[^=]*CONDA[^=]*")`, so anything exported under a name
# containing CONDA does not survive it -- including a caller's own override.
_env_prefix="${CONDA_ENV:-${BASE_DIR}/conda_env}"
_conda_base="${CONDA_PREFIX_BASE:-}"

case "${HOSTNAME}" in
    aurora*|x4*)       SYSTEM=aurora  FSYS=flare ;;
    sunspot*|uan*|x1*) SYSTEM=sunspot FSYS=tegu  ;;
    *) echo "setup_env.sh: unrecognized host '${HOSTNAME}'" >&2 ;;
esac
export SYSTEM FSYS
export PBS_QUEUE="${PBS_QUEUE:-$([ "$SYSTEM" = aurora ] && echo debug || echo workq)}"
export PBS_FILESYSTEMS="${PBS_FILESYSTEMS:-home:${FSYS}}"

# The oneAPI 2026.1 setup lives in the torchtitan checkout and is sourced from
# there, not copied here: every launcher sources the same file, so the toolchain
# cannot drift from the branch validated against it.
export ONEAPI_ENV_SCRIPT="${ONEAPI_ENV_SCRIPT:-${TORCHTITAN_DIR}/scripts/rl/xpu/env_torch214.sh}"
if [ -r "${ONEAPI_ENV_SCRIPT}" ]; then
    source "${ONEAPI_ENV_SCRIPT}"
else
    echo "setup_env.sh: ${ONEAPI_ENV_SCRIPT} not found; run scripts/create_env.sh first" >&2
fi

# Not `command -v conda`: a conda module activates by sourcing conda.sh, which
# defines `conda` as a shell FUNCTION, so `command -v` answers with the bare
# name and the path arithmetic resolves to the wrong directory.
if [ -z "$_conda_base" ]; then
    module load miniforge3 2>/dev/null || true
    _conda_exe="${CONDA_EXE:-$(type -P conda || true)}"
    [ -x "$_conda_exe" ] \
        && _conda_base="$(cd -- "$(dirname -- "$(dirname -- "$_conda_exe")")" && pwd)"
fi
export CONDA_PREFIX_BASE="$_conda_base"
export CONDA_ENV="$_env_prefix"
if [ -n "${CONDA_PREFIX_BASE}" ] && [ -d "${CONDA_ENV}" ]; then
    eval "$("${CONDA_PREFIX_BASE}/bin/conda" shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

# --- site paths the upstream launchers do not default for this system --------
#
# Where models and datasets live. HF_ASSETS_PATH points at the model directory
# itself, not its parent.
#
# /lus/<fs>/projects holds one directory per allocation, so there is no default
# that is right for everyone. Unset, this falls back to $HOME and says so,
# rather than silently writing a 1.5 GB model into a path chosen for someone
# else -- or failing on a directory the caller cannot write to.
if [ -z "${DATA_MODEL_PATH:-}" ]; then
    DATA_MODEL_PATH="${HOME}/huggingface"
    echo "setup_env.sh: DATA_MODEL_PATH unset, using ${DATA_MODEL_PATH}." \
         "Point it at a project filesystem to avoid the home quota, e.g." \
         "/lus/${FSYS}/projects/<project>/${USER}/huggingface" >&2
fi
export DATA_MODEL_PATH
export HF_MODEL_REPO="${HF_MODEL_REPO:-Qwen/Qwen3-0.6B}"
export HF_ASSETS_PATH="${HF_ASSETS_PATH:-${DATA_MODEL_PATH}/${HF_MODEL_REPO}}"
export HF_DATA_REPO="${HF_DATA_REPO:-kalomaze/alphabetic-arxiv-authors-it1}"

# HF_HOME is load-bearing and the launchers never set it. They export
# HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1 and then fall back to
# ~/.cache/huggingface, which is not where the dataset cache is on this system;
# every run then dies offline looking for a dataset that is already downloaded.
export HF_HOME="${HF_HOME:-${DATA_MODEL_PATH}/cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
[ -f "${HF_HOME}/token" ] && export HF_TOKEN="${HF_TOKEN:-$(cat "${HF_HOME}/token")}"

# Node-local, not $HOME: home is a shared Lustre filesystem here, and a single
# Inductor cache written concurrently by every node is both slow and a source
# of corrupt-cache failures.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/${USER}/torchinductor_xpu}"

# Monarch's HostMesh::attach() pushes config to every host and waits on a
# barrier, failing closed when it expires (monarch hyperactor_mesh/src/config.rs:
# MESH_ATTACH_CONFIG_TIMEOUT, default 60s). That default is too tight here: it
# has been measured failing 3 of 5 submissions on one day, always before any
# model work, as "attach failed: config push failed during attach on N host(s)
# ... reply timed out". Waiting longer costs nothing when the barrier completes
# normally, and a lost multi-node allocation is expensive.
#
# Parsed by humantime, so the unit is required: "300s", not "300".
export HYPERACTOR_MESH_ATTACH_CONFIG_TIMEOUT="${HYPERACTOR_MESH_ATTACH_CONFIG_TIMEOUT:-300s}"

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "torchtitan214: ${SYSTEM}, queue ${PBS_QUEUE}, env ${CONDA_ENV}"
