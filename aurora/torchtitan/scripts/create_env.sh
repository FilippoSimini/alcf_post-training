#!/bin/bash
# One-time: build the venv for GRPO with torchtitan. Run once, then use
# setup_env.sh. Takes roughly 30-45 minutes, most of it compiling Monarch's
# Rust extension.
#   bash scripts/create_env.sh
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export BASE_DIR="${BASE_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export REPOS_DIR="${REPOS_DIR:-${BASE_DIR}/venv_repos}"

# XPU support for all three lives in forks. `rl` carries the RL work for
# torchstore and torchtitan; monarch has no `rl` branch, so it uses
# `xpu-upstream`.
#
# These are branches, not fixed commits, so what you get moves with the forks.
# The resolved SHAs are written to $REPOS_DIR/PINNED.txt at the end of this
# script -- quote that file when reporting results, and set the *_REF variables
# to those SHAs to rebuild the same thing later.
#
# (An earlier version pinned commits taken from a working install. Two of the
# three did not exist on the public forks -- they were local commits from a
# debugging campaign that was never pushed -- so the build failed with
# "fatal: reference is not a tree".)
TORCHSTORE_REPO="${TORCHSTORE_REPO:-https://github.com/songhappy/torchstore.git}"
TORCHSTORE_REF="${TORCHSTORE_REF:-rl}"
MONARCH_REPO="${MONARCH_REPO:-https://github.com/songhappy/monarch.git}"
MONARCH_REF="${MONARCH_REF:-xpu-upstream}"
TORCHTITAN_REPO="${TORCHTITAN_REPO:-https://github.com/songhappy/torchtitan.git}"
TORCHTITAN_REF="${TORCHTITAN_REF:-rl}"

case "${HOSTNAME}" in
    aurora*|x4*)       SYSTEM=aurora  ;;
    sunspot*|uan*|x1*) SYSTEM=sunspot ;;
    *) echo "create_env.sh: unrecognized host '${HOSTNAME}'" >&2; exit 1 ;;
esac

# One stack for both systems: `frameworks` resolves to aurora_frameworks-2026.1.0
# (torch 2.13.0a0, vllm 0.26.1) on Aurora and Sunspot alike, so a venv built
# here is comparable across them. scripts/frameworks_sunspot.sh remains for the
# old 2025.3.1 conda path; select it with FRAMEWORKS_SCRIPT.
if [ -n "${FRAMEWORKS_SCRIPT:-}" ]; then
    source "${FRAMEWORKS_SCRIPT}"
else
    module load frameworks
fi

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-$HTTP_PROXY}"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"

echo "===== Creating venv at ${BASE_DIR}/venv"
mkdir -p "${REPOS_DIR}"
python3 -m venv --system-site-packages "${BASE_DIR}/venv"
source "${BASE_DIR}/venv/bin/activate"

# The frameworks stack sets PYTHONUSERBASE, so a pip that escapes the venv
# installs into ~/.local instead of failing.
export PIP_REQUIRE_VIRTUALENV=1

clone_at() {  # repo ref dest
    [ -d "$3" ] || git clone "$1" "$3"
    git -C "$3" fetch --all --quiet
    # Branch names need the remote-tracking ref; a raw SHA works as-is.
    git -C "$3" checkout --quiet "origin/$2" 2>/dev/null \
        || git -C "$3" checkout --quiet "$2"
    echo "      $(basename "$3") $2 -> $(git -C "$3" rev-parse HEAD)"
}

echo "===== TorchStore"
clone_at "$TORCHSTORE_REPO" "$TORCHSTORE_REF" "${REPOS_DIR}/torchstore"
pip install -e "${REPOS_DIR}/torchstore" --no-deps --no-build-isolation
pip install pygtrie portpicker

echo "===== Monarch toolchain (Rust nightly + protoc)"
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y
fi
source "$HOME/.cargo/env"
export PROTOC="${PROTOC:-$HOME/.local/protoc/bin/protoc}"
if [ ! -x "$PROTOC" ]; then
    curl -sSLO https://github.com/protocolbuffers/protobuf/releases/download/v29.3/protoc-29.3-linux-x86_64.zip
    mkdir -p "$HOME/.local"
    unzip -q -o protoc-29.3-linux-x86_64.zip -d "$HOME/.local/protoc"
    rm -f protoc-29.3-linux-x86_64.zip
fi

echo "===== Monarch"
clone_at "$MONARCH_REPO" "$MONARCH_REF" "${MONARCH_DIR:-${REPOS_DIR}/monarch}"
cd "${REPOS_DIR}/monarch"
# The Mesh Admin TUI is a separate Rust binary that is not needed here and adds
# several minutes to the build.
sed -i '/Mesh Admin TUI/,/^)$/{s/^/#/}' setup.py
pip install setuptools-rust
# icpx emits an undefined _intel_fast_memcpy from aws-lc-sys; gcc does not.
CC=gcc CXX=g++ pip install -e . --no-deps --no-build-isolation
pip install pyzmq pyarrow requests numpy pyre-extensions "typing-extensions>=4.12" \
    cloudpickle lark tabulate opentelemetry-api clusterscope "flask>=2.0" \
    xxhash py-spy aiohttp

echo "===== TorchTitan"
clone_at "$TORCHTITAN_REPO" "$TORCHTITAN_REF" "${REPOS_DIR}/torchtitan"
pip install -e "${REPOS_DIR}/torchtitan" --no-deps --no-build-isolation
pip install tyro tensorboard wandb
# spmd_types declares torch>=2.10, so it must stay --no-deps or pip pulls a
# PyPI (CUDA) torch into the venv and shadows the XPU build.
pip install spmd_types==0.2.1 --no-deps
# renderers must NOT be --no-deps: torchtitan's alphabet_sort imports it, and it
# needs prime-pydantic-config (`import pydantic_config`), jinja2, tiktoken,
# openai and openai-harmony. None of those pull torch, so this is safe.
pip install renderers

# Record exactly what was built. Branch heads move; this file does not.
PINNED="${REPOS_DIR}/PINNED.txt"
{
    echo "# built $(date -u +%Y-%m-%dT%H:%M:%SZ) on ${HOSTNAME}"
    for r in torchstore monarch torchtitan; do
        printf '%-12s %-14s %s\n' "$r" \
            "$(git -C "${REPOS_DIR}/$r" rev-parse --abbrev-ref HEAD 2>/dev/null)" \
            "$(git -C "${REPOS_DIR}/$r" rev-parse HEAD 2>/dev/null)"
    done
} | tee "$PINNED"

echo "===== Environment ready. Activate with: source ${SCRIPT_DIR}/setup_env.sh"
echo "===== Exact revisions recorded in ${PINNED}"
