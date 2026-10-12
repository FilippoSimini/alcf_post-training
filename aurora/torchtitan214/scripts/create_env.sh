#!/bin/bash
# One-time: build the torch 2.14.0+xpu / oneCCL 2022 conda environment for GRPO
# with torchtitan. Run once on a login node, then use setup_env.sh.
#
#   bash scripts/create_env.sh
#
# Roughly 60-90 minutes, most of it compiling Monarch's Rust extension. Needs
# outbound network, so a login node (UAN), never a compute node.
#
# This environment owns its own torch: unlike the `frameworks` module stack, it
# installs torch, Triton and vLLM from wheels and source. Nothing here reads or
# writes any other environment.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export BASE_DIR="${BASE_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export REPOS_DIR="${REPOS_DIR:-${BASE_DIR}/venv_repos}"

# Held in plain, non-CONDA-named shell variables until after the toolchain
# setup below, which runs `module reset`. A conda module's unload hook does
# `unset $(env | grep -o "[^=]*CONDA[^=]*")`, so anything exported under a name
# containing CONDA does not survive that -- including a value the caller
# deliberately passed in.
_env_prefix="${CONDA_ENV:-${BASE_DIR}/conda_env}"
_conda_base="${CONDA_PREFIX_BASE:-}"

# Four repos. XPU support is in forks for three of them; vLLM is upstream `main`
# at a commit whose requirements/xpu.txt pins the matching torch/Triton set.
#
# These are fixed commits, not branch heads, because this stack is a
# reproduction of a validated reference environment -- a moving branch would
# make a failure ambiguous between "Sunspot differs" and "the fork moved". The
# resolved SHAs are written to $REPOS_DIR/PINNED.txt; quote that file alongside
# any result.
TORCHTITAN_REPO="${TORCHTITAN_REPO:-https://github.com/songhappy/torchtitan.git}"
TORCHTITAN_REF="${TORCHTITAN_REF:-xpu/rl-rebase-20261001}"
TORCHSTORE_REPO="${TORCHSTORE_REPO:-https://github.com/songhappy/torchstore.git}"
TORCHSTORE_REF="${TORCHSTORE_REF:-1ebced4}"
MONARCH_REPO="${MONARCH_REPO:-https://github.com/songhappy/monarch.git}"
MONARCH_REF="${MONARCH_REF:-7761764a}"
# Upstream commit that makes default_bootstrap_cmd public, which torchtitan
# imports. It is not on the fork's branch, so it is cherry-picked on top and
# may need fetching from upstream first.
MONARCH_UPSTREAM="${MONARCH_UPSTREAM:-https://github.com/pytorch-labs/monarch.git}"
MONARCH_CHERRY="${MONARCH_CHERRY:-d9fc48862}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_REF="${VLLM_REF:-083060d04}"

RUST_VERSION="${RUST_VERSION:-1.97.1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

case "${HOSTNAME}" in
    aurora*|x4*)       SYSTEM=aurora  ;;
    sunspot*|uan*|x1*) SYSTEM=sunspot ;;
    *) echo "create_env.sh: unrecognized host '${HOSTNAME}'" >&2; exit 1 ;;
esac
echo "===== Building on ${SYSTEM} (${HOSTNAME})"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-$HTTP_PROXY}"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"

# Resolve conda to an ABSOLUTE path before anything touches the module
# environment: the oneAPI setup below runs `module reset`, which would unload a
# miniforge3 module and take `conda` off PATH with it.
#
# Not `command -v conda`: a conda module activates by sourcing conda.sh, which
# defines `conda` as a shell FUNCTION, so `command -v` answers with the bare
# name and the path arithmetic below silently resolves to the wrong directory.
# $CONDA_EXE and `type -P` both give a real path.
if [ -z "$_conda_base" ]; then
    module load miniforge3 2>/dev/null || true
    _conda_exe="${CONDA_EXE:-$(type -P conda || true)}"
    [ -x "$_conda_exe" ] || {
        echo "ERROR: no conda found. Load a conda module or set CONDA_PREFIX_BASE." >&2
        exit 1
    }
    _conda_base="$(cd -- "$(dirname -- "$(dirname -- "$_conda_exe")")" && pwd)"
fi
echo "===== conda base: ${_conda_base}"

clone_at() {  # repo ref dest
    [ -d "$3" ] || git clone "$1" "$3"
    git -C "$3" fetch --all --quiet
    # Branch names need the remote-tracking ref; a raw SHA works as-is.
    git -C "$3" checkout --quiet --force "origin/$2" 2>/dev/null \
        || git -C "$3" checkout --quiet --force "$2"
    echo "      $(basename "$3") $2 -> $(git -C "$3" rev-parse HEAD)"
}

# ---------------------------------------------------------------- toolchain --
# TorchTitan first, before anything else needs a compiler: it carries
# env_torch214.sh, the oneAPI 2026.1 setup every launcher sources. Sourcing the
# checked-out copy rather than a copy kept here means the toolchain cannot drift
# from the branch that was validated against it.
echo "===== TorchTitan checkout (for env_torch214.sh)"
mkdir -p "${REPOS_DIR}"
clone_at "$TORCHTITAN_REPO" "$TORCHTITAN_REF" "${REPOS_DIR}/torchtitan"
export TORCHTITAN_DIR="${REPOS_DIR}/torchtitan"

echo "===== oneAPI 2026.1 toolchain"
# torch 2.14.0+xpu ships the oneAPI 2026.1 SYCL runtime. Never source a oneAPI
# 2025.x environment in the same shell: two SYCL runtimes in one process
# segfault in sycl::device::get_backend().
source "${TORCHTITAN_DIR}/scripts/rl/xpu/env_torch214.sh"

# Safe to carry CONDA-named variables again now that `module reset` is behind us.
export CONDA_PREFIX_BASE="$_conda_base"
export CONDA_ENV="$_env_prefix"

# ------------------------------------------------------------------- python --
echo "===== Conda environment at ${CONDA_ENV}"
eval "$("${CONDA_PREFIX_BASE}/bin/conda" shell.bash hook)"
if [ ! -d "${CONDA_ENV}" ]; then
    conda create -p "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${CONDA_ENV}"
# cargo for Monarch's Rust extension and for vLLM's Rust tool-parser frontend.
# Tested on the env prefix, not on PATH: a rustup toolchain in $HOME/.cargo from
# some other build would otherwise satisfy `command -v cargo` and silently
# replace the pinned compiler.
[ -x "${CONDA_ENV}/bin/cargo" ] || conda install -y -c conda-forge "rust=${RUST_VERSION}"

# ---------------------------------------------------------------- torch/vLLM --
echo "===== torch, Triton and the vLLM XPU kernels (from vLLM's own pins)"
clone_at "$VLLM_REPO" "$VLLM_REF" "${REPOS_DIR}/vllm"
# requirements/xpu.txt is the authority for the matching set (torch 2.14.0,
# triton 3.8.0+xpu, vllm_xpu_kernels 0.1.15.4). Pinning them here instead would
# be a second copy free to drift from the vLLM commit they belong to.
pip install -r "${REPOS_DIR}/vllm/requirements/xpu.txt" \
    --extra-index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://wheels.vllm.ai/xpu/
python -c "import torch; print('torch', torch.__version__)"

echo "===== vLLM"
cd "${REPOS_DIR}/vllm"
# VLLM_REQUIRE_RUST_FRONTEND=1 makes a missing Rust toolchain an error instead
# of silently dropping _rust_tool_parser.
VLLM_TARGET_DEVICE=xpu VLLM_REQUIRE_RUST_FRONTEND=1 \
    pip install -e . --no-deps --no-build-isolation -v

# ----------------------------------------------------------------- monarch --
echo "===== Monarch"
clone_at "$MONARCH_REPO" "$MONARCH_REF" "${REPOS_DIR}/monarch"
cd "${REPOS_DIR}/monarch"
# The cherry-pick is an upstream commit, and cloning the fork does not
# necessarily bring it; fetch upstream only if the object is missing.
if ! git cat-file -e "${MONARCH_CHERRY}^{commit}" 2>/dev/null; then
    git remote add upstream "$MONARCH_UPSTREAM" 2>/dev/null || true
    git fetch --quiet upstream
fi
# clone_at resets to MONARCH_REF with --force, so re-running this script drops a
# previous cherry-pick and re-applies it cleanly. -c user.* because a cherry-pick
# writes a commit and a login node often has no git identity configured.
git -c user.name=create_env -c user.email=create_env@localhost \
    cherry-pick "$MONARCH_CHERRY"
GCC="$(dirname -- "$(command -v gcc)")"
# CC/CXX must be GNU gcc: the oneAPI env sets CC=icx, whose intrinsics GNU ld
# cannot link. The cargo linker override replaces conda's
# x86_64-conda-linux-gnu-cc, whose sysroot lacks the glibc 2.38 symbols
# aws-lc-sys emits. RUSTC_BOOTSTRAP lets a stable rustc accept the nightly-only
# flags setup.py passes. USE_TENSOR_ENGINE=0 builds only the actor runtime.
RUSTC_BOOTSTRAP=1 USE_TENSOR_ENGINE=0 \
PROTOC="${CONDA_ENV}/lib/python${PYTHON_VERSION}/site-packages/torch/bin/protoc" \
CC="$GCC/gcc" CXX="$GCC/g++" CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1" \
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$GCC/gcc" \
    pip install -e . --no-deps --no-build-isolation
pip install pyzmq pyarrow requests numpy pyre-extensions "typing-extensions>=4.12" \
    cloudpickle lark tabulate opentelemetry-api clusterscope "flask>=2.0" \
    xxhash py-spy aiohttp
python -c "from monarch.actor import Actor, default_bootstrap_cmd; import monarch._rust_bindings"

# --------------------------------------------------------------- torchstore --
echo "===== TorchStore"
clone_at "$TORCHSTORE_REPO" "$TORCHSTORE_REF" "${REPOS_DIR}/torchstore"
# --no-deps because TorchStore pins an older torchmonarch than the one just built.
pip install -e "${REPOS_DIR}/torchstore" --no-deps --no-build-isolation
pip install pygtrie portpicker

# -------------------------------------------------------- torchtitan deps ----
echo "===== TorchTitan training dependencies"
# TorchTitan runs from the source tree (the launchers cd into TORCHTITAN_DIR);
# it is not pip-installed, so its own dependencies are not resolved for it.
#
# Installed from the checkout's requirements.txt rather than a list written
# out here, for the same reason torch comes from vLLM's requirements/xpu.txt: a
# hand-copied list silently goes stale against the commit it belongs to. It
# already has, twice -- that list omitted `torch_remat`, without which
# `torchtitan.rl.train` does not import at all, and it let vLLM's `datasets`
# 5.1.0 stand against torchtitan's `datasets < 4.8.0` cap.
#
# torchdata, tyro and torch_checkpointing are added because they are imported
# but appear only in pyproject.toml, which nothing here reads.
#
# The constraint keeps pip from replacing the XPU torch with a PyPI CUDA build
# when a transitive dependency declares a plain `torch>=`.
CONSTRAINT="$(mktemp)"
trap 'rm -f "$CONSTRAINT"' EXIT
echo "torch==2.14.0+xpu" > "$CONSTRAINT"
pip install -r "${TORCHTITAN_DIR}/requirements.txt" \
    torchdata==0.11.0 tyro==1.0.15 torch_checkpointing \
    --constraint "$CONSTRAINT"

# Import the entry points now: a missing dependency here is a fast failure on a
# login node instead of a job that dies after claiming an allocation.
( cd "${TORCHTITAN_DIR}" && python -c "
import torchtitan.rl.train, torchtitan.rl.multinode_launcher
import torchtitan_recipes.rl.alphabet_sort_xpu
print('torchtitan RL entry points import OK')
" )

# ------------------------------------------------------------------ record --
PINNED="${REPOS_DIR}/PINNED.txt"
{
    echo "# built $(date -u +%Y-%m-%dT%H:%M:%SZ) on ${HOSTNAME}"
    echo "# conda base ${CONDA_PREFIX_BASE}"
    echo "# conda env  ${CONDA_ENV}"
    for r in torchtitan torchstore monarch vllm; do
        # Tracked modifications only: all four are built in place, so untracked
        # build artifacts are expected and would flag every checkout.
        dirty=""
        [ -n "$(git -C "${REPOS_DIR}/$r" status --porcelain -uno 2>/dev/null)" ] && dirty=" +dirty"
        printf '%-12s %s%s\n' "$r" \
            "$(git -C "${REPOS_DIR}/$r" rev-parse HEAD 2>/dev/null)" "$dirty"
    done
    echo "monarch cherry-pick ${MONARCH_CHERRY}"
    echo
    echo "# versions"
    pip list 2>/dev/null | grep -iE \
        '^(torch|triton|vllm|oneccl|intel-sycl-rt|torchmonarch|torchstore|transformers|datasets|renderers|tyro|spmd|torchdata)' \
        || true
} | tee "$PINNED"

echo "===== Environment ready. Activate with: source ${SCRIPT_DIR}/setup_env.sh"
echo "===== Exact revisions recorded in ${PINNED}"
