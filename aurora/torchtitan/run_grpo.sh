#!/bin/bash
# GRPO + LoRA launcher for torchtitan's experiments/rl on Intel XPU.
#
# The launcher splits the allocation in half: the first nodes run FSDP2
# trainers, the rest run vLLM generators. Weights move from trainer to
# generator through TorchStore. One MPI rank per node bootstraps Monarch, which
# then spawns PPN actors per node — hence `mpiexec -ppn 1`.
#
#   bash run_grpo.sh
#   NUM_NODES=8 NUM_STEPS=50 bash run_grpo.sh
set +e

RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${RUN_DIR}/scripts/setup_env.sh"
source "${RUN_DIR}/scripts/hf_env.sh"
source "${RUN_DIR}/configure_ccl.sh"

# Cray PALS by absolute path: the frameworks stack puts Intel MPI's Hydra
# mpiexec ahead of PALS on PATH, and under Hydra a cross-node run dies in a
# oneCCL SEND fault.
MPIEXEC="${MPIEXEC:-/opt/cray/pals/1.8/bin/mpiexec}"

# gcc-13 toolchain for the SYCL compiler driver, if present.
for GCC13_ROOT in /opt/aurora/*/spack/unified/*/install/linux-x86_64/gcc-13.4.0-*/; do
    if [ -d "$GCC13_ROOT" ]; then
        export CCC_OVERRIDE_OPTIONS="+--gcc-toolchain=${GCC13_ROOT%/}"
        export PATH="${GCC13_ROOT%/}/bin:$PATH"
        break
    fi
done

# torchtitan's --module takes a fully qualified module path, so our extra
# configs live in configs/alcf_rl rather than being patched into the checkout.
export PYTHONPATH="${RUN_DIR}/configs${PYTHONPATH:+:$PYTHONPATH}"
MODULE="${MODULE:-alcf_rl}"
CONFIG="${CONFIG:-rl_grpo_lora_qwen3_0_6b_lr2e5}"
NUM_STEPS="${NUM_STEPS:-10}"
# Non-zero so pre-training validation reports a baseline. At 0 the controller
# logs that it is validating and then reports nothing.
VAL_SAMPLES="${VAL_SAMPLES:-32}"
PPN="${PPN:-4}"
# TP=1, DP_REPLICATE=1 (pure dp_shard) is the fastest shape measured at every
# node count. Keep tensor parallelism inside a node: cross-node TP is correct
# but roughly 10x slower.
TP="${TP:-1}"
DP_REPLICATE="${DP_REPLICATE:-1}"
# Per-job by default. Concurrent jobs sharing one dump folder interleave their
# structured logs and rollout samples, and the checkpoint wipe below would
# delete a sibling run's checkpoint mid-flight. Interactive runs (no PBS_JOBID)
# keep the plain path.
DUMP_FOLDER="${DUMP_FOLDER:-outputs/rl_grpo${PBS_JOBID:+_${PBS_JOBID%%.*}}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ ! -f "${HF_ASSETS_PATH}/config.json" ]; then
    echo "ERROR: no model at ${HF_ASSETS_PATH} — run scripts/download_assets.sh" >&2
    exit 1
fi
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

# Bare management hostnames for both mpiexec --hosts and Monarch. The data
# plane still rides CXI. The .hsn names are multi-rail and cause
# MESH_ATTACH_CONFIG_TIMEOUT.
if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE}" ]]; then
    mapfile -t NODE_LIST < <(sort -u "$PBS_NODEFILE" | sed 's/[-.]hsn.*//')
else
    NODE_LIST=("$(hostname -s)")
fi
AVAILABLE_NODES=${#NODE_LIST[@]}
NUM_NODES="${NUM_NODES:-$AVAILABLE_NODES}"
if [ "$NUM_NODES" -gt "$AVAILABLE_NODES" ]; then
    echo "ERROR: NUM_NODES=$NUM_NODES but only $AVAILABLE_NODES allocated." >&2
    exit 1
fi
ALL_NODES=$(IFS=,; echo "${NODE_LIST[*]:0:$NUM_NODES}")

# Nodes occasionally come up with the project filesystem unreadable, which
# surfaces much later as a confusing training error.
UNHEALTHY=$("$MPIEXEC" -n "$NUM_NODES" -ppn 1 --hosts "$ALL_NODES" --cpu-bind none \
    /bin/bash -c "test -r '${HF_ASSETS_PATH}/config.json' || hostname" 2>/dev/null)
if [ -n "$UNHEALTHY" ]; then
    echo "ERROR: these nodes cannot read ${HF_ASSETS_PATH}:" >&2
    echo "$UNHEALTHY" >&2
    exit 1
fi

mkdir -p "${RUN_DIR}/runlogs"
LOG="${RUN_DIR}/runlogs/run_grpo_$(date +%Y%m%d_%H%M%S)_${NUM_NODES}n${PBS_JOBID:+_${PBS_JOBID%%.*}}.log"

# KEEP_CACHE=1 keeps the Triton cache between runs. Worth it when sweeping
# configs inside one allocation: recompiling SYCL kernels costs several minutes
# per run and the kernels do not depend on the knobs being swept.
rm -rf "${TORCHTITAN_DIR}/${DUMP_FOLDER}/checkpoint/" 2>/dev/null
[ -z "${KEEP_CACHE:-}" ] && rm -rf "/tmp/${USER}/torchinductor_xpu/triton" 2>/dev/null
cd "${TORCHTITAN_DIR}" || exit 1

echo "=== GRPO+LoRA: ${NUM_NODES}/${AVAILABLE_NODES} nodes x ${PPN} tiles," \
     "${MODULE}/${CONFIG}, TP=${TP} DP_REPLICATE=${DP_REPLICATE}, ${NUM_STEPS} steps ==="
echo "Nodes: ${ALL_NODES}"
echo "Log:   ${LOG}"

# Which code produced this run. Without these two lines a log cannot be tied to
# a repo state, and copies on different machines do drift.
echo "Repo:  $(git -C "${RUN_DIR}" describe --always --dirty --abbrev=12 2>/dev/null || echo 'not a git checkout') \
($(git -C "${RUN_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?'))"
[ -f "${REPOS_DIR}/PINNED.txt" ] && sed 's/^/Venv:  /' "${REPOS_DIR}/PINNED.txt"

# --cpu-bind none is required. At -ppn 1 PALS binds the rank to a single core,
# and every Monarch actor and vLLM worker forked from it inherits that mask —
# ~126 threads sharing 1 core of 208, costing 2-4x end to end.
"$MPIEXEC" -n "$NUM_NODES" -ppn 1 --hosts "$ALL_NODES" --cpu-bind none --envall \
    python3 "${RUN_DIR}/multinode_launcher.py" \
    --num_nodes="$NUM_NODES" \
    --gpus_per_node="$PPN" \
    --num_trainer_nodes="${NUM_TRAINER_NODES:-0}" \
    --all_nodes="$ALL_NODES" \
    --module "$MODULE" --config "$CONFIG" \
    --hf_assets_path="$HF_ASSETS_PATH" \
    --async_loop.num_training_steps="$NUM_STEPS" \
    --async_loop.validation.num_samples="$VAL_SAMPLES" \
    --trainer.parallelism.tensor_parallel_degree="$TP" \
    --trainer.parallelism.data_parallel_replicate_degree="$DP_REPLICATE" \
    --dump_folder="$DUMP_FOLDER" \
    $EXTRA_ARGS \
    2>&1 | tee "$LOG"

RC=${PIPESTATUS[0]}
# multinode_launcher.py exits 42 on success, deliberately non-zero so PALS
# tears down the remaining ranks instead of waiting ~17 minutes for them.
[ "$RC" -eq 42 ] && RC=0
echo "Exit code: $RC"
echo "Log: $LOG"
exit "$RC"
