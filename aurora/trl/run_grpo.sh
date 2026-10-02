#!/bin/bash -l
# GRPO launcher: one MPI rank per XPU tile (12 per node).
# Runs inside a PBS allocation, either interactive or via submit_grpo.sh.
#
#   bash run_grpo.sh
#   MAX_STEPS=20 USE_VLLM=1 bash run_grpo.sh

if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_PATH="${BASH_SOURCE[0]}"
elif [ -n "$ZSH_VERSION" ]; then
    SCRIPT_PATH="${(%):-%x}"
else
    SCRIPT_PATH="$0"
fi
export GRPO_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"

source "${GRPO_DIR}/scripts/setup_env.sh"
source "${GRPO_DIR}/scripts/hf_env.sh"
source "${GRPO_DIR}/configure_ccl.sh"

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1,2,3,4,5,6,7,8,9,10,11}"

# ── Node/rank setup ──────────────────────────────────────────────────────────
mapfile -t NODES < "$PBS_NODEFILE"
export NNODES=${#NODES[@]}
export NRANKS_PER_NODE="${NRANKS_PER_NODE:-12}"
export MASTER_NODE="${NODES[0]}"
if [[ "$MASTER_NODE" == *.hsn.cm.*.alcf.anl.gov ]]; then
    export MASTER_ADDR="$MASTER_NODE"
else
    export MASTER_ADDR="${MASTER_NODE}.hsn.cm.${SYSTEM}.alcf.anl.gov"
fi
export MASTER_PORT=$((20000 + RANDOM % 20000))
export GLOO_USE_IPV6=0

echo "MASTER=$MASTER_ADDR:$MASTER_PORT  NNODES=$NNODES  PPN=$NRANKS_PER_NODE"

# ── Node health preflight ────────────────────────────────────────────────────
# Nodes occasionally come up with the project filesystem unmounted. At 64 nodes
# that surfaces hundreds of steps in as a confusing training error, so check the
# model directory is readable everywhere before starting.
export MODEL_DIR="${MODEL_DIR:-${DATA_MODEL_PATH}/${HF_MODEL_REPO}}"
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "ERROR: no model at ${MODEL_DIR} — run scripts/download_assets.sh" >&2
    exit 1
fi
if ! mpiexec --hostfile "$PBS_NODEFILE" -n "$NNODES" --ppn 1 \
        /bin/bash -c "test -r '${MODEL_DIR}/config.json' || { echo \"UNHEALTHY: \$(hostname)\"; exit 1; }"; then
    echo "ERROR: some nodes cannot read ${MODEL_DIR}; resubmit" >&2
    exit 1
fi

mkdir -p "${GRPO_DIR}/runlogs"
export RUNLOG="${GRPO_DIR}/runlogs/run_grpo_$(date +%Y%m%d_%H%M%S)_${NNODES}n.log"

# ── Run configuration (override via env) ─────────────────────────────────────
export OUTDIR="${OUTDIR:-${GRPO_DIR}/output/$(basename "${HF_MODEL_REPO}")-countdown-grpo}"
export SCRIPT="${GRPO_DIR}/train_grpo.py"
export NUM_SAMPLES="${NUM_SAMPLES:-50000}"
export MAX_STEPS="${MAX_STEPS:-40}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export LR="${LR:-5e-6}"
# Qwen3/Qwen2/Llama name their decoder block differently and FSDP wraps by class
# name, so this has to track the model.
export FSDP_LAYER_CLS="${FSDP_LAYER_CLS:-Qwen3DecoderLayer}"
export VLLM_ARGS="${USE_VLLM:+--use_vllm}"
export EXTRA_ARGS="${EXTRA_ARGS:-}"

mpiexec --hostfile "$PBS_NODEFILE" -n "$((NNODES * NRANKS_PER_NODE))" --ppn "$NRANKS_PER_NODE" \
  --genvall \
  /bin/bash -c '
    source /etc/profile && source ~/.bashrc
    source "${GRPO_DIR}/scripts/setup_env.sh"
    # Idempotent: every value is ${VAR:-default}, so whatever --genvall carried
    # in from the launching shell wins.
    source "${GRPO_DIR}/scripts/hf_env.sh"

    # module load resets some of these, so re-apply after the env is up.
    source "${GRPO_DIR}/configure_ccl.sh"

    # Map MPI/PALS rank variables onto the ones torch.distributed reads.
    export RANK="${PMI_RANK:-${PMIX_RANK:-${PALS_RANKID:-0}}}"
    export LOCAL_RANK="${PALS_LOCAL_RANKID:-0}"
    export WORLD_SIZE="${PMI_SIZE:-$((NNODES * NRANKS_PER_NODE))}"
    export LOCAL_WORLD_SIZE="${NRANKS_PER_NODE}"

    # CCL_PROCESS_LAUNCHER=none means oneCCL will not derive these itself.
    for v in CCL_LOCAL_RANK PALS_LOCAL_RANKID; do
      [ -n "${!v-}" ] && export CCL_LOCAL_RANK="${!v}" && break
    done
    for v in CCL_LOCAL_SIZE PALS_LOCAL_SIZE; do
      [ -n "${!v-}" ] && export CCL_LOCAL_SIZE="${!v}" && break
    done

    [ "$RANK" -eq 0 ] && mkdir -p "$OUTDIR"

    python "$SCRIPT" \
        --model_id    "$MODEL_DIR" \
        --output_dir  "$OUTDIR" \
        --num_samples "$NUM_SAMPLES" \
        --batch_size  "$BATCH_SIZE" \
        --grad_accum  1 \
        --lr          "$LR" \
        --warmup_steps 10 \
        --max_steps   "$MAX_STEPS" \
        --num_generations 4 \
        --max_completion_length 512 \
        --temperature 1.0 \
        --beta        0.0 \
        --loss_type   grpo \
        --fsdp_transformer_layer_cls "$FSDP_LAYER_CLS" \
        --logging_steps 1 \
        --save_steps  "$MAX_STEPS" \
        --save_limit  1 \
        --num_workers 0 \
        --log_completions \
        $VLLM_ARGS \
        $EXTRA_ARGS
  ' 2>&1 | tee -a "$RUNLOG"

echo "Run log:    $RUNLOG"
echo "Output dir: ${OUTDIR}"
