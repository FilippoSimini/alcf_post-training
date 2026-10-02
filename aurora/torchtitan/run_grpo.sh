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

# TORCHTITAN_SRC selects an alternative torchtitan source tree -- typically a
# `git worktree` of venv_repos/torchtitan carrying a patch -- so a patched and
# an unpatched arm can run CONCURRENTLY against the same venv.
#
# This works because setuptools' PEP 660 editable install registers its finder
# with sys.meta_path.APPEND, i.e. after the default PathFinder, so sys.path
# still wins. Verified: with PYTHONPATH set to a worktree, `generator.__file__`
# resolves into that worktree. If a future setuptools inserts the finder at the
# front instead, this silently stops working -- which is why the header below
# prints the resolved path of the module that actually loaded.
if [ -n "${TORCHTITAN_SRC:-}" ]; then
    [ -d "${TORCHTITAN_SRC}/torchtitan" ] || {
        echo "ERROR: TORCHTITAN_SRC=${TORCHTITAN_SRC} has no torchtitan/ package" >&2
        exit 1
    }
    export PYTHONPATH="${TORCHTITAN_SRC}:${PYTHONPATH}"
fi
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
# cd into the tree that will be imported, not just the installed one. Python
# puts the cwd at the FRONT of sys.path for `python -c` and for Monarch's
# spawned actors, ahead of PYTHONPATH — so running from the unpatched checkout
# silently shadows a TORCHTITAN_SRC worktree in exactly the processes that run
# the generator. Making cwd the selected tree makes every resolution order
# (cwd, PYTHONPATH, editable finder) agree on the same source.
cd "${TORCHTITAN_SRC:-${TORCHTITAN_DIR}}" || exit 1

echo "Log:   ${LOG}"

# This header goes INTO the log, not just to the job's stdout. An earlier
# version echoed it before the tee'd pipeline, so it landed in the PBS .o file
# and the run log itself carried no provenance at all -- which is the exact
# failure it was added to prevent.
{
    echo "=== GRPO+LoRA: ${NUM_NODES}/${AVAILABLE_NODES} nodes x ${PPN} tiles," \
         "${MODULE}/${CONFIG}, TP=${TP} DP_REPLICATE=${DP_REPLICATE}, ${NUM_STEPS} steps ==="
    echo "Nodes: ${ALL_NODES}"
    echo "Args:  VAL_SAMPLES=${VAL_SAMPLES} EXTRA_ARGS=${EXTRA_ARGS:-none}"
    echo "Repo:  $(git -C "${RUN_DIR}" describe --always --dirty --abbrev=12 2>/dev/null || echo 'not a git checkout') \
($(git -C "${RUN_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?'))"
    [ -f "${REPOS_DIR}/PINNED.txt" ] && sed 's/^/Venv:  /' "${REPOS_DIR}/PINNED.txt"
    # PINNED.txt is written at build time, but the installs are editable, so a
    # `git apply` in a checkout — or a TORCHTITAN_SRC worktree — changes what
    # runs without changing that file. Report the live state, or the log will
    # mislabel a patched run as clean.
    for d in "${REPOS_DIR}"/torchstore "${REPOS_DIR}"/monarch \
             "${REPOS_DIR}"/torchtitan "${TORCHTITAN_SRC:-}"; do
        [ -n "$d" ] && [ -e "$d/.git" ] || continue
        if git -C "$d" diff --quiet 2>/dev/null; then
            echo "State: $(basename "$d") clean @ $(git -C "$d" rev-parse --short=12 HEAD)"
        else
            echo "State: $(basename "$d") MODIFIED @ $(git -C "$d" rev-parse --short=12 HEAD) -- \
$(git -C "$d" diff --shortstat | sed 's/^ *//')"
        fi
    done
    # Ground truth for which arm ran: ask Python which file it imports, from
    # the same cwd the actors will have. `tail -1` because importing torch
    # prints warnings to stdout that would otherwise be captured as the path.
    echo "Source: $(python3 -c 'import torchtitan.experiments.rl.actors.generator as g; print(g.__file__)' 2>/dev/null | tail -1)"
    echo "Pulled: $(grep -cE '^\s*for node_idx in range' \
        "$(python3 -c 'import torchtitan.experiments.rl.actors.generator as g; print(g.__file__)' 2>/dev/null | tail -1)" \
        2>/dev/null | sed 's/^0$/concurrent (patched)/; s/^1$/serialized (stock)/')"
} 2>&1 | tee "$LOG"

# The launcher never returns: after the last step is logged and the generators
# log shutdown, Monarch/PALS teardown hangs — 25 minutes to the full walltime.
# So the run ends when the LOG says the work is done, not when the process
# exits. This watcher reclaims the idle time; without it every job sits until
# PBS kills it.
#
# Both conditions must hold: every step done, AND the post-training validation
# summary present when validation is on. Requiring the step count too is what
# makes this safe against periodic validation -- `ValidationConfig` carries a
# "TODO: enable periodic validation", and the day that lands, a mid-run
# summary would otherwise look like the end and get the job killed with work
# still to do. Steps-complete cannot be true early, so the conjunction holds.
work_is_done() {
    [ -s "$LOG" ] || return 1
    local done_steps
    done_steps=$(grep -aoE 'Train \| Step: +[0-9]+' "$LOG" 2>/dev/null \
        | tail -1 | grep -oE '[0-9]+$')
    [ "${done_steps:-0}" -ge "$NUM_STEPS" ] 2>/dev/null || return 1
    # controller.py:835 prints this once, after the last step. Pre-training
    # validation logs "Validation | Step", which does not match.
    [ "${VAL_SAMPLES}" -gt 0 ] 2>/dev/null || return 0
    grep -aq "Validation reward" "$LOG"
}

(
    while sleep 15; do
        work_is_done || continue
        sleep 10   # let the final lines flush before killing the writers
        echo "=== work complete; tearing down (teardown otherwise hangs) ==="
        # Kills the launcher on every node, including this one, which lets the
        # foreground mpiexec below return. Stale Monarch workers on port 26601
        # silently corrupt a later run on the same nodes, so this is not
        # optional even when the job is about to end.
        "$MPIEXEC" -n "$NUM_NODES" -ppn 1 --hosts "$ALL_NODES" --cpu-bind none \
            pkill -9 -f 'multinode_launcher|torchtitan.experiments.rl|monarch|VLLM' \
            >/dev/null 2>&1
        break
    done
) &
WATCHER_PID=$!

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
    2>&1 | tee -a "$LOG"

RC=${PIPESTATUS[0]}
kill "$WATCHER_PID" 2>/dev/null

# The watcher kills the launcher, so a successful run now exits on a signal.
# Judge the run by what it produced, not by how the process died — the
# `exit 42` sentinel below already exists because exit codes here are not
# meaningful.
[ "$RC" -eq 42 ] && RC=0
STEPS_DONE=$(grep -aoE 'Train \| Step: +[0-9]+' "$LOG" 2>/dev/null \
    | tail -1 | grep -oE '[0-9]+$')
if [ "${STEPS_DONE:-0}" -ge "$NUM_STEPS" ] 2>/dev/null; then
    RC=0
else
    echo "WARNING: only ${STEPS_DONE:-0}/${NUM_STEPS} steps completed." >&2
    [ "$RC" -eq 0 ] && RC=1
fi
echo "Exit code: $RC"
echo "Log: $LOG"
exit "$RC"
