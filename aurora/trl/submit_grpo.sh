#!/bin/bash
# Submit the GRPO job to PBS.
#   PBS_ACCOUNT=<allocation> bash submit_grpo.sh
#   PBS_ACCOUNT=<allocation> NNODES=8 WALLTIME=02:00:00 bash submit_grpo.sh
set -eo pipefail

# Not SCRIPT_DIR: setup_env.sh sets that name too, and sourcing would clobber it.
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${RUN_DIR}/scripts/setup_env.sh"

: "${PBS_ACCOUNT:?set PBS_ACCOUNT to the allocation to charge}"
NNODES="${NNODES:-1}"
WALLTIME="${WALLTIME:-01:00:00}"

echo "Submitting: ${NNODES} node(s), ${WALLTIME}, queue ${PBS_QUEUE}, account ${PBS_ACCOUNT}"

# qsub -v rejects names that are not set, so forward only the ones that are.
FORWARD=""
for v in HF_HOME HF_TOKEN DATA_MODEL_PATH HF_MODEL_REPO BASE_DIR FRAMEWORKS_SCRIPT \
         CONDA_ENV_PATH NRANKS_PER_NODE MAX_STEPS BATCH_SIZE LR NUM_SAMPLES \
         USE_VLLM FSDP_LAYER_CLS OUTDIR EXTRA_ARGS; do
    [ -n "${!v-}" ] && FORWARD="${FORWARD}${FORWARD:+,}${v}=${!v}"
done

# Read the job script from stdin. `qsub -- cmd args` re-splits its arguments on
# whitespace, which breaks any quoted shell command.
qsub \
  -l select="${NNODES}",place=scatter \
  -l walltime="${WALLTIME}" \
  -l filesystems="${PBS_FILESYSTEMS}" \
  -A "${PBS_ACCOUNT}" \
  -q "${PBS_QUEUE}" \
  -N "grpo_trl_${NNODES}n" \
  -k doe -j oe \
  ${FORWARD:+-v "${FORWARD}"} \
  - <<EOF
#!/bin/bash -l
cd "${RUN_DIR}"
bash run_grpo.sh
EOF
