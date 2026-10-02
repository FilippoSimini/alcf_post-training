#!/bin/bash
# Submit the torchtitan GRPO job to PBS.
#   PBS_ACCOUNT=<allocation> bash submit_grpo.sh
#   PBS_ACCOUNT=<allocation> NNODES=8 WALLTIME=02:00:00 bash submit_grpo.sh
#
# The launcher splits the allocation into trainers and generators, so NNODES
# must be even and at least 2 for a multi-node run.
set -eo pipefail

# Not SCRIPT_DIR: setup_env.sh sets that name too, and sourcing would clobber it.
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${RUN_DIR}/scripts/setup_env.sh"

: "${PBS_ACCOUNT:?set PBS_ACCOUNT to the allocation to charge}"
NNODES="${NNODES:-1}"
WALLTIME="${WALLTIME:-01:30:00}"

echo "Submitting: ${NNODES} node(s), ${WALLTIME}, queue ${PBS_QUEUE}, account ${PBS_ACCOUNT}"

# qsub -v rejects names that are not set, so forward only the ones that are.
FORWARD=""
for v in HF_HOME HF_TOKEN DATA_MODEL_PATH HF_MODEL_REPO HF_DATA_REPO HF_ASSETS_PATH \
         BASE_DIR REPOS_DIR FRAMEWORKS_SCRIPT CONDA_ENV_PATH \
         MODULE CONFIG NUM_NODES NUM_TRAINER_NODES NUM_STEPS VAL_SAMPLES \
         PPN TP DP_REPLICATE LR KEEP_CACHE \
         TORCHSTORE_XCCL_ENABLED DUMP_FOLDER EXTRA_ARGS; do
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
  -N "grpo_titan_${NNODES}n" \
  -k doe -j oe \
  ${FORWARD:+-v "${FORWARD}"} \
  - <<EOF
#!/bin/bash -l
cd "${RUN_DIR}"
bash run_grpo.sh
EOF
