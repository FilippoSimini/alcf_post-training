#!/bin/bash
# qsub wrapper around run_grpo.sh.
#
#   PBS_ACCOUNT=<allocation> ARM=full SHAPE=sn NUM_STEPS=10 bash submit_grpo.sh
#   PBS_ACCOUNT=<allocation> ARM=full SHAPE=mn NNODES=8 DP_REPLICATE=4 \
#       NUM_STEPS=10 bash submit_grpo.sh
#
# The upstream launchers carry PBS directives for a different site (their own
# queue, account and filesystems). Flags on the qsub command line take
# precedence over directives inside a script, and this submits run_grpo.sh
# rather than a launcher directly, so none of them apply here -- but it is why
# the queue and filesystem must be passed explicitly rather than left to the
# script.
# No `set -u`: sourcing setup_env.sh calls Lmod's `module` function, whose body
# reads unset variables. Under -u that is a fatal error the surrounding
# `|| true` cannot catch, and the script dies at the module load printing
# nothing at all.
set -eo pipefail

RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${RUN_DIR}/scripts/setup_env.sh"

: "${PBS_ACCOUNT:?set PBS_ACCOUNT to the allocation to charge}"

ARM="${ARM:-lora}"
SHAPE="${SHAPE:-sn}"
case "$SHAPE" in
    sn) NNODES=1 ;;
    mn) NNODES="${NNODES:-4}" ;;
    *)  echo "ERROR: SHAPE must be sn or mn, got '${SHAPE}'" >&2; exit 2 ;;
esac
WALLTIME="${WALLTIME:-01:00:00}"

mkdir -p "${RUN_DIR}/runlogs"

# Forward by value, not by name: `qsub -v VAR` takes whatever the submitting
# shell happens to hold, so an arm's parameters would depend on shell state
# that is not recorded anywhere. Only variables that are actually set are
# passed, so the launchers' own defaults still apply to the rest.
VARS="ARM=${ARM},SHAPE=${SHAPE}"
for v in BASE_DIR REPOS_DIR TORCHTITAN_DIR CONDA_ENV CONDA_PREFIX_BASE \
         ONEAPI_ENV_SCRIPT DATA_MODEL_PATH \
         HF_ASSETS_PATH HF_HOME HF_DATASETS_CACHE \
         HYPERACTOR_MESH_ATTACH_CONFIG_TIMEOUT \
         TORCHINDUCTOR_CACHE_DIR NUM_STEPS VAL_SAMPLES NUM_NODES PPN TP \
         DP_REPLICATE CONFIG TAG EXTRA_ARGS ZE_AFFINITY_MASK; do
    [ -n "${!v:-}" ] && VARS="${VARS},${v}=${!v}"
done

NAME="tt214_${ARM}_${SHAPE}${TAG:+_${TAG}}"
echo "=== qsub ${NAME}: ${NNODES} node(s), ${WALLTIME}, queue ${PBS_QUEUE}, account ${PBS_ACCOUNT}"

qsub -N "${NAME}" \
     -l select="${NNODES}" \
     -l place=scatter \
     -l walltime="${WALLTIME}" \
     -l filesystems="${PBS_FILESYSTEMS}" \
     -q "${PBS_QUEUE}" \
     -A "${PBS_ACCOUNT}" \
     -j oe \
     -o "${RUN_DIR}/runlogs/" \
     -v "${VARS}" \
     "${RUN_DIR}/run_grpo.sh"
