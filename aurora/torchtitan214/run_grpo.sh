#!/bin/bash
# Run one GRPO arm. Works inside a batch job and inside an interactive
# allocation; submit_grpo.sh qsubs this file.
#
#   ARM=lora SHAPE=sn  NUM_STEPS=10 bash run_grpo.sh
#   ARM=full SHAPE=mn  NUM_STEPS=10 bash run_grpo.sh
#
# This is a thin wrapper. It sets the environment for this system and then
# execs the upstream launcher, which carries the fabric and oneCCL tuning --
# every export there has a comment naming the bug it closes, and a second copy
# of that block here would be free to drift from it.
#
# | variable      | default | meaning                                        |
# |---------------|---------|------------------------------------------------|
# | ARM           | lora    | lora (adapters, rank 32) or full (all params)  |
# | SHAPE         | sn      | sn (1 node, 4 tiles) or mn (multi-node)        |
# | NUM_STEPS     | 10      | training steps                                 |
# | VAL_SAMPLES   | 32      | validation samples, mn only (sn has no knob)   |
# | NUM_NODES     | all     | use the first N nodes of the allocation        |
# | PPN           | 4       | tiles per node                                 |
# | TP            | 1       | tensor parallel degree                         |
# | DP_REPLICATE  | 1       | data parallel replicate degree                 |
# | CONFIG        | per arm | entry in torchtitan_recipes.rl.alphabet_sort_xpu |
# | TAG           | empty   | distinguishes arms sharing one allocation      |
# | EXTRA_ARGS    | empty   | passed through to the training command         |
# No `set -u`: sourcing setup_env.sh calls Lmod's `module` function, whose body
# reads unset variables. Under -u that is a fatal error the surrounding
# `|| true` cannot catch, and the script dies at the module load printing
# nothing at all.
set -o pipefail

# NOT from BASH_SOURCE alone: PBS copies the submitted script into a spool
# directory, so under qsub BASH_SOURCE resolves to /var/spool/pbs/... and the
# source below silently fails. submit_grpo.sh forwards BASE_DIR for exactly
# this reason; fall back to BASH_SOURCE when run by hand from the repo.
if [ -n "${BASE_DIR:-}" ] && [ -r "${BASE_DIR}/scripts/setup_env.sh" ]; then
    RUN_DIR="${BASE_DIR}"
else
    RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${RUN_DIR}/scripts/setup_env.sh" || {
    echo "ERROR: cannot source ${RUN_DIR}/scripts/setup_env.sh" >&2; exit 2; }

ARM="${ARM:-lora}"
SHAPE="${SHAPE:-sn}"
export NUM_STEPS="${NUM_STEPS:-10}"
export VAL_SAMPLES="${VAL_SAMPLES:-32}"
# The single-node launchers never pass VAL_SAMPLES on to the recipe, so set the
# variable the recipe actually reads. The multi-node launchers overwrite it from
# their own VAL_SAMPLES, which is this same value.
export TITAN_XPU_VAL_SAMPLES="${VAL_SAMPLES}"

case "${ARM}:${SHAPE}" in
    lora:sn) LAUNCHER=run_grpo_lora_sn.sh        ;;
    full:sn) LAUNCHER=run_grpo_sn.sh             ;;
    lora:mn) LAUNCHER=run_grpo_lora_multinode.sh ;;
    full:mn) LAUNCHER=run_grpo_multinode.sh      ;;
    *) echo "ERROR: ARM must be lora|full and SHAPE must be sn|mn, got ${ARM}:${SHAPE}" >&2
       exit 2 ;;
esac
LAUNCHER="${TORCHTITAN_DIR}/scripts/rl/xpu/${LAUNCHER}"
[ -r "${LAUNCHER}" ] || { echo "ERROR: ${LAUNCHER} not found; run scripts/create_env.sh" >&2; exit 2; }

# One dump folder per job and arm. Sharing one folder between arms interleaves
# their rollout_samples.jsonl rows, which silently corrupts any measurement
# taken from the file afterwards.
JOBID="${PBS_JOBID%%.*}"
export DUMP_FOLDER="${DUMP_FOLDER:-outputs/rl_${ARM}_${SHAPE}${JOBID:+_${JOBID}}${TAG:+_${TAG}}}"

# The launchers default ZE_AFFINITY_MASK to the four tiles 0,1,2,3 and never
# widen it, so a PPN above 4 would ask for more ranks per node than there are
# visible tiles. Derive the mask from PPN unless the caller pinned one.
if [ -z "${ZE_AFFINITY_MASK:-}" ] && [ "${PPN:-4}" -ne 4 ] 2>/dev/null; then
    # `seq -s,` appends a trailing separator on BSD but not GNU; join instead.
    ZE_AFFINITY_MASK="$(seq 0 $(( ${PPN} - 1 )) | paste -sd, -)"
    export ZE_AFFINITY_MASK
fi

# The launcher derives its mpiexec --hosts list from PBS_NODEFILE and strips
# only the DNS domain (`sed 's/\..*//'`). That is enough where the file lists
# `<node>.hsn.<domain>`, but here it lists `<node>-hsn0.<domain>`, so the
# multi-rail interface name survives the strip and PALS rejects every host with
# "launch failed on <node>: Couldn't find this node in nodes array". Hand the
# launcher a nodefile of bare management names instead; its own strip then has
# nothing left to do. The data plane still rides CXI/hsn0 via oneCCL.
if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
    _nodefile="/tmp/tt214_nodes.${PBS_JOBID%%.*}"
    sed 's/[-.]hsn.*//' "${PBS_NODEFILE}" | sort -u > "${_nodefile}"
    export PBS_NODEFILE="${_nodefile}"
    echo "=== hosts:    $(tr '\n' ' ' < "${_nodefile}")"
fi

echo "=== torchtitan214: ARM=${ARM} SHAPE=${SHAPE} steps=${NUM_STEPS} val=${VAL_SAMPLES}"
echo "=== launcher: ${LAUNCHER}"
echo "=== dump:     ${TORCHTITAN_DIR}/${DUMP_FOLDER}"
# The four repo SHAs, read live from the checkouts. NOT from PINNED.txt: that
# file is written once, by create_env.sh, and says nothing about the working
# tree afterwards. Checking a repo out at another ref -- to bisect a defect, or
# to test a fix -- leaves the file behind, and every run then reports the SHA of
# a build that is no longer installed. Observed: two runs on a checkout moved to
# a newer torchtitan both reported the older build. `git rev-parse` cannot go
# stale, and +dirty marks a tree carrying edits that no SHA describes.
printf '=== revs:    '
for _repo in torchtitan torchstore monarch vllm; do
    _sha="$(git -C "${REPOS_DIR}/${_repo}" rev-parse --short=8 HEAD 2>/dev/null || echo unknown)"
    # -uno: all four are built in place, so untracked build artifacts are
    # expected and would otherwise mark every checkout dirty.
    [ -n "$(git -C "${REPOS_DIR}/${_repo}" status --porcelain -uno 2>/dev/null)" ] && _sha="${_sha}+dirty"
    printf '%s=%s ' "${_repo}" "${_sha}"
done
printf '\n'

exec bash "${LAUNCHER}"
