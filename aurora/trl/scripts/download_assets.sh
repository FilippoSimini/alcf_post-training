#!/bin/bash
# Download the model into $DATA_MODEL_PATH. Run from a login node, which is the
# only place with outbound network access.
#   source scripts/setup_env.sh && bash scripts/download_assets.sh
#   HF_MODEL_REPO=Qwen/Qwen3-4B bash scripts/download_assets.sh
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.sh"
source "${SCRIPT_DIR}/hf_env.sh"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-$HTTP_PROXY}"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"

TARGET="${DATA_MODEL_PATH}/${HF_MODEL_REPO}"
if [ -f "${TARGET}/config.json" ]; then
    echo "===== Already present: ${TARGET}"
else
    echo "===== Downloading ${HF_MODEL_REPO} to ${TARGET}"
    hf download "${HF_MODEL_REPO}" --repo-type=model --local-dir="${TARGET}"
fi

# The Countdown dataset is generated on the fly by generate_countdown.py, so
# there is nothing else to fetch.
echo "===== Done. MODEL_DIR=${TARGET}"
