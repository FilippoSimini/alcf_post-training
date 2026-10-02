#!/bin/bash
# Download the model and dataset into $DATA_MODEL_PATH. Run from a login node,
# which is the only place with outbound network access.
#   source scripts/setup_env.sh && bash scripts/download_assets.sh
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.sh"
source "${SCRIPT_DIR}/hf_env.sh"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-$HTTP_PROXY}"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"

# `hf download` can succeed and still exit non-zero: the CLI lets typer's
# `click.exceptions.Exit: 0` escape, printing a traceback after writing the
# files. Under `set -e` that aborts the script with everything already on disk.
# So ignore its status and judge by whether the files arrived.
fetch() {  # repo_id  repo_type  target  sentinel
    if [ -e "$3/$4" ]; then
        echo "===== Already present: $3"
        return
    fi
    echo "===== Downloading $1"
    hf download "$1" --repo-type="$2" --local-dir="$3" || true
    if [ ! -e "$3/$4" ]; then
        echo "ERROR: $1 did not produce $3/$4" >&2
        exit 1
    fi
}

MODEL_TARGET="${DATA_MODEL_PATH}/${HF_MODEL_REPO}"
fetch "${HF_MODEL_REPO}" model "${MODEL_TARGET}" config.json

DATA_TARGET="${DATA_MODEL_PATH}/${HF_DATA_REPO}"
fetch "${HF_DATA_REPO}" dataset "${DATA_TARGET}" alphabetic_names.jsonl

# The run sets HF_HUB_OFFLINE, so the dataset also has to be in the datasets
# cache that `load_dataset` consults, not only in the local-dir copy.
echo "===== Warming the datasets cache"
python3 -c "
import os
from datasets import load_dataset
load_dataset('${HF_DATA_REPO}', cache_dir=os.environ['HF_DATASETS_CACHE'])
"

echo "===== Done. HF_ASSETS_PATH=${MODEL_TARGET}"
