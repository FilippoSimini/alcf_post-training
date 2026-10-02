#!/bin/bash
# Where models, datasets and the HuggingFace cache live, and which ones to use.
# Separate from create_env.sh/setup_env.sh on purpose: those build and activate
# the Python environment and have nothing to do with assets.
#
# Sourced by download_assets.sh and run_grpo.sh. Every value is overridable;
# set them in your shell or your submit script to point somewhere else.
#
#   export HF_HOME=/lus/<fs>/projects/<project>/$USER/huggingface/cache
#   export DATA_MODEL_PATH=/lus/<fs>/projects/<project>/$USER/huggingface
#   source scripts/hf_env.sh

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

# The token normally lives under HF_HOME, but HF_HOME is usually redirected to
# a project filesystem while `hf auth login` writes to the home cache.
export HF_TOKEN="${HF_TOKEN:-$(cat "${HF_HOME}/token" 2>/dev/null \
    || cat "${HOME}/.cache/huggingface/token" 2>/dev/null)}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export DATA_MODEL_PATH="${DATA_MODEL_PATH:-${HF_HOME}/local}"

export HF_MODEL_REPO="${HF_MODEL_REPO:-Qwen/Qwen3-0.6B}"
export HF_DATA_REPO="${HF_DATA_REPO:-kalomaze/alphabetic-arxiv-authors-it1}"
export HF_ASSETS_PATH="${HF_ASSETS_PATH:-${DATA_MODEL_PATH}/${HF_MODEL_REPO}}"
