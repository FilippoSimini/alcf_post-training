# Post-training LLMs on ALCF systems

Reproducible reinforcement-learning (GRPO) examples for ALCF supercomputers.
Each leaf directory is self-contained: build an environment, submit a job, read
the result, with the exact commands in its README.

## What is here

| System | Framework | Task | Status |
|---|---|---|---|
| [Aurora / Sunspot](aurora/) | [TRL](aurora/trl/) | Countdown game, Qwen3 | |
| [Aurora / Sunspot](aurora/) | [torchtitan](aurora/torchtitan/) | Alphabet sort, Qwen3 + LoRA | |
| Polaris | TRL | | not yet |
| Polaris | torchtitan | | not yet |

Aurora and Sunspot share one directory: the scripts detect the host and adjust
the filesystem, queue, and how the software stack is loaded.

## Layout

```
<system>/<framework>/
├── README.md              end-to-end recipe, hyperparameters, measured results
├── scripts/
│   ├── create_env.sh      one-time: build the Python environment
│   ├── setup_env.sh       per-shell: load modules and activate
│   └── download_assets.sh fetch models and datasets
├── configure_ccl.sh       oneCCL / libfabric tuning for this launch shape
├── submit_grpo.sh         qsub wrapper
└── run_grpo.sh            the launcher that runs on the compute nodes
```

## Configuration

Nothing site- or user-specific is hardcoded. Every script reads its
configuration from the environment, with defaults derived from `$HOME`, `$USER`,
or the host name. Only `PBS_ACCOUNT` has no default.

| Variable | Default | Meaning |
|---|---|---|
| `PBS_ACCOUNT` | *(none — required)* | Allocation to charge |
| `PBS_QUEUE` | per host | `next-eval` on Aurora, `workq` on Sunspot |
| `PBS_FILESYSTEMS` | `home:$FSYS` | `flare` on Aurora, `tegu` on Sunspot |
| `NNODES` | `1` | Nodes to request |
| `WALLTIME` | `01:00:00` | Job walltime |
| `BASE_DIR` | the framework directory | Where `venv/` and `venv_repos/` are built |
| `HF_HOME` | `$HOME/.cache/huggingface` | HuggingFace cache |
| `HF_TOKEN` | contents of `$HF_HOME/token` | HuggingFace token |
| `DATA_MODEL_PATH` | `$HF_HOME/local` | Where models and datasets are downloaded |
| `HF_MODEL_REPO` | per framework | Model to train, e.g. `Qwen/Qwen3-0.6B` |

On a system with a small home directory, point the heavy paths at a project
filesystem before running anything:

```bash
export BASE_DIR=/lus/<fs>/projects/<project>/$USER/alcf_post-training
export HF_HOME=/lus/<fs>/projects/<project>/$USER/huggingface/cache
export DATA_MODEL_PATH=/lus/<fs>/projects/<project>/$USER/huggingface
```

## Getting started

```bash
git clone <this repo>
cd alcf_post-training/aurora/trl
bash scripts/create_env.sh          # once
source scripts/setup_env.sh
bash scripts/download_assets.sh
PBS_ACCOUNT=<your allocation> bash submit_grpo.sh
```

See [aurora/trl/README.md](aurora/trl/README.md) for the full walkthrough.
