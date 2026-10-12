# Post-training LLMs on ALCF systems

Reproducible reinforcement-learning (GRPO) examples for ALCF supercomputers.
Each leaf directory is self-contained: build an environment, submit a job, read
the result, with the exact commands in its README.

## What is here

One leaf directory per combination of system, framework and task. Each carries
its own README, which is the authority for that example: what it trains, the
hyperparameters, its current status and the results measured on real hardware.
Browse the top-level system directories to see which combinations exist.

Where a system has a test and development twin running the same stack, both
share one directory — the scripts detect the host and adjust the filesystem,
queue, and how the software stack is loaded.

## Layout

```
<system>/<framework>/
├── README.md              end-to-end recipe, hyperparameters, measured results
├── scripts/
│   ├── create_env.sh      one-time: build the Python environment
│   ├── setup_env.sh       per-shell: load modules and activate
│   ├── download_assets.sh fetch models and datasets
│   └── configure_ccl.sh   oneCCL / libfabric tuning for this launch shape
├── submit_grpo.sh         qsub wrapper
├── run_grpo.sh            the launcher that runs on the compute nodes
├── tools/                 read a finished run: metrics, scaling, fits
└── diagnostics/           poke the machine directly, without a run
```

Not every example has every entry; this is the naming convention, not a
required set. A file's directory says when it executes: `scripts/` is sourced
by the launcher, `tools/` is run by hand on the log afterwards. `diagnostics/`
holds reproducers for defects that are still open — they are kept because the
bug is, not for history.

## Configuration

Nothing site- or user-specific is hardcoded. Every script reads its
configuration from the environment, with defaults derived from `$HOME`,
`$USER`, or the host name — so an example runs unmodified for a different user
on a different allocation. Each leaf README documents the variables that
example actually reads, and what they default to there.

## Getting started

Pick a leaf directory and follow its README. They all have the same shape:

```bash
git clone <this repo>
cd <system>/<framework>
bash scripts/create_env.sh          # once
source scripts/setup_env.sh
bash scripts/download_assets.sh
bash submit_grpo.sh
```

The exact commands, the variables worth setting, and what a healthy run looks
like are in that directory's README. Start there rather than here.
