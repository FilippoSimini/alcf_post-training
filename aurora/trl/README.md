# GRPO with TRL on Aurora / Sunspot

Trains a Qwen3 model to solve **Countdown** puzzles with Group Relative Policy
Optimization. Given a target number and a list of numbers, the model must write
an arithmetic expression that equals the target:

```
Using the numbers [10, 5, 3, 2], create an equation that equals 42.
→ <think> 10 * 5 = 50, 50 - 5 = 45 ... </think>
  <answer> (10 - 3) * (5 + 2) - 7 </answer>
```

One reward function scores each completion — 1.0 for a correct expression using
exactly the given numbers, 0.1 for well-formed but wrong, 0.0 for no `<answer>`
tags. GRPO samples several completions per prompt, ranks them by reward within
the group, and pushes the policy toward the above-average ones. No reward model
and no reference model are needed, which is what makes it cheap enough to run
on one node.

## Files

| File | Purpose |
|---|---|
| `scripts/create_env.sh` | One-time: build the venv on top of the system stack |
| `scripts/setup_env.sh` | Per-shell: load modules, activate the venv, set defaults |
| `scripts/hf_env.sh` | Where models live and which model to train |
| `scripts/download_assets.sh` | Fetch the model into `$DATA_MODEL_PATH` |
| `scripts/frameworks_sunspot.sh` | Optional: the older Sunspot stack (miniforge3 + conda), via `FRAMEWORKS_SCRIPT` |
| `configure_ccl.sh` | oneCCL / libfabric tuning, sourced by `run_grpo.sh` |
| `submit_grpo.sh` | `qsub` wrapper |
| `run_grpo.sh` | Launcher: `mpiexec`, one rank per XPU tile |
| `train_grpo.py` | The training script |
| `countdown_rewards.py` | Reward function |
| `generate_countdown.py` | Write a Countdown dataset to JSONL (optional) |
| `generate_hf.py` | Run puzzles through a checkpoint to see what it learned |

## Run it

### 1. Build the environment (once)

```bash
cd alcf_post-training/aurora/trl
bash scripts/create_env.sh
```

### 2. Fetch the model

`download_assets.sh` needs outbound network, so run it on a login node.

```bash
source scripts/setup_env.sh
bash scripts/download_assets.sh
```

If your home directory is small, point the caches at a project filesystem
first — these must then be set for every later command too:

```bash
export HF_HOME=/lus/<fs>/projects/<project>/$USER/huggingface/cache
export DATA_MODEL_PATH=/lus/<fs>/projects/<project>/$USER/huggingface
```

### 3. Train

Batch:

```bash
PBS_ACCOUNT=<allocation> bash submit_grpo.sh
```

Interactive, which is easier to debug:

```bash
qsub -I -l select=1 -l walltime=01:00:00 -l filesystems=home:tegu \
     -A <allocation> -q workq
cd alcf_post-training/aurora/trl
bash run_grpo.sh
```

### 4. Watch it

```bash
qstat -u $USER
tail -f $(ls -t runlogs/run_grpo_*.log | head -1)
```

`reward` should climb and `grad_norm` should stay finite. A run that logs
`loss: 0.0` and `grad_norm: nan` every step is not training, even though it
completes.

### 5. See what changed

```bash
source scripts/setup_env.sh
MODEL_PATH=$DATA_MODEL_PATH/$HF_MODEL_REPO python generate_hf.py          # base
MODEL_PATH=./output/Qwen3-0.6B-countdown-grpo/checkpoint-40 python generate_hf.py
```

## Configuration

Every knob is an environment variable. See the
[repository README](../../README.md) for the ones shared across examples;
these are specific to this one:

| Variable | Default | Meaning |
|---|---|---|
| `HF_MODEL_REPO` | `Qwen/Qwen3-0.6B` | Model to train |
| `MAX_STEPS` | `40` | Training steps |
| `BATCH_SIZE` | `4` | Prompts per tile per step |
| `LR` | `5e-6` | Learning rate |
| `NUM_SAMPLES` | `50000` | Synthetic Countdown puzzles to generate |
| `NRANKS_PER_NODE` | `12` | MPI ranks per node — one per XPU tile |
| `FSDP_LAYER_CLS` | `Qwen3DecoderLayer` | Decoder block FSDP wraps; must match the model |
| `USE_VLLM` | unset | Set to `1` to generate rollouts with colocated vLLM |
| `OUTDIR` | `./output/<model>-countdown-grpo` | Checkpoints |
| `EXTRA_ARGS` | empty | Passed through to `train_grpo.py` |

Switching model family means switching `FSDP_LAYER_CLS` with it —
`Qwen3DecoderLayer`, `Qwen2DecoderLayer`, `LlamaDecoderLayer`. FSDP wraps by
class name and silently wraps nothing if the name is wrong.

## Hyperparameters

| Parameter | Value | Why |
|---|---|---|
| `num_generations` | 4 | Completions per prompt; the group GRPO ranks |
| `max_completion_length` | 512 | Enough for `<think>` plus an expression |
| `temperature` | 1.0 | Full entropy — low temperature collapses the group |
| `beta` | 0.0 | No KL penalty, so no reference model to hold in memory |
| `loss_type` | `grpo` | Symmetric clipping |
| `num_iterations` | 2 | Optimizer passes per batch of rollouts |
| `epsilon` | 0.2 | PPO clip range |
| `warmup_steps` | 10 | |
| Parallelism | FSDP2 | `full_shard auto_wrap`, activation checkpointing |

## Results

Filled in from measured runs.

## Notes

- **FSDP2, not FSDP1.** FSDP1 is incompatible with colocated vLLM: vLLM's
  `external_launcher` starts a second distributed context in-process and
  corrupts FSDP1's `_is_root` flags.
- **`beta=0.0` removes the reference model.** With a KL penalty you would hold a
  second copy of the model in memory for no benefit on this task.
- **One rank per tile.** A node has 6 GPUs of 2 tiles each; with
  `ZE_FLAT_DEVICE_HIERARCHY=FLAT` they appear as 12 devices.
