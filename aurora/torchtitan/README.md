# GRPO with torchtitan on Aurora / Sunspot

Trains Qwen3-0.6B with LoRA adapters on the **alphabet sort** task using
torchtitan's `experiments/rl`: given a list of author names, emit them in
alphabetical order. The reward is a graded `difflib` similarity against the
sorted list, not a 0/1 check.

Unlike the [TRL example](../trl/), which runs generation inside the training
process, this is a disaggregated pipeline:

```
 trainer nodes                        generator nodes
 ┌──────────────┐   TorchStore   ┌──────────────────┐
 │ FSDP2 policy │ ─── weights ──▶│ vLLM rollouts    │
 │   update     │◀── trajectories│  + reward        │
 └──────────────┘                └──────────────────┘
        └──────── Monarch actor mesh ────────┘
```

The allocation is split in half — the first nodes train, the rest generate.
That costs more nodes for a given model but keeps both sides busy, and it is
the shape that scales.

## Files

| File | Purpose |
|---|---|
| `scripts/create_env.sh` | One-time: venv, torchstore, Monarch (Rust), torchtitan |
| `scripts/setup_env.sh` | Per-shell: load modules, activate the venv, set defaults |
| `scripts/hf_env.sh` | Where models/datasets live and which to use |
| `scripts/download_assets.sh` | Fetch model and dataset, warm the datasets cache |
| `scripts/frameworks_sunspot.sh` | Sunspot software stack (miniforge3 + conda env) |
| `configure_ccl.sh` | oneCCL / libfabric / Level Zero settings |
| `submit_grpo.sh` | `qsub` wrapper |
| `run_grpo.sh` | Launcher: `mpiexec -ppn 1`, Monarch spawns the actors |
| `multinode_launcher.py` | Splits nodes into trainer and generator meshes |
| `configs/alcf_rl/` | Extra GRPO configs, kept out of the torchtitan checkout |

## Run it

### 1. Build the environment (once)

30-45 minutes, most of it compiling Monarch's Rust extension.

```bash
cd alcf_post-training/aurora/torchtitan
bash scripts/create_env.sh
```

This clones three forks. XPU support is not upstream yet:

| Repo | Branch | What the fork adds |
|---|---|---|
| `songhappy/torchstore` | `rl` | XCCL transport and XPU fixes |
| `songhappy/monarch` | `xpu-upstream` | Device-agnostic actor runtime (XPU as well as CUDA) |
| `songhappy/torchtitan` | `rl` | XPU paths in `experiments/rl` |

All three are installed with `pip install -e .`, so a moving branch changes
what runs. The script therefore writes the resolved revisions to
`venv_repos/PINNED.txt`:

```
torchstore   rl             0a4c18c584491a913d6316d46f600c60b928a120
monarch      xpu-upstream   7761764ae46a330d13de5ea878dd1419ae42546f
torchtitan   rl             a55da6b49dd13fb52879a70dd351f6e7468a71df
```

Quote that file alongside any result. To rebuild the same thing later, set the
refs to those SHAs:

```bash
TORCHSTORE_REF=0a4c18c5 MONARCH_REF=7761764a TORCHTITAN_REF=a55da6b4 \
    bash scripts/create_env.sh
```

### 2. Fetch the model and dataset

Needs outbound network, so run it on a login node.

```bash
source scripts/setup_env.sh
bash scripts/download_assets.sh
```

### 3. Train

```bash
PBS_ACCOUNT=<allocation> NNODES=2 bash submit_grpo.sh
```

Interactive:

```bash
qsub -I -l select=2 -l walltime=01:30:00 -l filesystems=home:tegu \
     -A <allocation> -q workq
cd alcf_post-training/aurora/torchtitan
bash run_grpo.sh
```

**Two nodes is the minimum.** `multinode_launcher.py` splits the allocation
into a trainer half and a generator half, and on a single host that slice is
empty — the controller dies with
`ValueError: out of range 1:1:1 for dimension hosts of size 1`. Use an even
node count so the split is balanced. A single-node, trainer-and-generator-
colocated run is a different entry point that bypasses the launcher entirely
and is not wired up here yet.

### 4. Check that it is actually learning

```bash
tail -f $(ls -t runlogs/run_grpo_*.log | head -1)
```

A run can complete all its steps and still have learned nothing. Check three
numbers, not the exit code:

| Metric | Healthy |
|---|---|
| `rollout_reward/_mean` | non-zero and rising |
| `trainer/grad_norm/mean` | 0.040 – 0.150 |
| `loss/mean` | −0.003 to −0.0093 |

All three sitting at exactly `0` means weight sync is broken — the generator is
sampling from an unchanged or zeroed policy, every completion in a group scores
the same, the GRPO advantage is identically zero, and the run logs healthy
looking steps forever.

### 5. Kill the job when the steps are done

**The job will not exit on its own.** After the last step is logged and the
generators log their shutdown, Monarch/PALS teardown hangs — observed from 25
minutes to the full walltime. The `exit 42` sentinel in `run_grpo.sh` does not
help, because the launcher process itself never returns.

Nothing is lost: metrics and checkpoints are written before teardown starts.
Watch for the final step in the log, then:

```bash
qdel <jobid>
```

If you run several configurations back to back inside one allocation, cap each
one and clean up after it, or the first hang will eat the whole walltime:

```bash
timeout -k 30 1500 bash run_grpo.sh

# Stale Monarch workers silently corrupt the next run, so this is not optional.
mpiexec -n "$NNODES" -ppn 1 --hosts "$HOSTS" --cpu-bind none \
    pkill -9 -f 'multinode_launcher|torchtitan.experiments.rl|monarch|VLLM'
```

Budget walltime on the timeout rather than the useful runtime. A 4-node,
10-step run needs roughly 17 minutes of real work: ~8 minutes of vLLM SYCL
compilation and engine startup, ~5 minutes of training, ~4 minutes of shutdown
logging.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODULE` | `alcf_rl` | Config registry module; `alphabet_sort` for the upstream one |
| `CONFIG` | `rl_grpo_lora_qwen3_0_6b_lr2e5` | Entry in that registry |
| `NUM_NODES` | all allocated | Nodes to use, split half trainer / half generator |
| `PPN` | `4` | Ranks (tiles) per node |
| `NUM_STEPS` | `10` | Training steps |
| `VAL_SAMPLES` | `32` | Pre-training validation samples |
| `TP` | `1` | Tensor parallel degree |
| `DP_REPLICATE` | `1` | Data parallel replicate degree |
| `LR` | `2e-5` | Learning rate |
| `TORCHSTORE_XCCL_ENABLED` | `0` | Weight-sync transport; see below |
| `DUMP_FOLDER` | `outputs/rl_grpo` | Relative to the torchtitan checkout |
| `EXTRA_ARGS` | empty | Passed through to `torchtitan.experiments.rl.train` |

### Configs live outside the torchtitan checkout

torchtitan's `--module` accepts a fully qualified module path, not just the
built-in shorthands. `configs/alcf_rl/config_registry.py` re-exports everything
from the upstream `alphabet_sort` registry and adds a couple of entries, and
`run_grpo.sh` puts `configs/` on `PYTHONPATH`. Nothing is patched into the
installed source tree, so `create_env.sh` produces a clean checkout and
upstream can be updated without conflicts.

| Config | What it changes |
|---|---|
| `rl_grpo_lora_qwen3_0_6b_lr2e5` | Qwen3-0.6B at `lr=2e-5` |
| `rl_grpo_lora_qwen3_4b` | Qwen3-4B at `lr=2e-5` |

To run an upstream config unchanged: `MODULE=alphabet_sort CONFIG=... bash run_grpo.sh`.

### Two defaults that differ from upstream

**`lr=2e-5`, not the shipped `2e-6`.** The learning rate in
`rl_grpo_lora_qwen3_0_6b` was carried over from a non-LoRA parallelism sweep
where it was never meant to converge; every other LoRA config upstream uses
`1e-4` or `2e-5`. At `2e-6` the adapters barely move and rollout reward drifts
*down* — confirmed here at 1, 2 and 8 nodes. There is no CLI override for it
(`OptimizersContainer.Config` exposes `param_groups`, not a flat `lr`), which
is why this is a config entry rather than a flag.

**`TORCHSTORE_XCCL_ENABLED=0`, falling back to Gloo.** TorchStore's XCCL
transport corrupts weight-sync payloads once storage volumes span more than one
host — tensors arrive zeroed or half-written. Gloo is byte-exact and measured
*faster* on this fabric (1.06 vs 0.90 GB/s), so there is no cost to the
workaround. Set to `1` only to reproduce the bug.

### Other settings worth knowing about

- `UR_L0_SERIALIZE=2` works around a Level Zero race in `urEventWait` that
  aborts multi-node runs after a few dozen steps. It costs roughly 40%
  throughput; without it the run crashes instead.
- `--cpu-bind none` is required. At `-ppn 1` PALS binds the rank to a single
  core and every Monarch actor and vLLM worker inherits the mask, putting ~126
  threads on 1 core of 208.
- `TMPDIR=/tmp` and `unset HOSTNAME`: PALS `--envall` forwards the head node's
  environment verbatim, and both variables mean something different on the
  other nodes.
- Management hostnames, not `.hsn` names, for `--hosts`. The `.hsn` names are
  multi-rail and cause `MESH_ATTACH_CONFIG_TIMEOUT`.

## Results

Filled in from measured runs.
