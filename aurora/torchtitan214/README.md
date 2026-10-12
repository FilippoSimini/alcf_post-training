# GRPO with torchtitan on the torch 2.14 / oneCCL 2022 stack

Trains Qwen3-0.6B on the **alphabet sort** task — given a list of author names,
emit them in alphabetical order — with a 0/1 reward, in two arms that are
controlled twins: same batch, learning rate, sampling and generator settings,
differing only in whether LoRA adapters or every parameter is trained.

A disaggregated pipeline: trainers hold the policy under FSDP2, generators
run vLLM rollouts, weights and trajectories move between them through
TorchStore, and Monarch supervises both as one actor mesh. The software stack
— torch 2.14.0+xpu, oneCCL 2022.1.1, oneAPI 2026.1 and vLLM from `main` — is
built here into a self-contained conda environment rather than taken from a
system module, so a run depends on nothing outside this directory but the
compiler and conda.

```
 trainer nodes                        generator nodes
 ┌──────────────┐   TorchStore   ┌──────────────────┐
 │ FSDP2 policy │ ─── weights ──▶│ vLLM rollouts    │
 │   update     │◀── trajectories│  + reward        │
 └──────────────┘                └──────────────────┘
        └──────── Monarch actor mesh ────────┘
```

There is a **single-node** entry point: 4 tiles, 2 trainer and 2 generator,
no `mpiexec`. That is the cheapest way to check the pipeline end to end.

## Files

| File | Purpose |
|---|---|
| `scripts/create_env.sh` | One-time: conda env, torch/Triton wheels, vLLM, Monarch (Rust), TorchStore, torchtitan |
| `scripts/setup_env.sh` | Per-shell: oneAPI 2026.1, activate the env, set the site paths |
| `scripts/download_assets.sh` | Fetch model and dataset, warm the offline caches |
| `run_grpo.sh` | Select an arm and exec the upstream launcher |
| `submit_grpo.sh` | `qsub` wrapper |

**The fabric and oneCCL tuning is not in this directory.** It lives in the
launchers inside the torchtitan checkout that `create_env.sh` clones
(`scripts/rl/xpu/`), where every export carries a comment naming the bug it
closes. `run_grpo.sh` sets site paths and execs one of them. A second copy of
that block here would be free to drift from the launcher it belongs to.

## Run it

Every step below sources `scripts/setup_env.sh`, which derives its paths from
the host name. You do not have to set anything for them to work, but three
are worth knowing before step 1, because a run that gets them wrong fails in
a way that does not name the cause.

| Variable | Set to | Why it matters |
|---|---|---|
| `DATA_MODEL_PATH` | `$HOME/huggingface`, with a notice on stderr | Where models and datasets are downloaded to and read from. `/lus/<fs>/projects` has one directory per allocation, so there is no default that suits everyone — set this to e.g. `/lus/<fs>/projects/<project>/$USER/huggingface`. Home is usually a small quota and the model alone is 1.5 GB. |
| `HF_HOME` | `$DATA_MODEL_PATH/cache` | The launchers export `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` but never set `HF_HOME`, so without this they fall back to `~/.cache/huggingface` and every run dies offline looking for a dataset that is already on disk. |
| `TORCHINDUCTOR_CACHE_DIR` | `/tmp/$USER/torchinductor_xpu` | Node-local on purpose. The launchers would default it to `~/.cache/torchinductor_xpu`; where home is shared, one Inductor cache written concurrently by every node is slow and a source of corrupt-cache failures. Do not point it at a shared path. |

Override any of them by exporting before you run.

### 1. Build the environment (once, on a login node)

About 35 minutes, measured on Aurora, most of it compiling Monarch's Rust
extension. Needs outbound network, so a login node and never a compute node.

Detach it — a dropped connection kills a foreground build and there is no
resume, you start over:

```bash
cd alcf_post-training/aurora/torchtitan214
nohup bash scripts/create_env.sh > create_env.log 2>&1 &
tail -f create_env.log        # done at "===== Environment ready."
```

This builds a conda environment at `conda_env/` and clones four repositories
into `venv_repos/`:

| Repo | Ref | What it is |
|---|---|---|
| `songhappy/torchtitan` | `xpu/rl-rebase-20261001` | the recipes, the launchers and the RL pipeline |
| `songhappy/torchstore` | `1ebced4` | XCCL transport and replica dedup |
| `songhappy/monarch` | `7761764a` + `d9fc48862` | device-agnostic actor runtime; the cherry-pick makes `default_bootstrap_cmd` public, which torchtitan imports |
| `vllm-project/vllm` | `083060d04` | upstream `main`, no local patches |

Three of the four are fixed commits. **torchtitan is a branch**, so
`create_env.sh` takes whatever is at its head — the recipe follows the fork
forward, and two builds a few days apart are not the same build. Pass
`TORCHTITAN_REF=<sha>` to pin it.

Every run prints the four commits it used on its `=== revs:` line; quote that
alongside any result. `venv_repos/PINNED.txt` records the installed Python
package versions at build time.

torch, Triton and the vLLM XPU kernels are not pinned here. vLLM's own
`requirements/xpu.txt` at the pinned commit is the authority for that matching
set, so it is installed from there.

### 2. Fetch the model and dataset

Login node, outbound network.

```bash
source scripts/setup_env.sh
bash scripts/download_assets.sh
```

### 3. Train

```bash
# single node, 4 tiles: the cheapest end-to-end check
PBS_ACCOUNT=<allocation> ARM=lora SHAPE=sn NUM_STEPS=10 bash submit_grpo.sh
PBS_ACCOUNT=<allocation> ARM=full SHAPE=sn NUM_STEPS=10 bash submit_grpo.sh

# multi-node: half the allocation trains, half generates
PBS_ACCOUNT=<allocation> ARM=lora SHAPE=mn NNODES=4 NUM_STEPS=10 bash submit_grpo.sh
```

Interactive. The queue and filesystem differ per host — `debug` and
`home:flare` on Aurora, `workq` and `home:tegu` on Sunspot. Aurora's `debug`
caps at 2 nodes and one hour:

```bash
qsub -I -l select=2 -l walltime=01:00:00 -l filesystems=home:flare \
     -A <allocation> -q debug
cd alcf_post-training/aurora/torchtitan214
ARM=lora SHAPE=mn bash run_grpo.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `ARM` | `lora` | `lora` (adapters, rank 32) or `full` (every parameter) |
| `SHAPE` | `sn` | `sn` (1 node, 4 tiles) or `mn` (multi-node) |
| `NNODES` | `4` for `mn` | Nodes to request |
| `NUM_STEPS` | `10` | Training steps |
| `VAL_SAMPLES` | `32` | Validation samples |
| `PPN` | `4` | Tiles per node |
| `TP` | `1` | Tensor parallel degree |
| `DP_REPLICATE` | `1` | Data parallel replicate degree |
| `CONFIG` | per arm | Entry in `torchtitan_recipes.rl.alphabet_sort_xpu` |
| `TAG` | empty | Distinguishes arms sharing one allocation |
| `WALLTIME` | `01:00:00` | |
| `PBS_ACCOUNT` | *(none — required)* | Allocation to charge |

Everything else — the HuggingFace paths, the queue, the filesystem list — is
derived from the host name in `scripts/setup_env.sh`. Override any of them by
exporting before submitting.

### 4. Check that it is actually learning

A run can complete every step and have learned nothing, so check the numbers,
not the exit code. A run is healthy when all of these hold:

| What | Healthy |
|---|---|
| step lines | every requested step logs `Train | Step: N` |
| reward | non-zero from step 1, and no `untrainable batches` warning |
| LoRA `grad_norm` | 0.02 – 0.15 |
| full-parameter `grad_norm` | several times higher, because it is taken over all parameters — measured 0.17–0.73 against a LoRA run's 0.023–0.071. Compare it only with other full runs |
| mesh | the `Mesh split` line shows the mesh you expect |

Read the mesh from that log line, not from the flags you passed: the launcher
derives `dp_shard` from the node count and the LoRA rank, so what you asked for
and what you got can differ.

The log is at `venv_repos/torchtitan/<dump folder>/`, printed in the run header.

### 5. When the steps are done

A completed run exits on its own with status 0. Measured on five runs across
both machines — 1 node `PPN=4`, 2 nodes `PPN=12`, 2 nodes `PPN=12` with
`DP_REPLICATE=2`, 4 nodes `PPN=4` with `DP_REPLICATE=2` — each finishing in
10–19 minutes of a one-hour walltime, none needing `qdel`.

Earlier builds of this stack hung after the last step and had to be killed.
If a run logs its final validation and then sits there, `qdel <jobid>`:
nothing is lost, metrics are written before teardown starts, and PBS will
report `Exit code: 143` for work that succeeded.

## Known issues

Open. The first two were measured on the runs in *Results*; the rest are
documented by the upstream branch this stack is built from and were not
contradicted here.

- **4 nodes at `PPN=12` does not train.** The mesh builds and all generators
  come up, then the run stops in the first weight transfer and never writes a
  `Train | Step` line. `PPN=12` at 2 nodes, and `PPN=4` up to 8 nodes, are
  unaffected.
- **Host-mesh attach timeout**: `attach failed: config push failed during
  attach on N host(s) ... MESH_ATTACH_CONFIG_TIMEOUT`, before any actor work.
  Monarch's `HostMesh::attach()` pushes config to every host and waits on a
  barrier, failing closed after `MESH_ATTACH_CONFIG_TIMEOUT` — **60 s** by
  default, which has been measured failing 3 of 5 submissions in a day.
  `scripts/setup_env.sh` therefore raises it to `300s` via
  `HYPERACTOR_MESH_ATTACH_CONFIG_TIMEOUT`. Override that if you still see it;
  the value is parsed by `humantime`, so give it a unit (`300s`, not `300`).
  Waiting longer costs nothing when the barrier completes normally.
- **Monarch startup flake**: `cast.service ... actor does not exist` before any
  step. Intermittent — resubmit rather than debug.
- **vLLM "not enough free memory" at startup with 0 steps** is a phantom from
  Level Zero's lagging memory report. Resubmit before changing the memory limit.
- **Inductor cache corruption** (`CompiledFxGraph has no attribute
  compiled_fn_runner`): `rm -rf $TORCHINDUCTOR_CACHE_DIR`.
- **Checkpoint saving is off** (`load_only=True` in the recipe): the DCP save
  ran out of device memory inside oneCCL at 8 nodes. Never disable the
  checkpointer to avoid saves — `enable` also gates loading the pretrained
  weights, so the run would silently train from random init.

## Results

Qwen3-0.6B, alphabet sort, 10 steps, 32 validation samples. Ten steps is a
plumbing check, not a convergence measurement — reward over so few steps moves
mostly with sampling noise, so these establish that the pipeline runs, nothing
more.

| machine | nodes | `PPN` | arm | mesh | steps | rollout reward |
|---|---|---|---|---|---|---|
| Aurora | 1 | 4 | lora | `dp_shard=4` | 10/10 | 0.47 → 0.15 |
| Aurora | 2 | 12 | full | `dp_shard=12` | 10/10 | 0.46 → 0.23 |
| Aurora | 2 | 12 | full | `dp_shard=6 × rep=2` | 10/10 | — |
| Sunspot | 2 | 12 | full | `dp_shard=12` | 10/10 | — |
| Sunspot | 4 | 4 | full | `dp_shard=4 × rep=2` | 10/10 | — |

LoRA `grad_norm` ran 0.023–0.071 and full-parameter 0.17–0.73, the expected
several-fold gap since the latter is taken over every parameter. The two
machines behaved the same at the one shape run on both (2 nodes, `PPN=12`,
`dp_shard=12`).
