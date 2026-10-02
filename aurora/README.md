# Aurora and Sunspot

Both machines use Intel Data Center GPU Max 1550 (Ponte Vecchio). Sunspot is
Aurora's test and development system and runs the same software stack, so the
scripts here work on either — they detect the host and adjust.

| | Aurora | Sunspot |
|---|---|---|
| Nodes | 10 624 | 128 |
| GPUs per node | 6 (12 tiles) | 6 (12 tiles) |
| HBM per tile | 64 GB | 64 GB |
| Project filesystem | `flare` | `tegu` |
| Default queue | `next-eval` | `workq` |
| Software stack | `module load frameworks` | `module load frameworks` |

Both systems resolve `frameworks` to the same stack —
`aurora_frameworks-2026.1.0`, torch 2.13.0a0, vllm 0.26.1 — so a venv built by
`scripts/create_env.sh` is comparable across them.

This was not always true. Sunspot's default used to be `frameworks/2025.3.1`,
whose module has a broken dependency on `intel_gpu_umd_aicoe`; the workaround
was to load the dependencies by hand and `conda activate` the env directly,
which is what `scripts/frameworks_sunspot.sh` still does. It is kept for that
older stack and for sites that provide something different — select it with
`FRAMEWORKS_SCRIPT=scripts/frameworks_sunspot.sh`. When `FRAMEWORKS_SCRIPT` is
unset, both systems just `module load frameworks`.

## Device layout

A node exposes 6 GPUs, each with 2 tiles. With
`ZE_FLAT_DEVICE_HIERARCHY=FLAT` the 12 tiles appear as 12 independent devices,
which is what the TRL example uses (one MPI rank per tile). The torchtitan
example needs per-tile affinity instead and unsets it, running 4 ranks per node
with `ZE_AFFINITY_MASK`.

## Examples

- [trl/](trl/) — GRPO on the Countdown game with TRL and FSDP2.
- [torchtitan/](torchtitan/) — GRPO + LoRA on alphabet sort with torchtitan's
  `experiments/rl`, using Monarch, TorchStore, and vLLM.
