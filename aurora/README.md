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
| Software stack | `module load frameworks` | miniforge3 + wheelforge conda env |

The software-stack row is the one asymmetry that matters. `module load
frameworks` does nothing on Sunspot, so `scripts/setup_env.sh` sources
`scripts/frameworks_sunspot.sh` there instead. Override the path with
`FRAMEWORKS_SCRIPT` if your site provides a different environment.

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
