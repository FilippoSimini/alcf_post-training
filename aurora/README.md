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
| Default queues | `debug`, `debug-scaling` | `workq` |
| System software stack | `module load frameworks` | `module load frameworks` |


An example may create a virtual environment on top of the `frameworks` software stack, anther example may instead build its own self-contained environment and not use `frameworks` at all; its README says
which.


## Device layout

A node exposes 6 GPUs, each with 2 tiles. Examples here take one of two
approaches. With `ZE_FLAT_DEVICE_HIERARCHY=FLAT` the 12 tiles appear as 12
independent devices, which suits one MPI rank per tile. An example that needs
per-tile affinity instead unsets it and selects tiles with `ZE_AFFINITY_MASK`,
running fewer ranks per node. Which one applies is in each example's README.
