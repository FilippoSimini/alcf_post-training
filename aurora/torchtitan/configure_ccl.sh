#!/bin/bash
# oneCCL, libfabric and Level Zero settings for the torchtitan RL launch shape:
# 4 ranks per node, each pinned to one XPU tile, with vLLM generators and FSDP2
# trainers on separate node groups talking over CXI.
#
# This differs from the TRL example's configure_ccl.sh on purpose. That one runs
# 12 ranks per node over the MPI transport with a flat device hierarchy; here
# each rank needs its own tile, so the flat hierarchy is unset and affinity is
# explicit.

# Per-tile affinity, not the flat hierarchy the frameworks stack defaults to.
unset ZE_FLAT_DEVICE_HIERARCHY
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1,2,3}"
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1

# Level Zero has a race in urEventWait that aborts multi-node runs after a few
# dozen steps. Serializing costs roughly 40% throughput but is the difference
# between finishing and crashing.
export UR_L0_SERIALIZE="${UR_L0_SERIALIZE:-2}"

# oneCCL over libfabric/CXI.
export CCL_OFI_LIBRARY_PATH=/opt/cray/libfabric/2.3.1/lib64/libfabric.so.1
export FI_PROVIDER=cxi
export FI_PROVIDER_PATH=/opt/cray/libfabric/2.3.1/lib64/libfabric
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_OFI_PROVIDER=cxi
export FI_CXI_DEFAULT_CQ_SIZE=1048576
# OFLOW, not OVFLOW. The torchtitan_xpu script this came from spells it
# FI_CXI_OVFLOW_BUF_SIZE, which is not a libfabric variable (check with
# `fi_info -e`), so the setting was silently ignored.
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=30

# Needed beyond ~16 nodes. The default hardware matching mode runs out of
# on-NIC match entries once there are enough peers, and the symptom is
# `cxil_map: write error` followed by
# `atl_ofi.cpp:483 send: fi_tsendmsg ... ret: -14, strerror: Bad address`
# thrown out of the first loss.backward(). Hybrid falls back to software
# matching instead of failing. Likewise the memory-registration cache monitors
# have to be off or large FSDP reduce-scatters trip them.
# These four come from post-training's configure_ccl.sh, added there for
# exactly this failure on >16-node SFT runs.
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_MR_ZE_CACHE_MONITOR_ENABLED=0
export FI_MR_CACHE_MONITOR=disabled
export PALS_PING_PERIOD=240
export PALS_RPC_TIMEOUT=240
export CCL_ALLREDUCE_SCALEOUT=direct
export CCL_BCAST=double_tree
export CCL_SYCL_SCALEOUT_HOST_BUF_SIZE=$((2 * 1024 * 1024 * 1024))
export CCL_OP_SYNC=1
export CCL_WORKER_AFFINITY="5,13,21,29,37,45,57,65,73,81,89,97"
export TORCH_LLM_ALLREDUCE=1
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-hsn0}"

# TorchStore's XCCL transport corrupts weight-sync payloads across nodes:
# tensors arrive zeroed or half-written, so reward stays identically 0 while the
# run looks healthy. Gloo is byte-exact and measured faster anyway
# (1.06 vs 0.90 GB/s). Set to 1 only to reproduce the bug.
export TORCHSTORE_XCCL_ENABLED="${TORCHSTORE_XCCL_ENABLED:-0}"

# PALS --envall forwards the head node's environment verbatim, so anything read
# from the environment rather than from a syscall has to be fixed up first.
# PBS sets TMPDIR to a per-job directory that exists only on the head node, and
# TorchStore reads HOSTNAME to decide shared-memory locality — leaving it set
# makes every rank think it is local to the storage volume.
export TMPDIR=/tmp
unset HOSTNAME

export TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}/torchinductor_xpu"
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export MONARCH_ACTOR_QUEUE_DISPATCH="${MONARCH_ACTOR_QUEUE_DISPATCH:-1}"
