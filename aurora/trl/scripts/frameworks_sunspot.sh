#!/bin/bash
# Sunspot software stack: miniforge3 plus a prebuilt conda environment with
# PyTorch-XPU, vLLM and the Intel GPU runtime. `module load frameworks` is a
# no-op on Sunspot, which is why this exists.
#
# Override CONDA_ENV_PATH to use a different prebuilt environment.

module add miniforge3/25.11.0-1
module add intel_gpu_umd_aicoe/2026.06.19
module add cmake
unset CMAKE_ROOT
module add ninja
module add pti-gpu
module add hdf5

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_OP_SYNC=1
export CCL_ATL_SYNC_COLL=1

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/lus/tegu/projects/datasets/software/26.181.0/wheelforge/envs/conda_envs/RC4_vllm_xpu_kernels_0.1.11.1_icu73_vllm_0.26.0_triton_3.7.2_nre_pt_2.13.0_rel_one_2026.1.0_np_2.3.5_python_3.12.12}"
conda activate "${CONDA_ENV_PATH}"
