#!/usr/bin/env bash
# Source from any directory. This experiment owns all caches and outputs below.
HANOI_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export HANOI_PROJECT_ROOT
export COSMOS_POLICY_PLATFORM=hanoi
export HANOI_DATA_ROOT=/scratch/cw5167/datasets
export HANOI_DIRECTION=aaaa_to_cccc
export HANOI_METADATA_DIR="$HANOI_PROJECT_ROOT/data/hanoi_cosmos/aaaa_to_cccc_pos_only"
export HANOI_T5_EMBEDDINGS="$HANOI_PROJECT_ROOT/data/hanoi_cosmos/t5_embeddings.pkl"
export HF_HOME="$HANOI_PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$HANOI_PROJECT_ROOT/.cache/torch"
export UV_CACHE_DIR="$HANOI_PROJECT_ROOT/.cache/uv"
export XDG_CACHE_HOME="$HANOI_PROJECT_ROOT/.cache"
export TMPDIR="$HANOI_PROJECT_ROOT/.cache/tmp"
export TRITON_CACHE_DIR="$HANOI_PROJECT_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$HANOI_PROJECT_ROOT/.cache/torchinductor"
export IMAGINAIRE_OUTPUT_ROOT="$HANOI_PROJECT_ROOT/data/hanoi_cosmos/runs"
export WANDB_MODE=disabled
export ENABLE_ONELOGGER=False
export NUMPY_MADVISE_HUGEPAGE=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=1
# Transformer Engine 2.2 needs an explicit NVRTC location outside the CUDA container.
# Use the CUDA 12.8 runtime wheel pinned by this project's Python 3.10 environment.
HANOI_NVRTC_ROOT="$HANOI_PROJECT_ROOT/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
if [[ -f "$HANOI_NVRTC_ROOT/lib/libnvrtc.so.12" ]]; then
    export CUDA_HOME="$HANOI_NVRTC_ROOT"
    export LD_LIBRARY_PATH="$HANOI_NVRTC_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$IMAGINAIRE_OUTPUT_ROOT"
