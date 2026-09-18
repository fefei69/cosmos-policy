#!/usr/bin/env bash
#SBATCH --job-name=cosmos-4h200-check
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=data/hanoi_cosmos/logs/%x-%j.out
#SBATCH --error=data/hanoi_cosmos/logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this diagnostic from the project root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
H200_CHECK_TMP=$(mktemp -d "/tmp/cosmos-4h200-${SLURM_JOB_ID}.XXXXXX")
export TMPDIR="$H200_CHECK_TMP"
trap 'rmdir "$H200_CHECK_TMP" 2>/dev/null || true' EXIT
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
# The command timeout is shorter than Slurm's limit and has no retries.
timeout --signal=TERM --kill-after=15s 420s \
    .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 --max_restarts=0 \
    examples/hanoi/check_four_h200.py \
    --output "data/hanoi_cosmos/operations/four_h200_check_${SLURM_JOB_ID}.json"
