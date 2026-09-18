#!/usr/bin/env bash
#SBATCH --job-name=cosmos-hanoi-long
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --output=cosmos-hanoi-long-%j.out
#SBATCH --error=cosmos-hanoi-long-%j.err

set -euo pipefail
source examples/hanoi/env.sh
: "${HANOI_RUN_NAME:?Set the verified run to resume}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Use node-local temporary files; avoid NFS multiprocessing cleanup errors.
HANOI_LOCAL_TMP=$(mktemp -d "/tmp/cosmos-hanoi-${SLURM_JOB_ID}.XXXXXX")
export TMPDIR="$HANOI_LOCAL_TMP"
trap 'rmdir "$HANOI_LOCAL_TMP" 2>/dev/null || true' EXIT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$HANOI_PROJECT_ROOT/.venv/bin/python" examples/hanoi/run_long.py
