#!/usr/bin/env bash
#SBATCH --job-name=cosmos-joint-4h100-test
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=data/hanoi_cosmos/logs/cosmos-joint-multigpu-%j.out
#SBATCH --error=data/hanoi_cosmos/logs/cosmos-joint-multigpu-%j.err
set -euo pipefail
source examples/hanoi/env.sh
export COSMOS_POLICY_PLATFORM=hanoi_joint
export HANOI_JOINT_WANDB_MODE=disabled WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$HANOI_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
HANOI_MULTIGPU_TMP=$(mktemp -d "/tmp/cosmos-multigpu-${SLURM_JOB_ID}.XXXXXX")
export TMPDIR="$HANOI_MULTIGPU_TMP"
trap 'rmdir "$HANOI_MULTIGPU_TMP" 2>/dev/null || true' EXIT
"$HANOI_PROJECT_ROOT/.venv/bin/python" examples/hanoi/qualify_joint_multigpu.py "$@"
