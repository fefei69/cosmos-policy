#!/usr/bin/env bash
#SBATCH --job-name=cosmos-hanoi-joint
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=data/hanoi_cosmos/logs/cosmos-hanoi-joint-%j.out
#SBATCH --error=data/hanoi_cosmos/logs/cosmos-hanoi-joint-%j.err
set -euo pipefail
source examples/hanoi/env.sh
export COSMOS_POLICY_PLATFORM=hanoi_joint
export HANOI_JOINT_RUN_NAME="${HANOI_JOINT_RUN_NAME:-hanoi_cosmos_joint_sparse_20260917}"
export HANOI_JOINT_WANDB_MODE="${HANOI_JOINT_WANDB_MODE:-online}"
export WANDB_MODE="$HANOI_JOINT_WANDB_MODE"
export WANDB_ENTITY="${WANDB_ENTITY:-cw5167-nyu}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$HANOI_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
HANOI_JOINT_TMP=$(mktemp -d "/tmp/cosmos-joint-${SLURM_JOB_ID}.XXXXXX")
export TMPDIR="$HANOI_JOINT_TMP"
trap 'rmdir "$HANOI_JOINT_TMP" 2>/dev/null || true' EXIT
"$HANOI_PROJECT_ROOT/.venv/bin/python" examples/hanoi/run_joint.py "$@"
