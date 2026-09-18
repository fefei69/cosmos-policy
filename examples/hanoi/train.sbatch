#!/usr/bin/env bash
#SBATCH --job-name=cosmos-hanoi-pilot
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=cosmos-hanoi-%j.out
#SBATCH --error=cosmos-hanoi-%j.err

set -euo pipefail
source examples/hanoi/env.sh
: "${HANOI_RUN_NAME:?Set a unique HANOI_RUN_NAME before submitting. Reuse only to resume that experiment.}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Finish at an optimizer boundary with ten minutes left for the final checkpoint.
# Override this for a different allocation length; default matches the 2h pilot.
export HANOI_STOP_AT_EPOCH
HANOI_STOP_AT_EPOCH="$("$HANOI_PROJECT_ROOT/.venv/bin/python" -c 'import os,time; print(time.time()+int(os.environ.get("HANOI_TRAIN_SECONDS", "6600")))')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
exec "$HANOI_PROJECT_ROOT/.venv/bin/python" examples/hanoi/run_pilot.py "$@"
