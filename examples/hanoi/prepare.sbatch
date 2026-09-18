#!/usr/bin/env bash
#SBATCH --job-name=cosmos-hanoi-prepare
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=cosmos-hanoi-prepare-%j.out
#SBATCH --error=cosmos-hanoi-prepare-%j.err

set -euo pipefail
source examples/hanoi/env.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$HANOI_PROJECT_ROOT/.venv/bin/python" "$@"
