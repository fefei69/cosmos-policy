#!/usr/bin/env bash
#SBATCH --job-name=cosmos-hanoi-waypoint
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=data/hanoi_cosmos/logs/cosmos-hanoi-waypoint-%j.out
#SBATCH --error=data/hanoi_cosmos/logs/cosmos-hanoi-waypoint-%j.err
# waypoint_v4 training on one H200 (or H100). The H100 pool is capped by a
# 60-GPU group limit; the H200 partitions had idle nodes when this was written.
set -euo pipefail
source examples/hanoi/env.sh
export COSMOS_POLICY_PLATFORM=hanoi_joint
export HANOI_WAYPOINT_RUN_NAME="${HANOI_WAYPOINT_RUN_NAME:-hanoi_cosmos_waypoint_v4_20260917}"
export HANOI_WAYPOINT_MICROBATCH="${HANOI_WAYPOINT_MICROBATCH:-16}"
export HANOI_WAYPOINT_ACTIVATION_CHECKPOINT="${HANOI_WAYPOINT_ACTIVATION_CHECKPOINT:-none}"
export HANOI_WAYPOINT_WANDB_MODE="${HANOI_WAYPOINT_WANDB_MODE:-online}"
export WANDB_MODE="$HANOI_WAYPOINT_WANDB_MODE"
export WANDB_ENTITY="${WANDB_ENTITY:-cw5167-nyu}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$HANOI_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# The launcher stops before the real Slurm limit, whatever --time was used.
# A controller RPC timeout must not kill the job at start: fall back to --time above.
limit=$(squeue -h -j "$SLURM_JOB_ID" -o '%l' 2>/dev/null || true)
if [[ -z "$limit" || "$limit" == *"NOT_SET"* || "$limit" == "UNLIMITED" ]]; then limit=10:00:00; echo "time limit lookup failed; assuming $limit"; fi
days=0; clock=$limit
if [[ "$limit" == *-* ]]; then days=${limit%%-*}; clock=${limit#*-}; fi
IFS=: read -r h m s <<<"$clock"
if [[ -z "${s:-}" ]]; then s=$m; m=$h; h=0; fi
export HANOI_WAYPOINT_ALLOCATION_SECONDS=$(( days * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
echo "allocation seconds: $HANOI_WAYPOINT_ALLOCATION_SECONDS (limit $limit)"
TMP=$(mktemp -d "/tmp/cosmos-waypoint-${SLURM_JOB_ID}.XXXXXX"); export TMPDIR="$TMP"
trap 'rm -rf "$TMP" 2>/dev/null || true' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv || true
"$HANOI_PROJECT_ROOT/.venv/bin/python" examples/hanoi/run_waypoint.py "$@"
