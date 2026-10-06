#!/usr/bin/env bash
#SBATCH --job-name=cosmos-hanoi-h1
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=data/hanoi_cosmos/logs/cosmos-hanoi-h1-%j.out
#SBATCH --error=data/hanoi_cosmos/logs/cosmos-hanoi-h1-%j.err
# hanoi_expert6_h1 training on one H200: the first move of the six tower tasks (48 one-move expert
# episodes), goal-sentence conditioned, on the play pipeline. Every setting that differs from the play
# run is exported here so the run does not depend on the submitting shell:
#   8,000 updates, export every 1,000 (the play and six-task runs needed about 8,000 updates from the
#   video base to get under 3 mm whatever the data size; selection on the held-out episodes guards
#   against memorising the 24,636 training rows), validation every third row per stage, every row for
#   the selected export.
#   sbatch examples/hanoi/train_h1.sbatch --init video
#   sbatch --dependency=afterany:<jobid> examples/hanoi/train_h1.sbatch --init video   # continuation
set -euo pipefail
source examples/hanoi/env.sh
export COSMOS_POLICY_PLATFORM=hanoi_dense
export HANOI_PLAY_METADATA="$HANOI_PROJECT_ROOT/data/hanoi_cosmos/expert6_h1"
export HANOI_PLAY_MAX_UPDATES=8000
export HANOI_PLAY_SAVE_EVERY=1000
export HANOI_PLAY_STAGE_STRIDE=3
export HANOI_PLAY_FINAL_STRIDE=1
export HANOI_PLAY_QUALIFY_STRIDE=60
unset HANOI_PLAY_RUN_NAME
export HANOI_DENSE_MICROBATCH="${HANOI_DENSE_MICROBATCH:-16}"
export HANOI_DENSE_ACTIVATION_CHECKPOINT="${HANOI_DENSE_ACTIVATION_CHECKPOINT:-none}"
export HANOI_DENSE_WANDB_MODE="${HANOI_DENSE_WANDB_MODE:-online}"
export WANDB_MODE="$HANOI_DENSE_WANDB_MODE"
export WANDB_ENTITY="${WANDB_ENTITY:-cw5167-nyu}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$HANOI_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# The launcher stops before the real Slurm limit; a controller RPC timeout must not kill the job at start.
limit=$(squeue -h -j "$SLURM_JOB_ID" -o '%l' 2>/dev/null || true)
if [[ -z "$limit" || "$limit" == *"NOT_SET"* || "$limit" == "UNLIMITED" ]]; then limit=12:00:00; echo "time limit lookup failed; assuming $limit"; fi
days=0; clock=$limit
if [[ "$limit" == *-* ]]; then days=${limit%%-*}; clock=${limit#*-}; fi
IFS=: read -r h m s <<<"$clock"
if [[ -z "${s:-}" ]]; then s=$m; m=$h; h=0; fi
export HANOI_DENSE_ALLOCATION_SECONDS=$(( days * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
echo "allocation seconds: $HANOI_DENSE_ALLOCATION_SECONDS (limit $limit)"
TMP=$(mktemp -d "/tmp/cosmos-h1-${SLURM_JOB_ID}.XXXXXX"); export TMPDIR="$TMP"
trap 'rm -rf "$TMP" 2>/dev/null || true' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv || true
"$HANOI_PROJECT_ROOT/.venv/bin/python" examples/hanoi/run_play.py "$@"
