#!/usr/bin/env bash
#SBATCH --job-name=cosmos-hanoi-future-frames
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public,a100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=data/hanoi_cosmos/logs/cosmos-hanoi-future-frames-%j.out
#SBATCH --error=data/hanoi_cosmos/logs/cosmos-hanoi-future-frames-%j.err
# Decode predicted future frames for held-out examples. Read-only with respect
# to every training run: outputs go under data/hanoi_cosmos/evals/future_frames/.
set -euo pipefail
source examples/hanoi/env.sh
export COSMOS_POLICY_PLATFORM=hanoi_joint
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$HANOI_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
TMP=$(mktemp -d "/tmp/cosmos-future-${SLURM_JOB_ID}.XXXXXX"); export TMPDIR="$TMP"
trap 'rm -rf "$TMP" 2>/dev/null || true' EXIT
RUNS=data/hanoi_cosmos/runs/cosmos_policy/hanoi
SAMPLES="${HANOI_FUTURE_SAMPLES:-24}"
# "<run>:<export step>:<metadata dir>" entries; v3 first as requested.
for entry in ${HANOI_FUTURE_TARGETS:-hanoi_cosmos_joint_sparse_20260917:16000:data/hanoi_cosmos/joint_sparse_v3 hanoi_cosmos_waypoint_v4_20260917:2000:data/hanoi_cosmos/waypoint_v4}; do
  IFS=: read -r run step metadata <<<"$entry"
  ckpt=$(printf '%s/%s/exports/iter_%09d.pt' "$RUNS" "$run" "$step")
  out=$(printf 'data/hanoi_cosmos/evals/future_frames/%s/iter_%09d' "$run" "$step")
  if [[ -d "$out" ]]; then echo "skip existing $out"; continue; fi
  if [[ ! -f "$ckpt" ]]; then echo "missing $ckpt"; continue; fi
  echo "=== $(date '+%F %T') $run step $step"
  "$HANOI_PROJECT_ROOT/.venv/bin/python" examples/hanoi/predict_future_images.py \
    --checkpoint "$ckpt" --metadata "$metadata" --split val --samples "$SAMPLES" --output-dir "$out"
done
echo "=== $(date '+%F %T') done"
