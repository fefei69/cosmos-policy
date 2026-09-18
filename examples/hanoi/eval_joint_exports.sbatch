#!/usr/bin/env bash
#SBATCH --job-name=cosmos-joint-physical-eval
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=data/hanoi_cosmos/logs/cosmos-joint-physical-eval-%j.out
#SBATCH --error=data/hanoi_cosmos/logs/cosmos-joint-physical-eval-%j.err
# Physical (decoded-action) evaluation of existing exports on the full
# validation split. Read-only with respect to the training run: outputs go to a
# separate directory, never into the run's own evaluation/ or selection files.
set -euo pipefail
source examples/hanoi/env.sh
export COSMOS_POLICY_PLATFORM=hanoi_joint
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$HANOI_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
TMP=$(mktemp -d "/tmp/cosmos-eval-${SLURM_JOB_ID}.XXXXXX"); export TMPDIR="$TMP"
trap 'rm -rf "$TMP" 2>/dev/null || true' EXIT
RUN="${HANOI_JOINT_RUN_NAME:-hanoi_cosmos_joint_sparse_20260917}"
EXPORTS="data/hanoi_cosmos/runs/cosmos_policy/hanoi/$RUN/exports"
OUT="data/hanoi_cosmos/evals/$RUN/physical_val"
# Generic evaluator: accepts H100 or H200 and reads the dataset contract from metadata.
mkdir -p "$OUT"
# Most informative first, so a partial job still answers the question.
for step in ${HANOI_EVAL_STEPS:-30000 14000 4000 8000 22000 2000 6000 10000 12000 16000 18000 20000 24000 26000 28000}; do
  ckpt=$(printf '%s/iter_%09d.pt' "$EXPORTS" "$step")
  out=$(printf '%s/iter_%09d_val_actions.json' "$OUT" "$step")
  if [[ -f "$out" ]]; then echo "skip existing $out"; continue; fi
  if [[ ! -f "$ckpt" ]]; then echo "missing $ckpt"; continue; fi
  echo "=== $(date '+%F %T') evaluating step $step"
  "$HANOI_PROJECT_ROOT/.venv/bin/python" -m cosmos_policy.experiments.robot.hanoi.run_hanoi_physical_eval \
    --checkpoint "$ckpt" --metadata data/hanoi_cosmos/joint_sparse_v3 --split val --samples 0 \
    --mode actions --output "$out"
done
echo "=== $(date '+%F %T') done"
