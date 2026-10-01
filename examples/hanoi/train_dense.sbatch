#!/usr/bin/env bash
#SBATCH --job-name=cosmos-hanoi-dense
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=data/hanoi_cosmos/logs/cosmos-hanoi-dense-%j.out
#SBATCH --error=data/hanoi_cosmos/logs/cosmos-hanoi-dense-%j.err
# hanoi_dense_v5 training on one H200 (per docs/hanoi_training_plan.md: account,
# --gres and a feature constraint; Torch routes the partition). A run needs two
# 12-hour allocations; the launcher resumes from its latest complete checkpoint.
#   sbatch examples/hanoi/train_dense.sbatch --init libero   # run A
#   sbatch examples/hanoi/train_dense.sbatch --init video    # run B
#   sbatch --dependency=afterany:<jobid> examples/hanoi/train_dense.sbatch --init <same>   # continuation
#   sbatch examples/hanoi/train_dense.sbatch --init video --horizon 32 --run-name <name>_h32   # chunk-32 comparison (dense_v5_h32)
# A continuation whose dense code changed since the run started is refused unless
# HANOI_DENSE_ACCEPT_CODE_CHANGES="<reason>" is exported; the launcher then records
# every changed file's old and new hash in the run's code_updates.json.
set -euo pipefail
source examples/hanoi/env.sh
export COSMOS_POLICY_PLATFORM=hanoi_dense
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
TMP=$(mktemp -d "/tmp/cosmos-dense-${SLURM_JOB_ID}.XXXXXX"); export TMPDIR="$TMP"
trap 'rm -rf "$TMP" 2>/dev/null || true' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv || true
"$HANOI_PROJECT_ROOT/.venv/bin/python" examples/hanoi/run_dense.py "$@"
