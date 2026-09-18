#!/usr/bin/env bash
#SBATCH --job-name=cosmos-hanoi-download
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=cosmos-hanoi-download-%j.out
#SBATCH --error=cosmos-hanoi-download-%j.err

set -euo pipefail
source examples/hanoi/env.sh
exec /usr/bin/python3 examples/hanoi/download_public_assets.py --group t5
