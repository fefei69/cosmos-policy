#!/usr/bin/env bash
# Fetch the gated Cosmos-Predict2-2B video base (run B initial weights).
# Usage: HF_TOKEN=hf_... bash examples/hanoi/fetch_video_base.sh
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face read token with the license accepted}"
cd "$(dirname "$0")/../.."
out=checkpoints/public/model-480p-16fps.pt
url="https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World/resolve/main/model-480p-16fps.pt"
wget -c --header="Authorization: Bearer $HF_TOKEN" -O "$out" "$url"
echo "size: $(stat -c %s "$out") bytes (expected 3913017214)"
echo "header: $(head -c 4 "$out" | xxd -p)  (504b0304 = zip container)"
