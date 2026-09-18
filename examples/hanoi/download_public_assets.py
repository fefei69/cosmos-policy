"""Fetch independently public model assets, pinning upstream revisions and hashes."""

import concurrent.futures
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "checkpoints" / "public"
ASSETS = [
    ("nvidia/Cosmos-Policy-LIBERO-Predict2-2B", "cb689ec0e3347c13667d70a78a3447388f5c3bb8", "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
    ("Wan-AI/Wan2.1-T2V-1.3B", "37ec512624d61f7aa208f7ea8140a131f93afc9a", "Wan2.1_VAE.pth"),
]


def fetch(asset):
    repo, revision, filename = asset
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true", timeout=60) as response:
        metadata = json.load(response)
    entry = next(x for x in metadata["siblings"] if x["rfilename"] == filename)
    expected_size = entry["size"]
    expected_sha = entry.get("lfs", {}).get("sha256")
    destination = DEST / filename
    partial = destination.with_suffix(destination.suffix + ".partial")
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    if not destination.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        if offset != expected_size:
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and response.status != 206:
                    offset = 0
                started = time.monotonic()
                transferred = 0
                last_report = started
                with partial.open("ab" if offset else "wb") as stream:
                    while chunk := response.read(8 * 1024 * 1024):
                        stream.write(chunk)
                        transferred += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 60:
                            print(json.dumps({"filename": filename, "downloaded_bytes": offset + transferred,
                                              "total_bytes": expected_size, "bytes_per_second": transferred / (now - started)}), flush=True)
                            last_report = now
        if partial.stat().st_size != expected_size:
            raise RuntimeError(f"Incomplete download: {filename}")
        candidate = partial
    else:
        candidate = destination
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    if candidate.stat().st_size != expected_size or (expected_sha and digest.hexdigest() != expected_sha):
        raise RuntimeError(f"Size or SHA256 verification failed: {filename}")
    if candidate == partial:
        partial.replace(destination)
    result = {"repository": repo, "revision": revision, "filename": filename, "bytes": expected_size, "sha256": digest.hexdigest(), "upstream_sha256": expected_sha, "path": str(destination)}
    print(json.dumps(result), flush=True)
    return result


def download_group(group="policy"):
    global DEST
    assets = ASSETS
    if group == "t5":
        DEST = ROOT / "checkpoints/google-t5/t5-11b"
        assets = [("google-t5/t5-11b", "90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3", name)
                  for name in ("config.json", "spiece.model", "tokenizer.json", "pytorch_model.bin")]
    DEST.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        manifest = list(executor.map(fetch, assets))
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("policy", "t5"), default="policy")
    download_group(parser.parse_args().group)
