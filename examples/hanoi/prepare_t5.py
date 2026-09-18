"""Download T5 on a CPU host, then cache the two Hanoi instruction embeddings."""

import argparse
import json
import os
import pickle
import resource
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "google-t5/t5-11b"
REVISION = "90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3"
PROMPTS = (
    "Move all four rings from peg A to peg C following Tower of Hanoi rules.",
    "Move all four rings from peg C to peg A following Tower of Hanoi rules.",
)


def peak_process_rss_gib():
    """Linux ru_maxrss is KiB; this includes the transient checkpoint load."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("download", "encode"))
    parser.add_argument("--model-dir", type=Path, default=ROOT / "checkpoints/google-t5/t5-11b")
    parser.add_argument("--output", type=Path, default=ROOT / "data/hanoi_cosmos/t5_embeddings.pkl")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ["HF_HUB_CACHE"] = str(ROOT / ".cache/huggingface/hub")
    os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.phase == "download":
        if args.model_dir != ROOT / "checkpoints/google-t5/t5-11b":
            raise ValueError("Download uses the fixed repository checkpoint directory")
        from download_public_assets import download_group

        download_group("t5")
        print(json.dumps({"phase": "downloaded", "model_dir": str(args.model_dir), "revision": REVISION}), flush=True)
        return

    import torch
    from transformers import T5EncoderModel, T5TokenizerFast

    torch.set_num_threads(args.threads)
    if args.output.exists():
        with args.output.open("rb") as stream:
            existing = pickle.load(stream)
        if set(existing) != set(PROMPTS) or any(
            tuple(existing[prompt].shape) != (1, 512, 1024) or not torch.isfinite(existing[prompt]).all()
            for prompt in PROMPTS
        ):
            raise RuntimeError(f"Existing cache is invalid; refusing to overwrite: {args.output}")
        print(f"Valid embeddings already exist: {args.output}", flush=True)
        return

    started = time.monotonic()
    dtype = getattr(torch, args.precision)
    tokenizer = T5TokenizerFast.from_pretrained(str(args.model_dir), local_files_only=True)
    print(json.dumps({"phase": "loading", "peak_process_rss_gib": peak_process_rss_gib()}), flush=True)
    model = T5EncoderModel.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
        use_safetensors=False,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).eval()
    print(
        json.dumps(
            {
                "phase": "loaded",
                "device": str(next(model.parameters()).device),
                "precision": args.precision,
                "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                "peak_process_rss_gib": peak_process_rss_gib(),
            }
        ),
        flush=True,
    )
    embeddings = {}
    with torch.inference_mode():
        for prompt in PROMPTS:
            encoded = tokenizer.batch_encode_plus(
                [prompt],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512,
                return_length=True,
                return_offsets_mapping=False,
            )
            output = model(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask).last_hidden_state
            length = int(encoded.attention_mask.sum())
            output[:, length:] = 0
            if tuple(output.shape) != (1, 512, 1024) or not torch.isfinite(output).all():
                raise RuntimeError(f"Invalid embedding for {prompt}")
            embeddings[prompt] = output.cpu().clone()
            print(
                json.dumps(
                    {
                        "phase": "encoded",
                        "prompt": prompt,
                        "tokens": length,
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_process_rss_gib": peak_process_rss_gib(),
                    }
                ),
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_suffix(args.output.suffix + ".partial")
    with pending.open("wb") as stream:
        pickle.dump(embeddings, stream, protocol=pickle.HIGHEST_PROTOCOL)
    pending.replace(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "repository": MODEL_ID,
                "revision": REVISION,
                "precision": args.precision,
                "device": "cpu",
                "prompts": list(PROMPTS),
                "shape": [1, 512, 1024],
                "elapsed_seconds": time.monotonic() - started,
                "peak_process_rss_gib": peak_process_rss_gib(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
