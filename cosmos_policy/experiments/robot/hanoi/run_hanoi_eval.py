r"""GPU smoke test and sampled held-out BC evaluation; no robot commands are sent.

Example (inside a Slurm GPU allocation, from this repository):
    COSMOS_POLICY_PLATFORM=hanoi python -m cosmos_policy.experiments.robot.hanoi.run_hanoi_eval \
        --checkpoint /path/to/hanoi/checkpoints/iter_000001000/model \
        --metadata-dir data/hanoi_cosmos/aaaa_to_cccc --samples 2 --output /path/to/run/smoke.json

Use --samples 100 (or larger) for a sampled validation report. Metrics measure
offline action imitation, not physical task success. The test split is opt-in.
"""

import argparse
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from cosmos_policy.experiments.robot.hanoi.policy import HanoiInferenceConfig, load_hanoi_policy, predict_hanoi_actions
from cosmos_policy.datasets.hanoi_data import PROPRIO_COLUMNS, PROPRIO_SCHEMA


def select_indices(index: dict, samples: int, seed: int) -> np.ndarray:
    """Choose anchors across episode/direction groups, without duplicate anchors."""
    if samples < 1:
        raise ValueError("samples must be positive")
    group_keys = np.column_stack((index["episode_index"], index["source_index"]))
    groups = np.unique(group_keys, axis=0)
    rng = np.random.default_rng(seed)
    candidates = []
    for episode, source in groups:
        choices = np.flatnonzero((index["episode_index"] == episode) & (index["source_index"] == source))
        candidates.append(rng.permutation(choices))
    selected = []
    depth = 0
    while len(selected) < min(samples, len(group_keys)):
        for group in candidates:
            if depth < len(group) and len(selected) < samples:
                selected.append(group[depth])
        depth += 1
    return np.asarray(selected, dtype=np.int64)


def read_sample(handle, row: int, episode: int) -> tuple:
    """Read a fresh anchor and same-row labels, padding only within its episode."""
    start = int(handle["ep_offset"][episode])
    end = start + int(handle["ep_len"][episode])
    if not start <= row < end:
        raise ValueError("Index anchor lies outside the recorded episode")
    age_ns = int(handle["command_monotonic_ns"][row]) - int(handle["image_receipt_monotonic_ns"][row])
    if not 0 <= age_ns <= 50_000_000:
        raise ValueError("Index contains an ineligible observation anchor; regenerate dataset metadata")
    target = np.asarray(handle["action_abs"][row : min(row + 63, end)], dtype=np.float32)
    if len(target) < 63:
        target = np.concatenate((target, np.repeat(target[-1:], 63 - len(target), axis=0)))
    return handle["pixels"][row], handle["proprio"][row, list(PROPRIO_COLUMNS)], target


def chunk_metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    """XYZ errors in physical units and jaw accuracy over the committed/chunk horizons."""
    metrics = {}
    for horizon, label in ((1, "first"), (9, "commit9"), (63, "chunk63")):
        xyz_error = predicted[:horizon, :3] - target[:horizon, :3]
        metrics[f"{label}_xyz_rmse_mm"] = float(1000 * np.sqrt(np.mean(xyz_error**2)))
        metrics[f"{label}_xyz_l2_mm"] = float(1000 * np.linalg.norm(xyz_error, axis=-1).mean())
        metrics[f"{label}_jaw_accuracy"] = float(np.mean(predicted[:horizon, 3] == target[:horizon, 3]))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Local .pt or DCP checkpoint model directory")
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path(os.environ.get("HANOI_METADATA_DIR", "data/hanoi_cosmos/aaaa_to_cccc_pos_only")),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/scratch/cw5167/datasets"))
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path(os.environ.get("HANOI_T5_EMBEDDINGS", "data/hanoi_cosmos/t5_embeddings.pkl")),
    )
    parser.add_argument(
        "--direction",
        choices=("aaaa_to_cccc", "cccc_to_aaaa"),
        default=os.environ.get("HANOI_DIRECTION", "aaaa_to_cccc"),
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--denoising-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")

    # Set before importing the model modules which bind platform dimensions.
    os.environ["COSMOS_POLICY_PLATFORM"] = "hanoi"
    import h5py
    import torch

    from cosmos_policy.datasets.hanoi_data import DIRECTIONS, source_path, validate_source_scope

    if not torch.cuda.is_available():
        raise RuntimeError("This entrypoint requires a CUDA GPU; run it inside a Slurm allocation")
    with np.load(args.metadata_dir / f"{args.split}_indices.npz", allow_pickle=False) as archive:
        index = {key: archive[key] for key in ("source_index", "episode_index", "row_index")}
    metadata = json.loads((args.metadata_dir / "metadata.json").read_text())
    selected_sources = validate_source_scope(metadata, index["source_index"], args.direction)
    selected = select_indices(index, args.samples, args.seed)
    if not len(selected):
        raise ValueError(f"No eligible anchors in {args.split} split")
    # Reject accidental train/test contamination before allocating model memory.
    expected_episodes = range(40, 45) if args.split == "val" else range(45, 50)
    if not np.isin(index["episode_index"], list(expected_episodes)).all():
        raise ValueError(f"{args.split} indices contain episode IDs outside the fixed paired split")
    cfg = HanoiInferenceConfig(
        ckpt_path=args.checkpoint,
        dataset_stats_path=str(args.metadata_dir / "dataset_statistics.json"),
        t5_text_embeddings_path=str(args.embeddings or args.metadata_dir / "t5_embeddings.pkl"),
        num_denoising_steps_action=args.denoising_steps,
    )
    torch.cuda.reset_peak_memory_stats()
    model, stats, _ = load_hanoi_policy(cfg)
    torch.cuda.synchronize()
    load_peak_allocated = torch.cuda.max_memory_allocated()
    load_peak_reserved = torch.cuda.max_memory_reserved()
    report = {
        "proprio_schema": PROPRIO_SCHEMA,
        "proprio_columns": list(PROPRIO_COLUMNS),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset_stats_path": str(Path(cfg.dataset_stats_path).resolve()),
        "embeddings_path": str(Path(cfg.t5_text_embeddings_path).resolve()),
        "split": args.split,
        "directions": list(selected_sources.values()),
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "denoising_steps": args.denoising_steps,
        "load_peak_allocated_bytes": load_peak_allocated,
        "load_peak_reserved_bytes": load_peak_reserved,
        "warmup_queries": args.warmup,
        "memory_note": "PyTorch allocator peaks; CUDA context and external library allocations may add overhead.",
        "latency_note": "Synchronized policy invocation on the named GPU; excludes HDF5 reads and robot transport.",
        "samples": [],
    }
    with ExitStack() as stack:
        handles = {
            i: stack.enter_context(h5py.File(source_path(args.data_dir, direction), "r"))
            for i, direction in selected_sources.items()
        }
        first = int(selected[0])
        source = int(index["source_index"][first])
        pixels, state, _ = read_sample(
            handles[source], int(index["row_index"][first]), int(index["episode_index"][first])
        )
        for _ in range(args.warmup):
            predict_hanoi_actions(cfg, model, stats, pixels, state, DIRECTIONS[source], seed=args.seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for selected_index in selected:
            source = int(index["source_index"][selected_index])
            episode = int(index["episode_index"][selected_index])
            row = int(index["row_index"][selected_index])
            pixels, state, target = read_sample(handles[source], row, episode)
            torch.cuda.synchronize()
            started = time.perf_counter()
            predicted = predict_hanoi_actions(cfg, model, stats, pixels, state, DIRECTIONS[source], seed=args.seed)
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
            sample = {
                "direction": DIRECTIONS[source],
                "episode_index": episode,
                "row_index": row,
                "latency_seconds": latency,
                **chunk_metrics(predicted, target),
            }
            report["samples"].append(sample)
            print(json.dumps(sample), flush=True)
    latencies = np.array([sample["latency_seconds"] for sample in report["samples"]])
    report.update(
        num_samples=len(report["samples"]),
        inference_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        inference_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        latency_mean_seconds=float(latencies.mean()),
        latency_p50_seconds=float(np.percentile(latencies, 50)),
        latency_p95_seconds=float(np.percentile(latencies, 95)),
        # Nine commands at 30 Hz gives a 0.3-second replanning budget.
        mean_latency_within_300ms=bool(latencies.mean() <= 0.3),
        metric_note="Unweighted mean of per-anchor metrics; offline imitation only, not robot task success.",
    )
    metric_names = chunk_metrics(np.zeros((63, 4)), np.zeros((63, 4)))
    report["mean_metrics"] = {
        key: float(np.mean([sample[key] for sample in report["samples"]])) for key in metric_names
    }
    report["mean_metrics_by_direction"] = {}
    for direction in DIRECTIONS:
        samples = [sample for sample in report["samples"] if sample["direction"] == direction]
        if samples:
            report["mean_metrics_by_direction"][direction] = {
                key: float(np.mean([sample[key] for sample in samples])) for key in metric_names
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
