"""CPU preparation and shared data contract for the Hanoi Cosmos Policy dataset.

Only metadata and numeric arrays are read during preparation. RGB remains in the
original read-only HDF5 files and is fetched one observation at a time in training.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

DIRECTIONS = ("aaaa_to_cccc", "cccc_to_aaaa")
SOURCE_FILENAMES = {
    direction: f"hanoi_wm_roundtrip_20260910_195558_{direction.upper().replace('_TO_', '_to_')}.h5"
    for direction in DIRECTIONS
}
PROMPTS = {
    "aaaa_to_cccc": "Move all four rings from peg A to peg C following Tower of Hanoi rules.",
    "cccc_to_aaaa": "Move all four rings from peg C to peg A following Tower of Hanoi rules.",
}
ACTION_HORIZON = 63
MAX_IMAGE_AGE_NS = 50_000_000
FORMAT_VERSION = 2
PROPRIO_COLUMNS = (0, 1, 2, 6)  # Measured XYZ and jaw stroke; no velocity or commanded jaw.
PROPRIO_SCHEMA = "xyz_measured_jaw_v1"


def source_path(data_dir: str | Path, direction: str) -> Path:
    return Path(data_dir) / SOURCE_FILENAMES[direction]


def validate_source_scope(metadata: dict, source_indices, expected_direction: str | None = None) -> dict:
    """Keep split indices and the requested experiment within their declared files."""
    directions = [source["direction"] for source in metadata["sources"]]
    if not directions or len(set(directions)) != len(directions) or not set(directions) <= set(DIRECTIONS):
        raise ValueError("Invalid Hanoi source directions in metadata")
    if expected_direction is not None and directions != [expected_direction]:
        raise ValueError(f"Expected only {expected_direction}; regenerate metadata for that direction")
    sources = {DIRECTIONS.index(direction): direction for direction in directions}
    if not np.isin(source_indices, list(sources)).all():
        raise ValueError("Split indices contain a source outside the selected dataset scope")
    return sources


def split_for_pair(pair: int) -> str:
    if not 0 <= pair < 50:
        raise ValueError(f"Collection pair must be in [0, 49], got {pair}")
    return "train" if pair < 40 else "val" if pair < 45 else "test"


def eligible_anchors(handle: h5py.File, start: int, stop: int) -> np.ndarray:
    age = handle["command_monotonic_ns"][start:stop] - handle["image_receipt_monotonic_ns"][start:stop]
    return (age >= 0) & (age <= MAX_IMAGE_AGE_NS)


def action_chunk(
    action_abs: np.ndarray, anchor: int, measured_xyz: np.ndarray, horizon: int = ACTION_HORIZON
) -> np.ndarray:
    """Transform an episode's t-aligned absolute targets using one measured anchor.

    The array must contain only this episode. Padding repeats its final target;
    the recorded per-row ``action`` differences are deliberately not used.
    """
    if horizon < 1 or not 0 <= anchor < len(action_abs):
        raise ValueError("Invalid action horizon or episode-local anchor")
    chunk = np.asarray(
        action_abs[np.minimum(np.arange(anchor, anchor + horizon), len(action_abs) - 1)], dtype=np.float32
    ).copy()
    if chunk.shape != (horizon, 4) or np.shape(measured_xyz) != (3,):
        raise ValueError("Expected absolute actions (T, 4) and measured XYZ (3,)")
    chunk[:, :3] -= np.asarray(measured_xyz, dtype=np.float32)
    return chunk


def normalize(values: np.ndarray, stats: dict, field: str) -> np.ndarray:
    """Stock Cosmos min/max scaling, without clipping trajectories."""
    lower = np.asarray(stats[f"{field}_min"], dtype=np.float32)
    upper = np.asarray(stats[f"{field}_max"], dtype=np.float32)
    if np.any(upper <= lower):
        raise ValueError(f"Invalid {field} normalization range")
    return np.asarray(2 * ((np.asarray(values, dtype=np.float32) - lower) / (upper - lower)) - 1, dtype=np.float32)


def _validate_metadata(handle: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    if int(handle.attrs.get("schema_version", -1)) != 2:
        raise ValueError("Hanoi source must have schema_version=2")
    if handle.attrs.get("action_abs_alignment") != "post_action_reference":
        raise ValueError("Unexpected action_abs alignment; targets must begin at the observation row")
    if float(handle.attrs.get("rate_hz", 0)) != 30 or bool(handle.attrs.get("dry_run", True)):
        raise ValueError("Expected real, 30 Hz Hanoi collection")
    n = len(handle["pixels"])
    shapes = {
        "pixels": (n, 224, 224, 3),
        "proprio": (n, 8),
        "action_abs": (n, 4),
        "command_monotonic_ns": (n,),
        "image_receipt_monotonic_ns": (n,),
    }
    for name, shape in shapes.items():
        if handle[name].shape != shape:
            raise ValueError(f"Unexpected {name} shape: {handle[name].shape}, expected {shape}")
    expected_dtypes = {
        "pixels": "uint8",
        "proprio": "float32",
        "action_abs": "float32",
        "command_monotonic_ns": "int64",
        "image_receipt_monotonic_ns": "int64",
    }
    for name, dtype in expected_dtypes.items():
        if handle[name].dtype != np.dtype(dtype):
            raise ValueError(f"Unexpected {name} dtype: {handle[name].dtype}")
    offsets = np.asarray(handle["ep_offset"], dtype=np.int64)
    lengths = np.asarray(handle["ep_len"], dtype=np.int64)
    if offsets.shape != (50,) or lengths.shape != (50,) or np.any(lengths <= 0):
        raise ValueError("Expected 50 nonempty episodes in each direction")
    expected_offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))
    if not np.array_equal(offsets, expected_offsets) or int(lengths.sum()) != n:
        raise ValueError("Episode bounds must partition the source rows without gaps or overlaps")
    return offsets, lengths


def _atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary_path = stream.name
    os.replace(temporary_path, path)


def prepare_hanoi_data(
    data_dir: str | Path,
    output_dir: str | Path,
    chunk_size: int = ACTION_HORIZON,
    directions: tuple[str, ...] = DIRECTIONS,
) -> dict:
    """Write fresh-anchor split indices and training-only normalization statistics.

    Statistics use all 63 target offsets (including endpoint padding) after
    subtraction of each eligible training anchor's measured XYZ. Proprio bounds
    include the observed and supervised future states from training episodes.
    Validation/test numeric labels never enter the statistics.
    """
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if not directions or len(set(directions)) != len(directions) or not set(directions) <= set(DIRECTIONS):
        raise ValueError(f"Select one or both unique directions from {DIRECTIONS}")
    if chunk_size != ACTION_HORIZON:
        raise ValueError("The Hanoi handoff fixes the action horizon at 63")
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise ValueError("Derived artifacts must be stored outside the raw source directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    selections = {split: [] for split in ("train", "val", "test")}
    actions_min, actions_max = np.full(4, np.inf), np.full(4, -np.inf)
    proprio_min, proprio_max = np.full(4, np.inf), np.full(4, -np.inf)
    episodes, sources = [], []
    for source_index, direction in enumerate(DIRECTIONS):
        if direction not in directions:
            continue
        path = source_path(data_dir, direction)
        identity = path.stat()
        sources.append(
            {
                "direction": direction,
                "filename": path.name,
                "path": str(path),
                "size_bytes": identity.st_size,
                "mtime_ns": identity.st_mtime_ns,
            }
        )
        with h5py.File(path, "r") as handle:
            offsets, lengths = _validate_metadata(handle)
            for episode_index, (start, length) in enumerate(zip(offsets, lengths, strict=True)):
                start, length = int(start), int(length)
                split = split_for_pair(episode_index)
                local_anchors = np.flatnonzero(eligible_anchors(handle, start, start + length))
                episodes.append(
                    {
                        "source_index": source_index,
                        "direction": direction,
                        "episode_index": episode_index,
                        "start": start,
                        "length": length,
                        "split": split,
                        "eligible": len(local_anchors),
                    }
                )
                selections[split].append(
                    (
                        np.full(len(local_anchors), source_index, dtype=np.uint8),
                        np.full(len(local_anchors), episode_index, dtype=np.int16),
                        local_anchors.astype(np.int64) + start,
                    )
                )
                if split != "train" or not len(local_anchors):
                    continue
                # Only small numeric arrays enter RAM; never scan pixels.
                measured = handle["proprio"][start : start + length, :][:, PROPRIO_COLUMNS]
                references = handle["action_abs"][start : start + length]
                if not np.isfinite(measured).all() or not np.isfinite(references).all():
                    raise ValueError(f"Nonfinite training state/target in {direction} episode {episode_index}")
                if not np.isin(references[:, 3], [0, 1]).all():
                    raise ValueError("Jaw targets must be absolute binary intent")
                future = np.minimum(local_anchors + chunk_size, length - 1)
                states = measured[np.concatenate((local_anchors, future))]
                proprio_min = np.minimum(proprio_min, states.min(axis=0))
                proprio_max = np.maximum(proprio_max, states.max(axis=0))
                for offset in range(chunk_size):
                    targets = references[np.minimum(local_anchors + offset, length - 1)].copy()
                    targets[:, :3] -= measured[local_anchors, :3]
                    actions_min = np.minimum(actions_min, targets.min(axis=0))
                    actions_max = np.maximum(actions_max, targets.max(axis=0))
    if not np.isfinite(actions_min).all() or not np.isfinite(proprio_min).all():
        raise ValueError("No finite eligible training targets were found")
    observed_bounds = {
        "actions_min": actions_min.tolist(),
        "actions_max": actions_max.tolist(),
        "proprio_min": proprio_min.tolist(),
        "proprio_max": proprio_max.tolist(),
    }
    constant_features = {}
    for name, lower, upper in (("actions", actions_min, actions_max), ("proprio", proprio_min, proprio_max)):
        constant = upper == lower
        constant_features[name] = np.flatnonzero(constant).tolist()
        # Symmetric bounds map constant features to zero and permit the exact
        # stock inverse transform without an inference-only epsilon convention.
        lower[constant] -= 0.5
        upper[constant] += 0.5
    stats = {
        "actions_min": actions_min.tolist(),
        "actions_max": actions_max.tolist(),
        "proprio_min": proprio_min.tolist(),
        "proprio_max": proprio_max.tolist(),
    }
    counts = {}
    for split, groups in selections.items():
        arrays = dict(
            zip(
                ("source_index", "episode_index", "row_index"),
                (np.concatenate(items) for items in zip(*groups, strict=True)),
                strict=True,
            )
        )
        counts[split] = len(arrays["row_index"])
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".npz", delete=False) as stream:
            np.savez_compressed(stream, **arrays)
            temporary_path = stream.name
        os.replace(temporary_path, output_dir / f"{split}_indices.npz")
    manifest = {
        "format_version": FORMAT_VERSION,
        "chunk_size": chunk_size,
        "reference_rate_hz": 30,
        "max_image_age_ns": MAX_IMAGE_AGE_NS,
        "proprio_columns": list(PROPRIO_COLUMNS),
        "proprio_schema": PROPRIO_SCHEMA,
        "action_representation": "action_abs[t:t+63]; XYZ minus measured XYZ at t; jaw absolute",
        "prompts": {direction: PROMPTS[direction] for direction in DIRECTIONS if direction in directions},
        "sources": sources,
        "episodes": episodes,
        "eligible_counts": counts,
        "normalization_split": "train",
        "normalization_proprio": "current and future training states",
        "constant_features": constant_features,
        "observed_training_bounds": observed_bounds,
    }
    _atomic_json(output_dir / "dataset_statistics.json", stats)
    _atomic_json(output_dir / "metadata.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("HANOI_DATA_ROOT", "/scratch/cw5167/datasets"))
    parser.add_argument("--output-dir", default=os.environ.get("HANOI_METADATA_DIR", "data/hanoi_cosmos"))
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=DIRECTIONS,
        default=[os.environ["HANOI_DIRECTION"]] if "HANOI_DIRECTION" in os.environ else list(DIRECTIONS),
    )
    args = parser.parse_args()
    manifest = prepare_hanoi_data(args.data_dir, args.output_dir, directions=tuple(args.directions))
    print(
        json.dumps(
            {"output_dir": str(Path(args.output_dir).resolve()), "eligible_counts": manifest["eligible_counts"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
