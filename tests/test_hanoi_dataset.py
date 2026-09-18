"""Contract tests using small numeric fixtures and sparse HDF5 image datasets."""

import json
import pickle

import h5py
import numpy as np
import pytest
import torch

from cosmos_policy.datasets.hanoi_data import (
    DIRECTIONS,
    PROMPTS,
    PROPRIO_COLUMNS,
    action_chunk,
    eligible_anchors,
    normalize,
    prepare_hanoi_data,
    source_path,
    validate_source_scope,
)
from cosmos_policy.datasets.hanoi_dataset import HanoiDataset


@pytest.fixture
def prepared(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "derived"
    raw.mkdir()
    length = 70
    image = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    for source_index, direction in enumerate(DIRECTIONS):
        with h5py.File(source_path(raw, direction), "w") as handle:
            handle.attrs.update(
                schema_version=2, action_abs_alignment="post_action_reference", rate_hz=30, dry_run=False
            )
            offsets = np.arange(50, dtype=np.int64) * length
            handle["ep_offset"] = offsets
            handle["ep_len"] = np.full(50, length, dtype=np.int32)
            row = np.tile(np.arange(length, dtype=np.float32), 50)
            episode = np.repeat(np.arange(50, dtype=np.float32), length)
            measured = np.zeros((50 * length, 8), dtype=np.float32)
            measured[:, 0] = episode + 100 * source_index + row * 0.001
            measured[:, 1] = row * 0.002
            measured[:, 2] = 0.2 + row * 0.003
            measured[:, 3] = 0.03
            measured[:, 6] = 0.04  # Constant measured jaw tests constant-feature normalization.
            measured[:, 7] = 1e9  # This leaky command must never reach observations or statistics.
            references = np.concatenate((measured[:, :3] + [0.001, 0.002, -0.003], (row % 2)[:, None]), axis=1).astype(
                np.float32
            )
            # Large held-out coordinates expose normalization contamination.
            measured[episode >= 40, :3] += 10000
            references[episode >= 40, :3] += 20000
            handle["proprio"] = measured
            handle["action_abs"] = references
            timestamps = np.arange(50 * length, dtype=np.int64) * 33_333_333 + 1_000_000_000
            ages = np.zeros(50 * length, dtype=np.int64)
            ages[offsets + 1] = 50_000_000  # Inclusive freshness limit.
            ages[offsets + 2] = 50_000_001
            ages[offsets + 3] = -1
            handle["command_monotonic_ns"] = timestamps
            handle["image_receipt_monotonic_ns"] = timestamps - ages
            # Allocate logical image shape without writing the full image corpus.
            pixels = handle.create_dataset(
                "pixels", shape=(50 * length, 224, 224, 3), dtype="uint8", chunks=(1, 224, 224, 3), compression="gzip"
            )
            pixels[0] = image
            pixels[63] = 255 - image
            handle["board"] = np.full((50 * length, 4), 99, dtype=np.int8)
    manifest = prepare_hanoi_data(raw, output)
    embeddings = output / "t5_embeddings.pkl"
    with embeddings.open("wb") as stream:
        pickle.dump(
            {prompt: torch.full((1, 512, 1024), i, dtype=torch.bfloat16) for i, prompt in enumerate(PROMPTS.values())},
            stream,
        )
    return raw, output, embeddings, manifest, image


def _dataset(prepared, **kwargs):
    raw, output, embeddings, _, _ = prepared
    return HanoiDataset(str(raw), str(embeddings), str(output), **kwargs)


def test_velocity_and_commanded_jaw_cannot_affect_inputs_targets_or_statistics(prepared):
    raw, output, _, _, _ = prepared
    dataset = _dataset(prepared)
    before = [dataset[i] for i in (0, 20, 67)]
    original_stats = (output / "dataset_statistics.json").read_text()
    dataset.close()
    for direction in DIRECTIONS:
        with h5py.File(source_path(raw, direction), "r+") as handle:
            handle["proprio"][:, 3:6] = 1e8
            handle["proprio"][:, 7] = float("nan")
    prepare_hanoi_data(raw, output)
    assert (output / "dataset_statistics.json").read_text() == original_stats
    dataset = _dataset(prepared)
    for old, i in zip(before, (0, 20, 67)):
        sample = dataset[i]
        for key in ("proprio", "future_proprio", "actions", "next_action_chunk", "video"):
            np.testing.assert_array_equal(sample[key], old[key])
        assert sample["proprio"].shape == (4,)
    dataset.close()


def test_legacy_velocity_metadata_is_rejected(prepared):
    _, output, _, _, _ = prepared
    path = output / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["proprio_columns"] = list(range(7))
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="velocity excluded"):
        _dataset(prepared)


def test_split_pairs_freshness_and_training_only_statistics(prepared):
    raw, output, _, manifest, _ = prepared
    assert manifest["eligible_counts"] == {"train": 5440, "val": 680, "test": 680}
    for split, pair_range in (("train", range(40)), ("val", range(40, 45)), ("test", range(45, 50))):
        with np.load(output / f"{split}_indices.npz") as indices:
            assert set(indices["source_index"]) == {0, 1}
            assert set(indices["episode_index"]) == set(pair_range)
            assert not np.isin(indices["row_index"] % 70, [2, 3]).any()
            assert (indices["row_index"] % 70 == 1).any()
    with h5py.File(source_path(raw, DIRECTIONS[0]), "r") as handle:
        assert eligible_anchors(handle, 0, 5).tolist() == [True, True, False, False, True]
    stats = json.loads((output / "dataset_statistics.json").read_text())
    assert len(stats["proprio_min"]) == 4
    assert max(stats["proprio_max"]) < 200  # Held-out XYZ and commanded jaw target are excluded.
    assert max(stats["actions_max"][:3]) < 1  # Statistics describe common-anchor deltas, not raw XYZ.
    assert stats["actions_min"][3] == 0 and stats["actions_max"][3] == 1
    assert manifest["proprio_columns"] == [0, 1, 2, 6]
    assert 3 in manifest["constant_features"]["proprio"]
    assert stats["proprio_min"][3] == pytest.approx(0.04 - 0.5) and stats["proprio_max"][3] == pytest.approx(0.04 + 0.5)

    # Changing validation/test labels cannot change any training statistics.
    for direction in DIRECTIONS:
        with h5py.File(source_path(raw, direction), "r+") as handle:
            handle["proprio"][40 * 70 :, :7] = 5e6
            handle["action_abs"][40 * 70 :, :3] = -5e6
    prepare_hanoi_data(raw, output)
    assert json.loads((output / "dataset_statistics.json").read_text()) == stats


def test_t_aligned_targets_keep_stale_rows_and_pad_within_episode(prepared):
    raw, _, _, _, _ = prepared
    dataset = _dataset(prepared, normalize_actions=False, normalize_proprio=False, gamma=0.9)
    sample = dataset[0]
    with h5py.File(source_path(raw, DIRECTIONS[0]), "r") as handle:
        measured = handle["proprio"][0, list(PROPRIO_COLUMNS)]
        expected = handle["action_abs"][:63].copy()
        expected[:, :3] -= measured[:3]
        np.testing.assert_array_equal(sample["proprio"], measured)
        np.testing.assert_allclose(sample["actions"], expected, atol=1e-7)
        # Row 2 is a stale observation anchor, but its dense action target survives.
        np.testing.assert_allclose(sample["actions"][2], expected[2], atol=1e-7)
        np.testing.assert_allclose(sample["future_proprio"], handle["proprio"][63, list(PROPRIO_COLUMNS)])
        last_index = np.flatnonzero((dataset.source_indices == 0) & (dataset.row_indices == 69))[0]
        final = dataset[int(last_index)]
        final_target = handle["action_abs"][69].copy()
        final_target[:3] -= handle["proprio"][69, :3]
        np.testing.assert_allclose(final["actions"], np.repeat(final_target[None], 63, axis=0), atol=1e-7)
        assert final["value_function_return"] == 1
    assert sample["value_function_return"] == pytest.approx(2 * 0.9**6 - 1)
    dataset.close()


def test_forward_only_preparation_does_not_require_reverse_file(prepared):
    raw, output, embeddings, _, _ = prepared
    source_path(raw, DIRECTIONS[1]).unlink()
    selected = output / "aaaa_to_cccc"
    manifest = prepare_hanoi_data(raw, selected, directions=(DIRECTIONS[0],))
    assert manifest["eligible_counts"] == {"train": 2720, "val": 340, "test": 340}
    assert len(manifest["episodes"]) == 50
    assert [source["direction"] for source in manifest["sources"]] == [DIRECTIONS[0]]
    assert manifest["prompts"] == {DIRECTIONS[0]: PROMPTS[DIRECTIONS[0]]}
    stats = json.loads((selected / "dataset_statistics.json").read_text())
    assert stats["proprio_max"][0] < 40  # Reverse training XYZ starts at 100.
    with embeddings.open("wb") as stream:
        pickle.dump({PROMPTS[DIRECTIONS[0]]: torch.zeros(512, 1024)}, stream)
    for split, episodes in (("train", range(40)), ("val", range(40, 45)), ("test", range(45, 50))):
        dataset = HanoiDataset(
            str(raw),
            str(embeddings),
            str(selected),
            split=split,
            expected_direction=DIRECTIONS[0],
            representative_order=split == "val",
        )
        assert set(dataset.source_indices) == {0}
        assert set(dataset.episode_indices) == set(episodes)
        assert dataset.unique_commands == {PROMPTS[DIRECTIONS[0]]}
        assert dataset[0]["actions"].shape == (63, 4)
        if split == "val":
            assert set(dataset.episode_indices[:5]) == set(range(40, 45))
        dataset.close()
    with pytest.raises(ValueError, match="outside the selected dataset"):
        validate_source_scope(manifest, np.array([0, 1]), DIRECTIONS[0])


def test_requested_direction_rejects_bidirectional_metadata(prepared):
    with pytest.raises(ValueError, match="Expected only aaaa_to_cccc"):
        _dataset(prepared, expected_direction=DIRECTIONS[0])


def test_video_slots_no_recrop_no_symbolic_inputs_and_text_shape(prepared):
    _, _, _, _, image = prepared
    dataset = _dataset(prepared)
    sample = dataset[0]
    assert sample["video"].shape == (3, 25, 224, 224)
    assert sample["video"].dtype == torch.uint8
    frames = sample["video"].permute(1, 2, 3, 0).numpy()
    np.testing.assert_array_equal(frames[5:9], np.repeat(image[None], 4, axis=0))
    np.testing.assert_array_equal(frames[17:21], np.repeat((255 - image)[None], 4, axis=0))
    assert not frames[[0, 1, 9, 13, 21]].any()
    assert sample["actions"].shape == (63, 4)
    assert sample["proprio"].shape == sample["future_proprio"].shape == (4,)
    assert sample["t5_text_embeddings"].shape == (512, 1024)
    assert sample["t5_text_embeddings"].dtype == torch.bfloat16
    assert [
        sample[k]
        for k in (
            "current_proprio_latent_idx",
            "current_image_latent_idx",
            "action_latent_idx",
            "future_proprio_latent_idx",
            "future_image_latent_idx",
            "value_latent_idx",
        )
    ] == list(range(1, 7))
    assert sample["future_wrist_image_latent_idx"] == -1
    assert sample["rollout_data_mask"] == sample["world_model_sample_mask"] == sample["value_function_sample_mask"] == 0
    assert not {"board", "goal_board", "phase", "move_idx", "held_disk", "episode_success", "route"} & sample.keys()
    reverse = int(np.flatnonzero(dataset.source_indices == 1)[0])
    assert torch.all(dataset[reverse]["t5_text_embeddings"] == 1)
    dataset.close()


def test_normalization_roundtrip_and_worker_serialization(prepared):
    raw_dataset = _dataset(prepared, normalize_actions=False, normalize_proprio=False)
    dataset = _dataset(prepared)
    sample, raw = dataset[0], raw_dataset[0]
    for field in ("actions", "proprio"):
        lower = np.asarray(dataset.dataset_stats[f"{field}_min"], dtype=np.float32)
        upper = np.asarray(dataset.dataset_stats[f"{field}_max"], dtype=np.float32)
        restored = 0.5 * (sample[field] + 1) * (upper - lower) + lower
        np.testing.assert_allclose(restored, raw[field], atol=1e-6)
    assert sample["proprio"][3] == 0
    # An already-open HDF5 handle must not be serialized into spawned workers.
    restored_dataset = pickle.loads(pickle.dumps(dataset))
    assert not restored_dataset._handles
    np.testing.assert_array_equal(restored_dataset[0]["actions"], sample["actions"])
    raw_dataset.close()
    dataset.close()
    restored_dataset.close()


def test_contract_rejects_changed_sources_and_wrong_horizon(prepared):
    raw, output, _, _, _ = prepared
    with pytest.raises(ValueError, match="horizon 63"):
        _dataset(prepared, chunk_size=64)
    with pytest.raises(ValueError, match="augmentation"):
        _dataset(prepared, use_image_aug=True)
    with pytest.raises(ValueError, match="validation split"):
        _dataset(prepared, representative_order=True)
    with h5py.File(source_path(raw, DIRECTIONS[0]), "r+") as handle:
        handle.attrs["test_changed"] = True
    with pytest.raises(ValueError, match="Source changed"):
        _dataset(prepared)
    with pytest.raises(ValueError, match="outside the raw"):
        prepare_hanoi_data(raw, raw / "derived")


def test_short_validation_prefix_covers_both_directions_episodes_and_time(prepared):
    ordinary = _dataset(prepared, split="val")
    representative = _dataset(prepared, split="val", representative_order=True)
    repeated = _dataset(prepared, split="val", representative_order=True)
    # The first ten entries visit all ten episode/direction groups. Three rounds
    # visit early, middle and late trajectory thirds in every group.
    first_groups = set(zip(representative.source_indices[:10], representative.episode_indices[:10], strict=True))
    assert first_groups == {(source, episode) for source in [0, 1] for episode in range(40, 45)}
    for source, episode in first_groups:
        selected = (representative.source_indices[:30] == source) & (representative.episode_indices[:30] == episode)
        rows = representative.row_indices[:30][selected] % 70
        assert len(rows) == 3 and np.ptp(rows) > 70 / 3
    np.testing.assert_array_equal(representative.row_indices, repeated.row_indices)
    original_rows = sorted(zip(ordinary.source_indices, ordinary.episode_indices, ordinary.row_indices, strict=True))
    reordered_rows = sorted(
        zip(representative.source_indices, representative.episode_indices, representative.row_indices, strict=True)
    )
    assert original_rows == reordered_rows


def test_common_anchor_helper_and_no_clipping():
    assert source_path("/tmp", "aaaa_to_cccc").name == "hanoi_wm_roundtrip_20260910_195558_AAAA_to_CCCC.h5"
    assert source_path("/tmp", "cccc_to_aaaa").name == "hanoi_wm_roundtrip_20260910_195558_CCCC_to_AAAA.h5"
    absolute = np.array([[10, 20, 30, 0], [11, 22, 33, 1], [13, 24, 35, 0]], dtype=np.float32)
    chunk = action_chunk(absolute, 1, np.array([9, 18, 27]), horizon=4)
    np.testing.assert_array_equal(chunk, [[2, 4, 6, 1], [4, 6, 8, 0], [4, 6, 8, 0], [4, 6, 8, 0]])
    stats = {"actions_min": [0, 0, 0, 0], "actions_max": [1, 1, 1, 1]}
    assert normalize(np.array([2, 0, 0, 1]), stats, "actions")[0] == 3
