"""CPU contract tests; the model and T5 loader are replaced, not imported."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from cosmos_policy.experiments.robot.hanoi.policy import (
    HanoiInferenceConfig,
    make_observation,
    to_absolute_actions,
    validate_config,
    validate_dataset_stats,
)
from cosmos_policy.experiments.robot.hanoi.run_hanoi_eval import chunk_metrics, read_sample, select_indices


@pytest.fixture
def cosmos_utils(monkeypatch):
    # GPU-only model dependencies are irrelevant to packing, cache policy, and
    # action extraction. Keep the real torch/numpy and image transformations.
    for name, symbol in (
        ("cosmos_policy._src.predict2.inference.get_t5_emb", "get_text_embedding"),
        ("cosmos_policy._src.predict2.utils.model_loader", "load_model_from_checkpoint"),
    ):
        module = ModuleType(name)

        def unexpected_call(*args, **kwargs):
            raise AssertionError("A model/text encoder must not be loaded in a CPU contract test")

        setattr(module, symbol, unexpected_call)
        monkeypatch.setitem(sys.modules, name, module)
    # Load the real lightweight array helpers without the utils package's eager
    # checkpoint/logging imports (those require the full training environment).
    helper_path = Path(__file__).parents[1] / "cosmos_policy/utils/utils.py"
    helper_spec = importlib.util.spec_from_file_location("cosmos_policy.utils.utils", helper_path)
    helpers = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helpers)
    monkeypatch.setitem(sys.modules, "cosmos_policy.utils.utils", helpers)
    path = Path(__file__).parents[1] / "cosmos_policy/experiments/robot/cosmos_utils.py"
    spec = importlib.util.spec_from_file_location("hanoi_test_cosmos_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    return module


def test_measured_only_observation_and_anchor_inverse():
    image = np.zeros((224, 224, 3), np.uint8)
    state = np.array([0.2, -0.1, 0.3, 0.04], np.float32)
    observation = make_observation(image, state)
    np.testing.assert_array_equal(observation["proprio"], state)
    assert set(observation) == {"primary_image", "proprio"}
    with pytest.raises(ValueError, match="exclude velocity"):
        make_observation(image, np.append(state, 1))
    for legacy_size in (7, 8):
        with pytest.raises(ValueError, match="exclude velocity"):
            make_observation(image, np.zeros(legacy_size, np.float32))
    with pytest.raises(ValueError, match="do not crop"):
        make_observation(np.zeros((480, 640, 3), np.uint8), state)
    relative = np.zeros((63, 4), np.float32)
    relative[:, 0] = np.arange(63) * 0.001
    relative[:, 3] = np.where(np.arange(63) % 2, 0.49, 0.51)
    original = relative.copy()
    result = to_absolute_actions(relative, state[:3])
    np.testing.assert_allclose(result[:, :3], original[:, :3] + state[None, :3])
    np.testing.assert_array_equal(result[:, 3], np.arange(63) % 2 == 0)
    np.testing.assert_array_equal(relative, original)


def test_stats_and_configuration_fail_early():
    cfg = HanoiInferenceConfig("checkpoint.pt", "stats.json", "cache.pkl")
    validate_config(cfg)
    cfg.trained_with_image_aug = True
    with pytest.raises(ValueError, match="trained_with_image_aug"):
        validate_config(cfg)
    stats = {"actions_min": [0] * 4, "actions_max": [1] * 4, "proprio_min": [0] * 4, "proprio_max": [1] * 4}
    validate_dataset_stats(stats)
    stats["proprio_max"][0] = 0
    with pytest.raises(ValueError, match="positive span"):
        validate_dataset_stats(stats)


def test_hanoi_cache_miss_refuses_encoder(cosmos_utils):
    with pytest.raises(KeyError, match="loading T5.*disabled"):
        cosmos_utils.get_t5_embedding_from_cache("missing instruction", allow_compute=False)
    cfg = HanoiInferenceConfig("checkpoint.pt", "stats.json", "cache.pkl")
    with pytest.raises(KeyError, match="loading T5.*disabled"):
        cosmos_utils.get_action(cfg, None, {}, {}, "missing instruction")


@pytest.mark.parametrize(
    "suite,wrist_count,camera_count,latent_count,proprio_dim,action_dim,chunk_size",
    [("hanoi", 0, 1, 7, 4, 4, 63), ("hanoi", 0, 1, 7, 7, 4, 8), ("libero", 1, 1, 9, 9, 7, 16),
     ("robocasa", 1, 2, 11, 9, 7, 32), ("aloha", 2, 1, 11, 14, 14, 50)],
)
def test_camera_packing_and_actions(
    cosmos_utils, suite, wrist_count, camera_count, latent_count, proprio_dim, action_dim, chunk_size
):
    cfg = HanoiInferenceConfig("checkpoint.pt", "stats.json", "cache.pkl")
    cfg.suite, cfg.use_wrist_image, cfg.num_wrist_images = suite, wrist_count > 0, wrist_count
    cfg.num_third_person_images, cfg.action_dim, cfg.chunk_size = camera_count, action_dim, chunk_size
    stats = {
        "proprio_min": np.zeros(proprio_dim, np.float32), "proprio_max": np.ones(proprio_dim, np.float32),
        "actions_min": np.zeros(action_dim, np.float32), "actions_max": np.ones(action_dim, np.float32),
    }
    image = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    obs = {"primary_image": image, "proprio": np.full(proprio_dim, 0.5, np.float32)}
    if suite == "aloha":
        obs.update(left_wrist_image=image, right_wrist_image=image)
    elif wrist_count:
        obs["wrist_image"] = image
    if camera_count == 2:
        obs["secondary_image"] = image
    conditional = 2 + wrist_count + camera_count

    class Model:
        config = SimpleNamespace(state_t=latent_count, min_num_conditional_frames=conditional)

        def generate_samples_from_batch(self, data_batch, **kwargs):
            self.batch = data_batch
            assert data_batch["video"].shape == (1, 3, 1 + 4 * (latent_count - 1), 224, 224)
            assert data_batch["action_latent_idx"].item() == conditional
            assert data_batch["future_proprio_latent_idx"].item() == conditional + 1
            assert data_batch["value_latent_idx"].item() == latent_count - 1
            assert data_batch["proprio"].shape == (1, proprio_dim)
            torch.testing.assert_close(data_batch["proprio"], torch.zeros_like(data_batch["proprio"]))
            if not wrist_count:
                assert data_batch["current_wrist_image_latent_idx"].item() == -1
                assert data_batch["future_wrist_image_latent_idx"].item() == -1
                # Single-camera images remain byte-for-byte the stored RGB.
                np.testing.assert_array_equal(data_batch["video"][0, :, 5].permute(1, 2, 0), image)
            latent = torch.zeros((1, 16, latent_count, 28, 28))
            return latent, latent.clone()

    model = Model()
    output = cosmos_utils.get_action(
        cfg, model, stats, obs, torch.zeros((1, 512, 1024)),
        generate_future_state_and_value_in_parallel=False,
    )
    assert np.asarray(output["actions"]).shape == (chunk_size, action_dim)
    np.testing.assert_allclose(output["actions"], 0.5)
    assert cosmos_utils.get_latent_indices_from_model_config(model) == (1, conditional - 1, conditional + 1, latent_count - 2)


def test_read_sample_preserves_same_row_and_episode_end(tmp_path):
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as handle:
        handle["ep_offset"] = [0, 3]
        handle["ep_len"] = [3, 3]
        handle["action_abs"] = np.arange(24, dtype=np.float32).reshape(6, 4)
        handle["proprio"] = np.arange(48, dtype=np.float32).reshape(6, 8)
        handle["pixels"] = np.zeros((6, 224, 224, 3), np.uint8)
        handle["command_monotonic_ns"] = np.full(6, 100_000_000, np.int64)
        handle["image_receipt_monotonic_ns"] = np.full(6, 90_000_000, np.int64)
        _, state, actions = read_sample(handle, row=1, episode=0)
        np.testing.assert_array_equal(state, handle["proprio"][1, [0, 1, 2, 6]])
        np.testing.assert_array_equal(actions[0], handle["action_abs"][1])
        np.testing.assert_array_equal(actions[1:], np.repeat(handle["action_abs"][2:3], 62, axis=0))
        handle["image_receipt_monotonic_ns"][1] = 0
        with pytest.raises(ValueError, match="ineligible"):
            read_sample(handle, row=1, episode=0)


def test_selection_balances_directions_and_reports_physical_units():
    index = {"source_index": np.repeat([0, 1, 0, 1], 4), "episode_index": np.repeat([40, 40, 41, 41], 4)}
    chosen = select_indices(index, samples=2, seed=3)
    assert set(index["source_index"][chosen]) == {0, 1}
    np.testing.assert_array_equal(chosen, select_indices(index, samples=2, seed=3))
    assert len(np.unique(select_indices(index, samples=100, seed=3))) == 16
    target = np.zeros((63, 4))
    predicted = target.copy()
    predicted[:, 0] = 0.003
    metrics = chunk_metrics(predicted, target)
    assert metrics["chunk63_xyz_l2_mm"] == pytest.approx(3)
    assert metrics["commit9_jaw_accuracy"] == 1
