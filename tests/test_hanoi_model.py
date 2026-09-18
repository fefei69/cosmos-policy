"""Exercise Hanoi loading/RNG contracts without importing GPU-only Cosmos ops."""

import importlib.util
import pickle
import random
import sys
import types
from pathlib import Path

import attrs
import numpy as np
import pytest
import torch


@pytest.fixture
def hanoi_model_module(monkeypatch):
    # Replace only the expensive base model. The actual Hanoi subclass and
    # checkpoint/RNG implementation are imported directly from the repository.
    base_module = types.ModuleType("cosmos_policy.models.policy_video2world_model")

    @attrs.define(slots=False)
    class BaseConfig:
        pass

    class BaseModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.net = torch.nn.Linear(2, 1)

        def on_train_start(self, memory_format=torch.preserve_format):
            self.net.to(dtype=torch.bfloat16)

        def training_step(self, data, iteration):
            noise = torch.tensor([torch.rand(()).item(), np.random.uniform(), random.random()])
            return {"noise": noise}, noise.sum()

    base_module.CosmosPolicyVideo2WorldConfig = BaseConfig
    base_module.CosmosPolicyVideo2WorldModel = BaseModel
    monkeypatch.setitem(sys.modules, base_module.__name__, base_module)
    path = Path(__file__).resolve().parents[1] / "cosmos_policy/models/hanoi_model.py"
    spec = importlib.util.spec_from_file_location("_hanoi_model_contract_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_checkpoint_is_loaded_before_optimizer_and_only_once(hanoi_model_module, tmp_path):
    module = hanoi_model_module
    path = tmp_path / "public.pt"
    torch.save({"net.weight": torch.full((1, 2), 3.0), "net.bias": torch.full((1,), 4.0)}, path)
    model = module.HanoiPolicyModel(module.HanoiPolicyVideo2WorldConfig(initial_checkpoint=str(path)))
    model.on_train_start()
    assert model.net.weight.dtype == torch.bfloat16
    assert torch.all(model.net.weight == 3) and torch.all(model.net.bias == 4)
    # A later hook must not overwrite values restored from the run's DCP state.
    with torch.no_grad():
        model.net.weight.fill_(9)
    model.on_train_start()
    assert torch.all(model.net.weight == 9)


def test_initial_checkpoint_missing_keys_and_shapes_fail_strictly(hanoi_model_module, tmp_path):
    module = hanoi_model_module
    path = tmp_path / "invalid.pt"
    torch.save({"net.weight": torch.ones(1, 2)}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        module.load_initial_policy_checkpoint(torch.nn.Linear(2, 1), str(path))
    torch.save({"net.weight": torch.ones(1, 3), "net.bias": torch.ones(1)}, path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        module.load_initial_policy_checkpoint(torch.nn.Linear(2, 1), str(path))
    torch.save({"net.weight": torch.ones(1, 2), "net.bias": torch.ones(1), "net.extra": torch.ones(1)}, path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        module.load_initial_policy_checkpoint(torch.nn.Linear(2, 1), str(path))


def test_inference_does_not_load_initial_public_weights(hanoi_model_module, monkeypatch):
    module = hanoi_model_module
    model = module.HanoiPolicyModel(module.HanoiPolicyVideo2WorldConfig(initial_checkpoint=""))

    def forbidden_load(*args, **kwargs):
        raise AssertionError("Inference tried to load the training initialization checkpoint")

    monkeypatch.setattr(torch, "load", forbidden_load)
    model.on_train_start()


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("serialized", [False, True])
def test_empty_attention_metadata_allows_backend_change_but_keeps_strict_weights(
    hanoi_model_module, tmp_path, wrapped, serialized
):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    block = torch.nn.Module()
    block.cross_attn = torch.nn.Module()
    block.cross_attn.q_proj = torch.nn.Linear(2, 2)
    block.cross_attn.attn_op = torch.nn.Identity()
    net = torch.nn.Module()
    net.blocks = torch.nn.ModuleList([checkpoint_wrapper(block) if wrapped else block])
    reference = {f"net.{key}": torch.full_like(value, 3.0) for key, value in net.state_dict().items()}
    extra_key = "net.blocks.0.cross_attn.attn_op._extra_state"
    path = tmp_path / "te_public.pt"
    empty_bytes = torch.tensor(list(pickle.dumps(None, protocol=4)), dtype=torch.uint8)
    extra = empty_bytes if serialized else None
    torch.save({**reference, extra_key: extra}, path)
    hanoi_model_module.load_initial_policy_checkpoint(net, str(path))
    assert all(torch.all(value == 3) for value in net.state_dict().values())

    # Preserve strict rejection of nonempty metadata and unrelated tensor state.
    fp8_bytes = torch.tensor(list(pickle.dumps({"fp8": True}, protocol=4)), dtype=torch.uint8)
    for value in (torch.ones(1), {"fp8": True}, fp8_bytes, empty_bytes.float(), empty_bytes.reshape(2, 2)):
        torch.save({**reference, extra_key: value}, path)
        with pytest.raises(RuntimeError, match="Unexpected key"):
            hanoi_model_module.load_initial_policy_checkpoint(net, str(path))
    torch.save({**reference, "net.unknown.attn_op._extra_state": extra}, path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        hanoi_model_module.load_initial_policy_checkpoint(net, str(path))
    missing_weight = {key: value for key, value in reference.items() if not key.endswith("q_proj.weight")}
    torch.save({**missing_weight, extra_key: extra}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        hanoi_model_module.load_initial_policy_checkpoint(net, str(path))


def test_expected_attention_extra_state_is_restored(hanoi_model_module, tmp_path):
    class StatefulAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = 1

        def get_extra_state(self):
            return {"value": self.value}

        def set_extra_state(self, state):
            self.value = state["value"]

    net = torch.nn.Module()
    net.cross_attn = torch.nn.Module()
    net.cross_attn.attn_op = StatefulAttention()
    path = tmp_path / "stateful.pt"
    torch.save({"net.cross_attn.attn_op._extra_state": {"value": 7}}, path)
    hanoi_model_module.load_initial_policy_checkpoint(net, str(path))
    assert net.cross_attn.attn_op.value == 7


def test_validation_repeats_noise_and_restores_torch_numpy_python_rng(hanoi_model_module, monkeypatch):
    module = hanoi_model_module
    model = module.HanoiPolicyModel(module.HanoiPolicyVideo2WorldConfig())
    original_fork = torch.random.fork_rng
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.random, "fork_rng", lambda devices: original_fork(devices=[]))
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)
    torch_state, numpy_state, python_state = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    first, _ = model.validation_step({}, 0)
    second, _ = model.validation_step({}, 100)
    assert torch.equal(first["noise"], second["noise"])
    assert torch.equal(torch.get_rng_state(), torch_state)
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_state[0] and restored_numpy[2:] == numpy_state[2:]
    np.testing.assert_array_equal(restored_numpy[1], numpy_state[1])
    assert random.getstate() == python_state
