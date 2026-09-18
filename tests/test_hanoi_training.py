"""Real CPU DCP/RNG state tests; only the CUDA RNG and callback base are mocked."""

import importlib.util
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp

_checkpoint_path = Path(__file__).resolve().parents[1] / "cosmos_policy/utils/hanoi_checkpoint.py"
_checkpoint_spec = importlib.util.spec_from_file_location("_hanoi_checkpoint_test", _checkpoint_path)
_checkpoint_module = importlib.util.module_from_spec(_checkpoint_spec)
_checkpoint_spec.loader.exec_module(_checkpoint_module)
latest_complete_checkpoint = _checkpoint_module.latest_complete_checkpoint
validate_dcp_parts = _checkpoint_module.validate_dcp_parts


@pytest.fixture
def monitor(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "cosmos_policy.utils.hanoi_checkpoint", _checkpoint_module)
    callback = types.ModuleType("cosmos_policy._src.imaginaire.utils.callback")
    callback.Callback = type("Callback", (), {})
    distributed = types.ModuleType("cosmos_policy._src.imaginaire.utils.distributed")
    distributed.is_rank0 = lambda: True
    monkeypatch.setitem(sys.modules, callback.__name__, callback)
    monkeypatch.setitem(sys.modules, distributed.__name__, distributed)
    # Ensure an earlier import of the real distributed module cannot bypass the mock.
    import cosmos_policy._src.imaginaire.utils as utils

    monkeypatch.setattr(utils, "distributed", distributed, raising=False)
    path = Path(__file__).resolve().parents[1] / "cosmos_policy/utils/hanoi_training.py"
    spec = importlib.util.spec_from_file_location("_hanoi_training_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cuda_generator = torch.Generator().manual_seed(97)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [cuda_generator.get_state()])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: cuda_generator.set_state(states[0]))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 200)
    instance = module.HanoiTrainingMonitor()
    instance.config = types.SimpleNamespace(
        job=types.SimpleNamespace(path_local=tmp_path), trainer=types.SimpleNamespace(logging_iter=1)
    )
    instance.trainer = types.SimpleNamespace(stop_requested=False)
    instance.cuda_generator = cuda_generator
    return instance


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "checkpoints/iter_000000002"
    for part in ("model", "optim", "scheduler", "trainer"):
        dcp.save({"state": torch.arange(5)}, checkpoint_id=path / part)
    (tmp_path / "checkpoints/latest_checkpoint.txt").write_text(path.name)
    (tmp_path / "data_order.json").write_text(
        json.dumps({"format_version": 1, "dataset_identity": {"source": "fixture"}})
    )
    return path


def snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()[0]


def assert_snapshot(expected):
    actual = snapshot()
    assert actual[0] == expected[0]
    assert actual[1][0] == expected[1][0] and actual[1][2:] == expected[1][2:]
    np.testing.assert_array_equal(actual[1][1], expected[1][1])
    assert torch.equal(actual[2], expected[2]) and torch.equal(actual[3], expected[3])


def draws(cuda_generator):
    return random.random(), np.random.uniform(), torch.rand(()).item(), torch.rand((), generator=cuda_generator).item()


def test_checkpoint_rng_roundtrip_preserves_next_draws(monitor, checkpoint):
    random.seed(42)
    np.random.seed(45)
    torch.manual_seed(48)
    before = snapshot()
    monitor.on_save_checkpoint_start(None, iteration=2)
    monitor.on_save_checkpoint_success(iteration=2)
    assert_snapshot(before)
    assert latest_complete_checkpoint(checkpoint.parents[1]) == checkpoint
    expected = draws(monitor.cuda_generator)
    draws(monitor.cuda_generator)
    monitor.on_load_checkpoint_end(None, iteration=2, checkpoint_path=checkpoint)
    assert draws(monitor.cuda_generator) == expected
    events = [json.loads(line) for line in (checkpoint.parents[1] / "metrics.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["checkpoint_saved", "rng_restored"]


def test_malformed_late_rng_field_does_not_partially_restore(monitor, checkpoint):
    monitor.on_save_checkpoint_start(None, iteration=2)
    monitor.on_save_checkpoint_success(iteration=2)
    draws(monitor.cuda_generator)
    before = snapshot()
    path = checkpoint / "hanoi_rng.json"
    state = json.loads(path.read_text())
    state["torch_cuda"] = [[1, 2, 3]]
    path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="Invalid Hanoi checkpoint RNG"):
        monitor.on_load_checkpoint_end(None, iteration=2, checkpoint_path=checkpoint)
    assert_snapshot(before)


def test_missing_rng_and_incomplete_shards_refuse_resume(monitor, checkpoint):
    before = snapshot()
    with pytest.raises(RuntimeError, match="RNG sidecar"):
        monitor.on_load_checkpoint_end(None, iteration=2, checkpoint_path=checkpoint)
    assert_snapshot(before)
    monitor.on_save_checkpoint_start(None, iteration=2)
    shard = next((checkpoint / "optim").glob("*.distcp"))
    shard.write_bytes(b"truncated")
    monitor.on_save_checkpoint_success(iteration=2)
    assert not (checkpoint / "hanoi_rng.json").exists()
    with pytest.raises(RuntimeError, match="Incomplete Hanoi checkpoint component"):
        latest_complete_checkpoint(checkpoint.parents[1])
    with pytest.raises(RuntimeError, match="truncated"):
        validate_dcp_parts(checkpoint)
    events = (checkpoint.parents[1] / "metrics.jsonl").read_text()
    assert "checkpoint_incomplete" in events and "checkpoint_saved" not in events


def test_snapshot_mismatch_data_contract_and_wrong_iteration_are_refused(monitor, checkpoint):
    monitor.on_save_checkpoint_start(None, iteration=1)
    monitor.on_save_checkpoint_success(iteration=2)
    assert not (checkpoint / "hanoi_rng.json").exists()
    monitor.on_save_checkpoint_start(None, iteration=2)
    monitor.on_save_checkpoint_success(iteration=2)
    (checkpoint.parents[1] / "data_order.json").unlink()
    with pytest.raises(RuntimeError, match="data-order contract"):
        latest_complete_checkpoint(checkpoint.parents[1])
    with pytest.raises(RuntimeError, match="iteration"):
        monitor.on_load_checkpoint_end(None, iteration=3, checkpoint_path=checkpoint)
    (checkpoint.parents[1] / "checkpoints/latest_checkpoint.txt").write_text("../../other/iter_000000002")
    with pytest.raises(RuntimeError, match="belong to this Hanoi run"):
        latest_complete_checkpoint(checkpoint.parents[1])


def test_metrics_are_weighted_and_stop_before_overrunning_step(monitor):
    metric = "demo_sample_action_l1_loss"
    first, second = {metric: torch.tensor(1.0)}, {metric: torch.tensor(4.0)}
    monitor.on_validation_start(None, None)
    monitor.on_validation_step_end(None, {"actions": torch.zeros(2, 63, 4)}, first, torch.tensor(2.0))
    monitor.on_validation_step_end(None, {"actions": torch.zeros(1, 63, 4)}, second, torch.tensor(5.0))
    monitor.on_validation_end(None, iteration=10)
    report = json.loads((Path(monitor.config.job.path_local) / "metrics.jsonl").read_text())
    assert report["samples"] == 3 and report["metrics"]["loss"] == 3 and report["metrics"][metric] == 2
    with pytest.raises(FloatingPointError, match="objective"):
        monitor._metrics(first, torch.tensor(float("nan")))
    with pytest.raises(FloatingPointError, match="missing or non-finite"):
        monitor._metrics({metric: torch.tensor(float("inf"))}, torch.tensor(1.0))
    with pytest.raises(FloatingPointError, match="missing or non-finite"):
        monitor._metrics({}, torch.tensor(1.0))
    # Budget is checked only by the optimizer-step callback, never midway
    # through gradient accumulation; predict another equally long step.
    import time

    monitor.stop_at = time.time() + 5
    monitor._last_step_at = time.time() - 10
    monitor.on_training_step_end(None, {}, {}, torch.tensor(0.0), iteration=10)
    assert monitor.trainer.stop_requested


def test_preload_hook_rejects_wrong_model_before_upstream_restore(monitor, checkpoint):
    monitor.on_save_checkpoint_start(None, iteration=2)
    monitor.on_save_checkpoint_success(iteration=2)
    monitor.trainer.checkpointer = types.SimpleNamespace(
        keys_to_resume_during_load=lambda: ({"model", "optim", "scheduler", "trainer"}, str(checkpoint))
    )
    # Fixture stores only "state". A real policy expects different tensor names;
    # the pre-load hook must refuse rather than accepting DCP partial loading.
    with pytest.raises(RuntimeError, match="missing|mismatch|Missing"):
        monitor.on_load_checkpoint_start(torch.nn.Linear(2, 1))
    monitor.trainer.checkpointer.keys_to_resume_during_load = lambda: (set(), None)
    monitor.on_load_checkpoint_start(torch.nn.Linear(2, 1))


def test_preload_hook_rejects_missing_optimizer_master_before_restore(monitor, checkpoint):
    model = torch.nn.Module()
    model.net = torch.nn.Linear(2, 1, dtype=torch.bfloat16)
    dcp.save(model.state_dict(), checkpoint_id=checkpoint / "model")
    optimizer_state = {
        f"state.net.{parameter}.{field}": torch.zeros(shape, dtype=torch.float32)
        for parameter, shape in (("weight", (1, 2)), ("bias", (1,)))
        for field in ("exp_avg", "exp_avg_sq")
    }
    dcp.save(optimizer_state, checkpoint_id=checkpoint / "optim")
    monitor.on_save_checkpoint_start(model, iteration=2)
    monitor.on_save_checkpoint_success(iteration=2)
    monitor.trainer.checkpointer = types.SimpleNamespace(
        keys_to_resume_during_load=lambda: ({"model", "optim", "scheduler", "trainer"}, str(checkpoint))
    )
    original = model.net.weight.detach().clone()
    with pytest.raises(RuntimeError, match="missing=.*master_param"):
        monitor.on_load_checkpoint_start(model)
    torch.testing.assert_close(model.net.weight, original)
