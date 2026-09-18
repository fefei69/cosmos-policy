"""Exercise CPU DCP consolidation and complete inference weight restoration."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp

from cosmos_policy.experiments.robot.hanoi.policy import load_hanoi_weights
from examples.hanoi.export_checkpoint import export_checkpoint


@pytest.fixture
def checkpoint_validators():
    # The helper itself needs only Torch; avoid eager imports from the package's
    # unrelated GPU training/logging utilities in a minimal CPU environment.
    path = Path(__file__).parents[1] / "cosmos_policy/utils/hanoi_checkpoint.py"
    spec = importlib.util.spec_from_file_location("_hanoi_checkpoint_metadata_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def validate_model_metadata(checkpoint_validators):
    return checkpoint_validators.validate_model_metadata


def test_cpu_dcp_export_roundtrip_and_no_overwrite(tmp_path):
    iteration = tmp_path / "iter_000000010"
    state = {
        "net.weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        "net.bias": torch.tensor([1, 2, 3], dtype=torch.bfloat16),
    }
    dcp.save(state, checkpoint_id=iteration / "model")
    output = tmp_path / "export" / "hanoi.pt"
    report = export_checkpoint(iteration, output)
    restored = torch.load(output, map_location="cpu", weights_only=True, mmap=True)
    assert report["tensor_count"] == 2
    assert report["element_count"] == 15
    assert (iteration / "model" / ".metadata").is_file()
    for key, tensor in state.items():
        torch.testing.assert_close(restored[key], tensor, rtol=0, atol=0)
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        export_checkpoint(iteration, output)
    assert output.read_bytes() == original
    assert not list(output.parent.glob(".hanoi.pt.*"))


def test_export_rejects_optimizer_and_nonfinite_spotcheck(tmp_path):
    optimizer = tmp_path / "optimizer"
    dcp.save({"optimizer.exp_avg": torch.zeros(3)}, checkpoint_id=optimizer)
    output = tmp_path / "hanoi.pt"
    with pytest.raises(ValueError, match="model weights only"):
        export_checkpoint(optimizer, output)
    source = tmp_path / "model"
    dcp.save({"net.weight": torch.tensor([float("nan")])}, checkpoint_id=source)
    with pytest.raises(ValueError, match="Non-finite"):
        export_checkpoint(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".hanoi.pt.*"))


def test_load_requires_all_weights_and_shapes_before_mutating_model():
    net = torch.nn.Linear(4, 3)
    model = SimpleNamespace(net=net)
    original = net.weight.detach().clone()
    with pytest.raises(ValueError, match="missing=.*bias"):
        load_hanoi_weights(model, {"net.weight": torch.ones_like(net.weight)})
    torch.testing.assert_close(net.weight, original)
    with pytest.raises(ValueError, match="mismatched_shapes=.*weight"):
        load_hanoi_weights(model, {"net.weight": torch.ones(4, 3), "net.bias": torch.zeros(3)})
    torch.testing.assert_close(net.weight, original)
    with pytest.raises(ValueError, match="unexpected=.*wrong"):
        load_hanoi_weights(
            model, {"net.weight": torch.ones(3, 4), "net.bias": torch.zeros(3), "net.wrong": torch.zeros(1)}
        )
    load_hanoi_weights(
        model,
        {"net.weight": torch.ones(3, 4), "net.bias": torch.zeros(3), "net.optional._extra_state": None},
    )
    torch.testing.assert_close(net.weight, torch.ones(3, 4))
    torch.testing.assert_close(net.bias, torch.zeros(3))


def test_dcp_model_metadata_requires_complete_exact_schema(tmp_path, validate_model_metadata):
    expected = {"net.weight": torch.ones(3, 4), "net.bias": torch.zeros(3)}
    source = tmp_path / "complete"
    dcp.save({**expected, "net.block._extra_state": None}, checkpoint_id=source)
    assert validate_model_metadata(source, expected) == {"validated_model_tensors": 2}
    cases = (
        ("missing", {"net.weight": expected["net.weight"]}, "missing=.*net.bias"),
        ("extra", {**expected, "net.unexpected": torch.zeros(1)}, "unexpected=.*net.unexpected"),
        ("shape", {**expected, "net.weight": torch.ones(4, 3)}, "mismatched_shapes=.*net.weight"),
        ("dtype", {key: value.bfloat16() for key, value in expected.items()}, "mismatched_dtypes=.*net"),
    )
    for name, state, message in cases:
        checkpoint = tmp_path / name
        dcp.save(state, checkpoint_id=checkpoint)
        with pytest.raises(RuntimeError, match=message):
            validate_model_metadata(checkpoint, expected)
    # A read-only metadata check cannot overwrite any existing model tensors.
    torch.testing.assert_close(expected["net.weight"], torch.ones(3, 4))


def test_extra_state_exception_does_not_hide_real_parameter_names(tmp_path, validate_model_metadata):
    net = torch.nn.Linear(4, 3)
    net.register_parameter("trainable_extra_state", torch.nn.Parameter(torch.ones(1)))
    partial = {"net.weight": net.weight.detach().clone(), "net.bias": net.bias.detach().clone()}
    with pytest.raises(ValueError, match="missing=.*trainable_extra_state"):
        load_hanoi_weights(SimpleNamespace(net=net), partial)
    dcp.save(partial, checkpoint_id=tmp_path)
    expected = {f"net.{key}": value for key, value in net.state_dict().items()}
    with pytest.raises(RuntimeError, match="missing=.*trainable_extra_state"):
        validate_model_metadata(tmp_path, expected)


def test_optimizer_metadata_requires_fp32_masters_and_moments_before_restore(tmp_path, checkpoint_validators):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    net = torch.nn.Module()
    net.add_module("projection", checkpoint_wrapper(torch.nn.Linear(3, 2, dtype=torch.bfloat16)))
    net.register_parameter("frozen", torch.nn.Parameter(torch.zeros(4), requires_grad=False))
    original = net.projection.weight.detach().clone()
    state = {
        f"state.net.projection.{parameter}.{field}": torch.zeros(shape, dtype=torch.float32)
        for parameter, shape in (("weight", (2, 3)), ("bias", (2,)))
        for field in ("master_param", "exp_avg", "exp_avg_sq")
    }
    source = tmp_path / "complete_optimizer"
    dcp.save(state, checkpoint_id=source)
    assert checkpoint_validators.validate_optimizer_metadata(source, net) == {"validated_optimizer_parameters": 2}
    cases = (
        ("missing_master", "state.net.projection.weight.master_param", None, "missing=.*master_param"),
        ("missing_moment", "state.net.projection.bias.exp_avg_sq", None, "missing=.*exp_avg_sq"),
        (
            "rounded_moment",
            "state.net.projection.weight.exp_avg",
            torch.zeros((2, 3), dtype=torch.bfloat16),
            "mismatched_dtypes=.*exp_avg",
        ),
        (
            "wrong_shape",
            "state.net.projection.weight.master_param",
            torch.zeros((3, 2)),
            "mismatched_shapes=.*master_param",
        ),
    )
    for name, key, replacement, message in cases:
        recorded = dict(state)
        if replacement is None:
            recorded.pop(key)
        else:
            recorded[key] = replacement
        directory = tmp_path / name
        dcp.save(recorded, checkpoint_id=directory)
        with pytest.raises(RuntimeError, match=message):
            checkpoint_validators.validate_optimizer_metadata(directory, net)
        torch.testing.assert_close(net.projection.weight, original)
