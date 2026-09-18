"""Exercise real FusedAdam Python logic and DCP, replacing only CUDA kernels."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)


@pytest.fixture
def optimizer_classes(monkeypatch):
    # The original optimizer class and the Hanoi subclass are both real. Small
    # CPU tensors and an Adam reference kernel replace CUDA allocation/launch.
    def reference_kernel(kernel, overflow, lists, lr, beta1, beta2, eps, step, adam_w, correction, decay, inv_scale):
        assert lr.device == lists[1][0].device
        gradients, parameters, moments, variances, masters = lists
        learning_rate, count = float(lr), int(step)
        with torch.no_grad():
            for gradient, parameter, moment, variance, master in zip(
                gradients, parameters, moments, variances, masters
            ):
                gradient = gradient.float() * inv_scale
                moment.mul_(beta1).add_(gradient, alpha=1 - beta1)
                variance.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                if adam_w:
                    master.mul_(1 - learning_rate * decay)
                denominator = variance.sqrt() / ((1 - beta2**count) ** 0.5 if correction else 1)
                denominator.add_(eps)
                master.addcdiv_(moment, denominator, value=-learning_rate / (1 - beta1**count if correction else 1))
                parameter.copy_(master)

    te = types.ModuleType("transformer_engine")
    te.pytorch = types.SimpleNamespace(optimizers=types.SimpleNamespace(multi_tensor_applier=reference_kernel))
    tex = types.ModuleType("transformer_engine_torch")
    tex.multi_tensor_adam = tex.multi_tensor_adam_capturable = tex.multi_tensor_adam_capturable_master = object()
    distributed = types.ModuleType("cosmos_policy._src.imaginaire.utils.distributed")
    distributed.get_rank = lambda: 0
    distributed.broadcast = lambda *args: None
    log = types.ModuleType("cosmos_policy._src.imaginaire.utils.log")
    log.warning = lambda *args, **kwargs: None
    misc = types.ModuleType("cosmos_policy._src.imaginaire.utils.misc")
    misc.get_local_tensor_if_DTensor = lambda tensor: tensor
    for module in (te, tex, distributed, log, misc):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    import cosmos_policy._src.imaginaire.utils as utils

    monkeypatch.setattr(utils, "distributed", distributed, raising=False)
    monkeypatch.setattr(utils, "log", log, raising=False)
    original_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    root = Path(__file__).resolve().parents[1]
    base_name = "cosmos_policy._src.predict2.utils.fused_adam_dtensor"
    base_spec = importlib.util.spec_from_file_location(
        base_name, root / "cosmos_policy/_src/predict2/utils/fused_adam_dtensor.py"
    )
    base = importlib.util.module_from_spec(base_spec)
    monkeypatch.setitem(sys.modules, base_name, base)
    base_spec.loader.exec_module(base)
    spec = importlib.util.spec_from_file_location(
        "_hanoi_optimizer_test", root / "cosmos_policy/utils/hanoi_optimizer.py"
    )
    hanoi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hanoi)
    return base.FusedAdam, hanoi.HanoiFusedAdam


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(3, 2, dtype=torch.bfloat16)


def make_optimizer(optimizer_class, model):
    return optimizer_class(model.parameters(), lr=0.00317, betas=(0.9, 0.99), master_weights=True, capturable=True)


def update(model, optimizer, step):
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.2 + 0.031 * step)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def assert_same_state(first, second):
    a, b = first.state_dict(), second.state_dict()
    for state_a, state_b in zip(a["state"].values(), b["state"].values()):
        assert state_a.keys() == state_b.keys()
        for name in state_a:
            torch.testing.assert_close(state_a[name], state_b[name], rtol=0, atol=0)


def test_serialization_does_not_change_base_updates_or_first_step_initialization(optimizer_classes):
    base_class, hanoi_class = optimizer_classes
    original, adapted = Policy(), Policy()
    adapted.load_state_dict(original.state_dict())
    base, hanoi = make_optimizer(base_class, original), make_optimizer(hanoi_class, adapted)
    assert hanoi.state_dict()["state"] == {} and not hanoi.state and hanoi.param_groups_master is None
    for step in range(5):
        update(original, base, step)
        update(adapted, hanoi, step)
        saved = hanoi.state_dict()
        for index, (a, b) in enumerate(zip(original.parameters(), adapted.parameters())):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            for name in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(base.state[a][name], hanoi.state[b][name], rtol=0, atol=0)
            assert set(hanoi.state[b]) == {"exp_avg", "exp_avg_sq", "master_param"}
            master = hanoi.param_groups_master[0]["params"][index]
            assert hanoi.state[b]["master_param"].data_ptr() == master.data_ptr()
            assert saved["state"][index]["master_param"].data_ptr() == master.data_ptr()
            torch.testing.assert_close(base.param_groups_master[0]["params"][index], master, rtol=0, atol=0)


def test_real_dcp_roundtrip_restores_fp32_masters_moments_and_next_updates(optimizer_classes, tmp_path):
    _, optimizer_class = optimizer_classes
    model = Policy()
    optimizer = make_optimizer(optimizer_class, model)
    for step in range(3):
        update(model, optimizer, step)
    # Deliberately preserve mantissa bits that a BF16 roundtrip would destroy.
    for state in optimizer.state.values():
        state["exp_avg"].add_(0.00012345)
        state["exp_avg_sq"].add_(0.00003123)
    for master in optimizer.param_groups_master[0]["params"]:
        master.add_(0.00006123)
        assert not torch.equal(master, master.bfloat16().float())
    options = StateDictOptions(flatten_optimizer_state_dict=True)
    serialized = get_optimizer_state_dict(model, optimizer, options=options)
    for parameter_name, parameter in model.named_parameters():
        for name in ("master_param", "exp_avg", "exp_avg_sq"):
            value = serialized[f"state.{parameter_name}.{name}"]
            assert value.dtype == torch.float32 and value.shape == parameter.shape
            assert not torch.equal(value, value.bfloat16().float())
    dcp.save(serialized, checkpoint_id=tmp_path / "optim")
    helper_path = Path(__file__).resolve().parents[1] / "cosmos_policy/utils/hanoi_checkpoint.py"
    helper_spec = importlib.util.spec_from_file_location("_hanoi_optimizer_checkpoint_test", helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    assert helper.validate_optimizer_metadata(tmp_path / "optim", model.net) == {"validated_optimizer_parameters": 2}

    resumed_model = Policy()
    resumed_model.load_state_dict(model.state_dict())
    resumed = make_optimizer(optimizer_class, resumed_model)
    before_init = {name: value.clone() for name, value in resumed_model.state_dict().items()}
    # This is the stock OptimizerWrapper load sequence. It invokes DCP's
    # _init_optim_state dummy step on the previously untouched optimizer.
    destination = get_optimizer_state_dict(resumed_model, resumed, options=options)
    assert resumed.param_groups_master is not None
    for name, value in resumed_model.state_dict().items():
        torch.testing.assert_close(value, before_init[name], rtol=0, atol=0)
    dcp.load(destination, checkpoint_id=tmp_path / "optim")
    set_optimizer_state_dict(resumed_model, resumed, destination, options=options)
    assert_same_state(optimizer, resumed)
    assert int(resumed.param_groups[0]["step"]) == int(optimizer.param_groups[0]["step"])
    for step in range(3, 6):
        update(model, optimizer, step)
        update(resumed_model, resumed, step)
        assert_same_state(optimizer, resumed)
        for a, b in zip(model.parameters(), resumed_model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["missing_master", "rounded_moment", "wrong_shape"])
def test_incomplete_or_rounded_optimizer_state_is_rejected_before_mutation(optimizer_classes, invalid):
    _, optimizer_class = optimizer_classes
    model = Policy()
    optimizer = make_optimizer(optimizer_class, model)
    update(model, optimizer, 0)
    state = optimizer.state_dict()
    if invalid == "missing_master":
        state["state"][0].pop("master_param")
    elif invalid == "rounded_moment":
        state["state"][0]["exp_avg"] = state["state"][0]["exp_avg"].bfloat16()
    else:
        state["state"][0]["master_param"] = torch.zeros(7, dtype=torch.float32)
    original_master = optimizer.param_groups_master[0]["params"][0].clone()
    with pytest.raises(ValueError, match="FP32|shape"):
        optimizer.load_state_dict(state)
    torch.testing.assert_close(optimizer.param_groups_master[0]["params"][0], original_master, rtol=0, atol=0)
