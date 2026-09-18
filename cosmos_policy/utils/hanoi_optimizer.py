"""Hanoi's FusedAdam with lossless FP32 optimizer checkpoint state.

The stock optimizer keeps master weights outside its state dictionary, and its
loader casts Adam moments through the BF16 parameter dtype. The update itself is
unchanged here; only serialization/restoration of FP32 state is specialized.
"""

import torch

from cosmos_policy._src.predict2.utils.fused_adam_dtensor import FusedAdam


class HanoiFusedAdam(FusedAdam):
    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)
        if self.master_weights and self.param_groups_master is not None:
            for group, masters in zip(self.param_groups, self.param_groups_master):
                for parameter, master in zip(group["params"], masters["params"]):
                    state = self.state.get(parameter, {})
                    if "exp_avg" in state and "exp_avg_sq" in state:
                        # Torch DCP's unflattening enumerates live state keys.
                        # Add the alias only after the base initialized moments;
                        # its first-step initialization relies on an empty state.
                        state["master_param"] = master
        return result

    def state_dict(self):
        state_dict = super().state_dict()
        # Copy only mappings. Tensor storage is shared with the live optimizer,
        # so saving does not allocate another model-sized master-weight copy.
        state_dict["state"] = {key: dict(value) for key, value in state_dict["state"].items()}
        if self.master_weights and self.param_groups_master is not None:
            for group, saved_group, masters in zip(
                self.param_groups, state_dict["param_groups"], self.param_groups_master
            ):
                for parameter, saved_id, master in zip(group["params"], saved_group["params"], masters["params"]):
                    if self.state.get(parameter):
                        state_dict["state"][saved_id]["master_param"] = master
        return state_dict

    def load_state_dict(self, state_dict):
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups) or any(
            len(saved["params"]) != len(current["params"]) for saved, current in zip(saved_groups, self.param_groups)
        ):
            raise ValueError("Hanoi optimizer checkpoint parameter groups do not match")

        required = ("exp_avg", "exp_avg_sq", "master_param") if self.master_weights else ("exp_avg", "exp_avg_sq")
        full_precision = {}
        stripped_state = {key: dict(value) for key, value in state_dict["state"].items()}
        # Validate before changing the optimizer. DCP passes FQN keys here;
        # ordinary torch optimizer checkpoints pass numeric parameter IDs.
        for saved, current in zip(saved_groups, self.param_groups):
            for saved_id, parameter in zip(saved["params"], current["params"]):
                recorded = state_dict["state"].get(saved_id, {})
                if not recorded:
                    continue
                tensors = {}
                for name in required:
                    value = recorded.get(name)
                    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
                        raise ValueError(f"Hanoi optimizer resume requires original FP32 {name} for {saved_id}")
                    if value.shape != parameter.shape:
                        raise ValueError(f"Hanoi optimizer {name} shape does not match parameter {saved_id}")
                    tensors[name] = value.detach().to(device=parameter.device)
                    stripped_state[saved_id].pop(name)
                full_precision[parameter] = tensors

        # Let the base loader restore parameter groups, steps, device handling,
        # and hooks, but prevent torch.Optimizer from rounding these tensors to
        # the parameter dtype before the base casts them back to FP32.
        super().load_state_dict({**state_dict, "state": stripped_state})
        for parameter, values in full_precision.items():
            self.state[parameter].update(values)

        self.param_groups_master = None
        if self.master_weights and full_precision:
            self.param_groups_master = [
                {
                    "params": [
                        full_precision[parameter]["master_param"]
                        if parameter in full_precision
                        else parameter.detach().float().clone()
                        for parameter in group["params"]
                    ]
                }
                for group in self.param_groups
            ]


def get_hanoi_optimizer(model, lr, weight_decay, optim_type="fusedadam", **kwargs):
    """Use the stock parameter grouping and FusedAdam hyperparameters."""
    from omegaconf import ListConfig

    from cosmos_policy._src.predict2.utils.optim_instantiate import get_regular_param_group

    if optim_type != "fusedadam":
        raise ValueError("The Hanoi optimizer configuration requires fusedadam")
    decay, no_decay = get_regular_param_group(model)
    kwargs = {key: list(value) if isinstance(value, ListConfig) else value for key, value in kwargs.items()}
    return HanoiFusedAdam([{"params": decay + no_decay, "lr": lr, "weight_decay": weight_decay}], **kwargs)
