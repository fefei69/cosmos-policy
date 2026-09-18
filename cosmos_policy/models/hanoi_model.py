"""Pretrained initialization and joint denoising validation for Hanoi."""

import pickle
import random
from collections.abc import Mapping

import attrs
import numpy as np
import torch

from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldConfig, CosmosPolicyVideo2WorldModel


def _empty_attention_metadata(value):
    # TE serializes unused FP8 bookkeeping as a uint8 pickle of None. Match
    # that exact payload without unpickling arbitrary checkpoint contents.
    serialized_none = pickle.dumps(None, protocol=4)
    return value is None or (
        isinstance(value, torch.Tensor)
        and value.dtype == torch.uint8
        and value.ndim == 1
        and value.numel() == len(serialized_none)
        and bytes(value.tolist()) == serialized_none
    )


@attrs.define(slots=False)
class HanoiPolicyVideo2WorldConfig(CosmosPolicyVideo2WorldConfig):
    # A consolidated public checkpoint initializes training only. Inference
    # loads its selected fine-tuned checkpoint through the normal model loader.
    initial_checkpoint: str = ""


def load_initial_policy_checkpoint(net: torch.nn.Module, path: str) -> None:
    """Strictly copy public net.* weights without a duplicate GPU allocation."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(state, Mapping) and "model" in state:
        state = state["model"]
    elif isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Initial policy checkpoint must contain a nonempty model state dictionary")
    if any(not isinstance(key, str) or not key.startswith("net.") for key in state):
        raise ValueError("Initial policy checkpoint must contain only net.* state entries")
    state = {key.removeprefix("net."): value for key, value in state.items()}
    expected = net.state_dict()
    omitted = []
    for key, value in list(state.items()):
        # Released TE attention operators serialize unused extra-state entries.
        # The configured MinimalA2AAttnOp has no corresponding bookkeeping. Only
        # omit empty metadata for an existing attention operator; learned weights,
        # nonempty state, and unknown modules remain subject to strict loading.
        if key not in expected and key.endswith(".attn_op._extra_state") and _empty_attention_metadata(value):
            try:
                net.get_submodule(key.removesuffix("._extra_state"))
            except AttributeError:
                continue
            del state[key]
            omitted.append(key)
    net.load_state_dict(state, strict=True, assign=False)
    if omitted:
        print(f"Omitted {len(omitted)} empty attention metadata entries from public initialization", flush=True)


class HanoiPolicyModel(CosmosPolicyVideo2WorldModel):
    def on_train_start(self, memory_format=torch.preserve_format):
        super().on_train_start(memory_format)
        if not getattr(self, "_hanoi_initial_checkpoint_loaded", False):
            if self.config.initial_checkpoint:
                load_initial_policy_checkpoint(self.net, self.config.initial_checkpoint)
                print(f"Loaded strict Hanoi initial policy weights: {self.config.initial_checkpoint}", flush=True)
            self._hanoi_initial_checkpoint_loaded = True
        # The trainer initializes its optimizer and restores any DCP checkpoint
        # after this hook. Never reload public weights in a later training hook.

    def _update_train_stats(self, data_batch):
        if self.training:
            super()._update_train_stats(data_batch)

    @torch.no_grad()
    def validation_step(self, data, iteration):
        # Fixed noise makes successive held-out loss measurements comparable,
        # while preserving every training RNG. HybridEDMSDE draws its lognormal
        # quantiles from NumPy, not Torch; conditioning may use Python random.
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(195)
                np.random.seed(195)
                random.seed(195)
                return self.training_step(data, iteration)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
