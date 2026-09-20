"""Hanoi policy model whose initial weights may come from the Cosmos-Predict2 video base.

Run B of the dense guide starts from ``Cosmos-Predict2-2B-Video2World`` rather
than from a released policy checkpoint. The network is the same DiT, so no new
parameters exist: the action, proprioception, future-state and value latent
slots are ordinary latent frames that the base model has simply never been
trained to fill. The base release stores its weights under different key
layouts than the policy release, so this loader accepts a nested ``model``
dictionary, keys with or without the ``net.`` prefix, and drops EMA entries,
then loads the remainder strictly. It is untested until the gated checkpoint
is available on the cluster.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

import torch

from cosmos_policy.models.hanoi_model import HanoiPolicyModel, _empty_attention_metadata, load_initial_policy_checkpoint

VIDEO_BASE = 'video_base'
POLICY = 'policy'
EMA_PREFIXES = ('net_ema.', 'ema.', 'model_ema.')


def normalise_video_base_state(state):
    """Return net.* entries of a Cosmos-Predict2 release checkpoint as a flat net state dict."""
    if isinstance(state, Mapping) and 'model' in state and isinstance(state['model'], Mapping):
        state = state['model']
    elif isinstance(state, Mapping) and 'state_dict' in state and isinstance(state['state_dict'], Mapping):
        state = state['state_dict']
    if not isinstance(state, Mapping) or not state:
        raise ValueError('Video base checkpoint must contain a nonempty model state dictionary')
    result = {}
    for key, value in state.items():
        if not isinstance(key, str) or key.startswith(EMA_PREFIXES):
            continue
        if key.startswith('net.'):
            key = key[len('net.'):]
        elif '.' in key and key.split('.', 1)[0] in ('tokenizer', 'conditioner', 'text_encoder'):
            continue  # Frozen modules carried by some releases; the policy loads its own.
        result[key] = value
    if not result:
        raise ValueError('Video base checkpoint has no network weights after filtering')
    return result


def load_video_base_checkpoint(net: torch.nn.Module, path: str) -> dict:
    state = normalise_video_base_state(torch.load(path, map_location='cpu', weights_only=True, mmap=True))
    expected = net.state_dict()
    omitted = [key for key, value in state.items()
               if key not in expected and key.endswith('.attn_op._extra_state') and _empty_attention_metadata(value)]
    for key in omitted:
        del state[key]
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(f'Video base does not match the policy network: {len(missing)} missing '
                         f'(first {missing[:5]}), {len(unexpected)} unexpected (first {unexpected[:5]})')
    mismatched = [key for key in expected if tuple(expected[key].shape) != tuple(state[key].shape)]
    if mismatched:
        raise ValueError(f'Video base tensor shapes differ for {len(mismatched)} keys (first {mismatched[:5]})')
    net.load_state_dict(state, strict=True, assign=False)
    return {'loaded': len(state), 'omitted_attention_metadata': len(omitted)}


class HanoiDensePolicyModel(HanoiPolicyModel):
    """Selects the initial-weight loader from HANOI_INIT_FORMAT: 'policy' (strict net.*) or 'video_base'."""

    def on_train_start(self, memory_format=torch.preserve_format):
        if not getattr(self, '_hanoi_initial_checkpoint_loaded', False) and self.config.initial_checkpoint:
            fmt = os.environ.get('HANOI_INIT_FORMAT', POLICY)
            if fmt == VIDEO_BASE:
                report = load_video_base_checkpoint(self.net, self.config.initial_checkpoint)
                print(f'Loaded Cosmos-Predict2 video base weights: {self.config.initial_checkpoint} {report}', flush=True)
                self._hanoi_initial_checkpoint_loaded = True
            elif fmt != POLICY:
                raise ValueError(f'Unknown HANOI_INIT_FORMAT {fmt!r}')
        # The policy format (or nothing left to do) falls through to the strict loader.
        super().on_train_start(memory_format)


__all__ = ['HanoiDensePolicyModel', 'load_initial_policy_checkpoint', 'load_video_base_checkpoint', 'normalise_video_base_state']
