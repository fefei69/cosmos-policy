"""Inference for waypoint_v4 checkpoints: the joint model with the v4 dataset contract.

``joint_policy.validate_checkpoint_contract`` pins the joint_v3 contract, so the
loader here is a contract-parameterised copy. Observation encoding, action
decoding and the serving adapter are reused unchanged.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import numpy as np

from cosmos_policy.datasets.hanoi_joint_data import CONTRACT as JOINT_CONTRACT, PROMPT, sha256
from cosmos_policy.datasets.hanoi_waypoint_data import CONTRACT as WAYPOINT_CONTRACT
from cosmos_policy.experiments.robot.hanoi.joint_policy import (
    HanoiJointInferenceConfig, HanoiJointPolicy, load_joint_policy, predict_joint_actions, validate_joint_config,
)
from cosmos_policy.experiments.robot.hanoi.policy import load_hanoi_weights

KNOWN_CONTRACTS = (JOINT_CONTRACT, WAYPOINT_CONTRACT)


@dataclass
class HanoiWaypointInferenceConfig(HanoiJointInferenceConfig):
    config: str = 'cosmos_predict2_2b_hanoi_waypoint__inference'
    config_file: str = 'cosmos_policy/config/hanoi_waypoint_config.py'


def inference_config_for(contract, checkpoint, stats_path, embeddings):
    if contract == JOINT_CONTRACT:
        return HanoiJointInferenceConfig(str(checkpoint), str(stats_path), str(embeddings))
    if contract == WAYPOINT_CONTRACT:
        return HanoiWaypointInferenceConfig(str(checkpoint), str(stats_path), str(embeddings))
    raise ValueError(f'Unknown Hanoi dataset contract {contract!r}')


def validate_checkpoint_contract(checkpoint, stats_path, contract):
    path = Path(checkpoint).resolve()
    if path.name == 'model':
        path = path.parent
    run = path.parent.parent  # run/checkpoints/iter_* OR run/exports/iter_*.pt
    identity = json.loads((run / 'joint_contract.json').read_text())
    if identity['contract'] != contract or sha256(stats_path) != identity['statistics_sha256']:
        raise ValueError('Checkpoint and normalization belong to a different observation/action contract')
    return identity


def load_policy(cfg, contract):
    """Load a joint_v3 or waypoint_v4 checkpoint; the network and inputs are identical."""
    if contract == JOINT_CONTRACT:
        return load_joint_policy(cfg)
    if contract != WAYPOINT_CONTRACT:
        raise ValueError(f'Unknown Hanoi dataset contract {contract!r}')
    import torch
    from cosmos_policy.constants import ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK
    from cosmos_policy.experiments.robot import cosmos_utils
    validate_joint_config(cfg)
    if (NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM) != (8, 4, 7):
        raise ValueError('Set COSMOS_POLICY_PLATFORM=hanoi_joint before model imports')
    validate_checkpoint_contract(cfg.ckpt_path, cfg.dataset_stats_path, contract)
    stats = json.loads(Path(cfg.dataset_stats_path).read_text())
    for group, size in [('proprio', 7), ('actions', 4)]:
        lo, hi = (np.asarray(stats[group + suffix], np.float32) for suffix in ('_min', '_max'))
        if lo.shape != (size,) or hi.shape != (size,) or not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
            raise ValueError('Invalid normalization bounds')
        stats[group + '_min'], stats[group + '_max'] = lo, hi
    with open(cfg.t5_text_embeddings_path, 'rb') as stream:
        embedding = torch.as_tensor(pickle.load(stream)[PROMPT])
    if embedding.shape == (512, 1024):
        embedding = embedding[None]
    if embedding.shape != (1, 512, 1024) or not torch.isfinite(embedding).all():
        raise ValueError('Invalid forward prompt embedding')
    cosmos_utils.t5_text_embeddings_cache[PROMPT] = embedding.to(device='cuda', dtype=torch.bfloat16)
    checkpoint = Path(cfg.ckpt_path).resolve()
    if (checkpoint / 'model/.metadata').is_file():
        checkpoint = checkpoint / 'model'
    model, config = cosmos_utils.load_model_from_checkpoint(
        experiment_name=cfg.config, config_file=cfg.config_file, s3_checkpoint_dir=str(checkpoint),
        instantiate_ema=False, load_ema_to_reg=False, skip_load_model=True)
    if checkpoint.is_dir():
        from torch.distributed.checkpoint import FileSystemReader
        from cosmos_policy._src.predict2.checkpointer.dcp import DefaultLoadPlanner, ModelWrapper, dcp_load_state_dict
        from cosmos_policy.utils.hanoi_checkpoint import validate_model_metadata
        wrapper = ModelWrapper(model)
        state = wrapper.state_dict()
        validate_model_metadata(checkpoint, state)
        dcp_load_state_dict(state, FileSystemReader(str(checkpoint)), DefaultLoadPlanner(allow_partial_load=False))
        wrapper.load_state_dict(state)
        del state, wrapper
    else:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        load_hanoi_weights(model, state)
        del state
    if (model.config.state_t, model.config.min_num_conditional_frames) != (7, 3):
        raise ValueError('Expected one current RGB + current joint state conditioning only')
    model.eval().to('cuda')
    torch.cuda.empty_cache()
    return model, stats, config


class HanoiWaypointPolicy(HanoiJointPolicy):
    """Same executor adapter (commit one waypoint, then re-observe) for v4 checkpoints."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.model, self.stats, _ = load_policy(cfg, WAYPOINT_CONTRACT)


__all__ = ['HanoiWaypointInferenceConfig', 'HanoiWaypointPolicy', 'KNOWN_CONTRACTS', 'inference_config_for',
           'load_policy', 'predict_joint_actions', 'validate_checkpoint_contract']
