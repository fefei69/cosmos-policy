"""Inference for hanoi_dense_v5 checkpoints: a 16 x 4 chunk of absolute reference poses at 10 Hz.

The reply contract (guide section 6) is ``{"actions": (16, 4) float32 absolute,
"reference_rate_hz": 10, "execution_prefix": 8}`` with jaw intent thresholded
at 0.5, plus an identity block under ``cosmos_hanoi`` carrying the deployment
contract and the hashes of the export and the normalisation statistics.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import numpy as np

from cosmos_policy.datasets.hanoi_dense_data import CONTRACT, DEPLOYMENT_CONTRACT, EXECUTION_PREFIX, HORIZON, REFERENCE_RATE_HZ
from cosmos_policy.datasets.hanoi_joint_data import PROMPT, sha256
from cosmos_policy.experiments.robot.hanoi.joint_policy import make_joint_observation
from cosmos_policy.experiments.robot.hanoi.policy import HanoiInferenceConfig, load_hanoi_weights


@dataclass
class HanoiDenseInferenceConfig(HanoiInferenceConfig):
    config: str = 'cosmos_predict2_2b_hanoi_dense__inference'
    config_file: str = 'cosmos_policy/config/hanoi_dense_config.py'
    chunk_size: int = HORIZON


def validate_dense_config(cfg):
    expected = {'suite': 'hanoi', 'use_third_person_image': True, 'num_third_person_images': 1,
                'use_wrist_image': False, 'num_wrist_images': 0, 'use_proprio': True,
                'normalize_proprio': True, 'unnormalize_actions': True, 'action_dim': 4,
                'chunk_size': HORIZON, 'use_jpeg_compression': False, 'trained_with_image_aug': False}
    for key, value in expected.items():
        if getattr(cfg, key) != value:
            raise ValueError(f'Dense policy requires {key}={value!r}')
    if cfg.num_denoising_steps_action < 1:
        raise ValueError('Positive denoising step count required')


def validate_checkpoint_contract(checkpoint, stats_path):
    path = Path(checkpoint).resolve()
    if path.name == 'model':
        path = path.parent
    run = path.parent.parent  # run/checkpoints/iter_* OR run/exports/iter_*.pt
    identity = json.loads((run / 'joint_contract.json').read_text())
    if identity['contract'] != CONTRACT or sha256(stats_path) != identity['statistics_sha256']:
        raise ValueError('Checkpoint and normalization belong to a different observation/action contract')
    return identity


def load_dense_policy(cfg):
    import torch
    from cosmos_policy.constants import ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK
    from cosmos_policy.experiments.robot import cosmos_utils
    validate_dense_config(cfg)
    if (NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM) != (HORIZON, 4, 7):
        raise ValueError('Set COSMOS_POLICY_PLATFORM=hanoi_dense before model imports')
    identity = validate_checkpoint_contract(cfg.ckpt_path, cfg.dataset_stats_path)
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
    return model, stats, config, identity


def threshold_jaw(actions):
    actions = np.asarray(actions, np.float32).copy()
    if actions.shape != (HORIZON, 4) or not np.isfinite(actions).all():
        raise ValueError('Expected sixteen finite absolute XYZ/jaw predictions')
    actions[:, 3] = actions[:, 3] >= .5
    return actions


def predict_dense_actions(cfg, model, stats, image, state, *, seed=1, num_denoising_steps=None):
    """Return the 16 x 4 absolute chunk; the executor commits the first EXECUTION_PREFIX rows."""
    from cosmos_policy.experiments.robot.cosmos_utils import get_action
    validate_dense_config(cfg)
    observation = make_joint_observation(image, state)
    steps = cfg.num_denoising_steps_action if num_denoising_steps is None else int(num_denoising_steps)
    prediction = get_action(cfg, model, stats, observation, PROMPT, seed=seed, randomize_seed=False,
                            num_denoising_steps_action=steps,
                            generate_future_state_and_value_in_parallel=False, batch_size=1)
    return threshold_jaw(prediction['actions'])


class HanoiDensePolicy:
    """Adapter for the asynchronous chunk executor, without robot I/O."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.model, self.stats, _, self.checkpoint_identity = load_dense_policy(cfg)
        export = Path(cfg.ckpt_path).resolve()
        self.identity = {
            'cosmos_hanoi': {
                'contract': dict(DEPLOYMENT_CONTRACT), 'contract_name': CONTRACT, 'prompt': PROMPT,
                'export_sha256': sha256(export) if export.is_file() else None, 'checkpoint': str(export),
                'normalization_sha256': sha256(cfg.dataset_stats_path), 'num_steps': cfg.num_denoising_steps_action,
                'config_name': cfg.config, 'reference_rate_hz': REFERENCE_RATE_HZ, 'execution_prefix': EXECUTION_PREFIX,
                'initial_weights': self.checkpoint_identity.get('initial_weights'),
                'initial_weights_sha256': self.checkpoint_identity.get('initial_weights_sha256'),
            }
        }

    def infer(self, observation, *, seed=1):
        if observation.get('prompt', PROMPT) != PROMPT:
            raise ValueError('This policy was trained for AAAA to CCCC only')
        actions = predict_dense_actions(self.cfg, self.model, self.stats, observation['observation/image'],
                                        observation['observation/state'], seed=seed)
        return {'actions': actions, 'reference_rate_hz': REFERENCE_RATE_HZ, 'execution_prefix': EXECUTION_PREFIX}


__all__ = ['HanoiDenseInferenceConfig', 'HanoiDensePolicy', 'load_dense_policy', 'predict_dense_actions',
           'threshold_jaw', 'validate_checkpoint_contract', 'validate_dense_config']
