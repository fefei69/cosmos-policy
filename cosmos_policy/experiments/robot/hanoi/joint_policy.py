"""Single-camera joint conditioning and prefix-one sparse Cartesian inference."""
from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import numpy as np

from cosmos_policy.datasets.hanoi_joint_data import CONTRACT, PROMPT, sha256
from cosmos_policy.experiments.robot.hanoi.policy import HanoiInferenceConfig, load_hanoi_weights


@dataclass
class HanoiJointInferenceConfig(HanoiInferenceConfig):
    config: str = 'cosmos_predict2_2b_hanoi_joint__inference'
    config_file: str = 'cosmos_policy/config/hanoi_joint_config.py'
    chunk_size: int = 8


def validate_joint_config(cfg):
    expected = {'suite': 'hanoi', 'use_third_person_image': True, 'num_third_person_images': 1,
                'use_wrist_image': False, 'num_wrist_images': 0, 'use_proprio': True,
                'normalize_proprio': True, 'unnormalize_actions': True, 'action_dim': 4,
                'chunk_size': 8, 'use_jpeg_compression': False, 'trained_with_image_aug': False}
    for key, value in expected.items():
        if getattr(cfg, key) != value:
            raise ValueError(f'Joint sparse policy requires {key}={value!r}')
    if cfg.num_denoising_steps_action < 1:
        raise ValueError('Positive denoising step count required')


def make_joint_observation(image, state):
    image, state = np.asarray(image), np.asarray(state, np.float32)
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError('Expected once-cropped RGB224 uint8')
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError('Expected six measured joint angles and measured gripper stroke; no velocity')
    return {'primary_image': image, 'proprio': state.copy()}


def absolute_joint_actions(relative, xyz):
    actions, xyz = np.asarray(relative, np.float32).copy(), np.asarray(xyz, np.float32)
    if actions.shape != (8, 4) or xyz.shape != (3,) or not np.isfinite(actions).all() or not np.isfinite(xyz).all():
        raise ValueError('Expected eight finite XYZ/jaw predictions and independent measured XYZ context')
    actions[:, :3] += xyz[None]
    actions[:, 3] = actions[:, 3] >= .5
    return actions


def validate_checkpoint_contract(checkpoint, stats_path):
    path = Path(checkpoint).resolve()
    if path.name == 'model':
        path = path.parent
    run = path.parent.parent  # run/checkpoints/iter_* OR run/exports/iter_*.pt
    identity = json.loads((run / 'joint_contract.json').read_text())
    if identity['contract'] != CONTRACT or sha256(stats_path) != identity['statistics_sha256']:
        raise ValueError('Checkpoint and normalization belong to a different observation/action contract')
    return identity


def load_joint_policy(cfg):
    import torch
    from cosmos_policy.constants import ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK
    from cosmos_policy.experiments.robot import cosmos_utils
    validate_joint_config(cfg)
    if (NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM) != (8, 4, 7):
        raise ValueError('Set COSMOS_POLICY_PLATFORM=hanoi_joint before model imports')
    validate_checkpoint_contract(cfg.ckpt_path, cfg.dataset_stats_path)
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


def predict_joint_actions(cfg, model, stats, image, state, cartesian_position, *, seed=1, return_future=False):
    """Return all eight absolute targets; the executor commits ONLY row zero.

    With ``return_future`` the model's predicted future frame (224 x 224 x 3 uint8) and its
    value estimate in [0, 1] come back as well. Both are decoded from the same generated
    latent, so the actions are unchanged; the cost is one VAE decode.
    """
    from cosmos_policy.experiments.robot.cosmos_utils import get_action
    validate_joint_config(cfg)
    xyz = np.asarray(cartesian_position, np.float32)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError('Measured XYZ context must be supplied separately from joint angles')
    observation = make_joint_observation(image, state)
    prediction = get_action(cfg, model, stats, observation, PROMPT, seed=seed, randomize_seed=False,
                            num_denoising_steps_action=cfg.num_denoising_steps_action,
                            generate_future_state_and_value_in_parallel=return_future, batch_size=1)
    actions = absolute_joint_actions(prediction['actions'], xyz)
    if not return_future:
        return actions
    future = np.asarray(prediction['future_image_predictions']['future_image'], dtype=np.uint8)
    return actions, future, float(prediction['value_prediction'])


class HanoiJointPolicy:
    """Adapter for a waypoint-completion-driven local executor, without robot I/O."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.model, self.stats, _ = load_joint_policy(cfg)

    def infer(self, observation, *, seed=1, dream=False):
        if observation.get('prompt', PROMPT) != PROMPT:
            raise ValueError('This policy was trained for AAAA to CCCC only')
        result = predict_joint_actions(self.cfg, self.model, self.stats, observation['observation/image'],
                                       observation['observation/state'], observation['observation/cartesian_position'],
                                       seed=seed, return_future=dream)
        if not dream:
            return {'actions': result, 'commit_count': 1, 'reference_rate_hz': None}
        actions, future, value = result
        return {'actions': actions, 'commit_count': 1, 'reference_rate_hz': None, 'future_image': future, 'value': value}
