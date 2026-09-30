"""Inference for hanoi_play_k5 checkpoints: goal-sentence-conditioned 16 x 4 absolute chunks at 10 Hz.

The request must carry the sentence of the goal board verbatim
(``prompt_for_board`` in ``hanoi_play_data``, one of 81); there is no default
goal, and any other string is refused, because the sentence is the model's only
goal signal. The reply is the dense contract (``actions`` (16, 4) absolute,
``reference_rate_hz`` 10, ``execution_prefix`` 8) plus the resolved goal board,
and the identity block carries the prompt template, the hash of the 81
sentences and the hashes of the export, the normalisation and the embedding
cache.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from cosmos_policy.datasets.hanoi_dense_data import EXECUTION_PREFIX, HORIZON, REFERENCE_RATE_HZ
from cosmos_policy.datasets.hanoi_joint_data import sha256
from cosmos_policy.datasets.hanoi_play_data import (
    BOARD_BY_PROMPT, BOARDS, CONTRACT, DEFAULT_EMBEDDINGS, DEPLOYMENT_CONTRACT_V7, PROMPT_TEMPLATE, PROMPTS, PROMPTS_SHA256,
    prompt_for_board,
)
from cosmos_policy.datasets.hanoi_play_dataset import load_prompt_embeddings
from cosmos_policy.experiments.robot.hanoi.dense_policy import HanoiDenseInferenceConfig, threshold_jaw, validate_dense_config
from cosmos_policy.experiments.robot.hanoi.joint_policy import make_joint_observation
from cosmos_policy.experiments.robot.hanoi.policy import load_hanoi_weights


@dataclass
class HanoiPlayInferenceConfig(HanoiDenseInferenceConfig):
    config: str = 'cosmos_predict2_2b_hanoi_play__inference'
    config_file: str = 'cosmos_policy/config/hanoi_play_config.py'
    t5_text_embeddings_path: str = str(DEFAULT_EMBEDDINGS)
    chunk_size: int = HORIZON


def resolve_goal(prompt):
    """The goal board whose sentence this is, or a ValueError; sentences must match verbatim."""
    if not isinstance(prompt, str) or prompt not in BOARD_BY_PROMPT:
        raise ValueError('The prompt must be the goal board sentence verbatim (one of the 81 trained sentences); there is no default goal')
    return BOARD_BY_PROMPT[prompt]


def validate_checkpoint_contract(checkpoint, stats_path, embeddings_path):
    path = Path(checkpoint).resolve()
    if path.name == 'model':
        path = path.parent
    run = path.parent.parent  # run/checkpoints/iter_* OR run/exports/iter_*.pt
    identity = json.loads((run / 'joint_contract.json').read_text())
    if identity['contract'] != CONTRACT or sha256(stats_path) != identity['statistics_sha256']:
        raise ValueError('Checkpoint and normalization belong to a different observation/action contract')
    if identity.get('prompts_sha256') != PROMPTS_SHA256 or identity.get('embeddings_sha256') != sha256(embeddings_path):
        raise ValueError('Checkpoint was trained with different goal sentences or a different embedding cache')
    if int(identity.get('horizon', HORIZON)) != HORIZON:
        raise ValueError('Checkpoint was trained with a different chunk horizon')
    return identity


def load_play_policy(cfg):
    import torch
    from cosmos_policy.constants import ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK
    from cosmos_policy.experiments.robot import cosmos_utils
    validate_dense_config(cfg)
    if (NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM) != (HORIZON, 4, 7):
        raise ValueError('Set COSMOS_POLICY_PLATFORM=hanoi_dense (chunk 16) before model imports')
    identity = validate_checkpoint_contract(cfg.ckpt_path, cfg.dataset_stats_path, cfg.t5_text_embeddings_path)
    stats = json.loads(Path(cfg.dataset_stats_path).read_text())
    for group, size in [('proprio', 7), ('actions', 4)]:
        lo, hi = (np.asarray(stats[group + suffix], np.float32) for suffix in ('_min', '_max'))
        if lo.shape != (size,) or hi.shape != (size,) or not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
            raise ValueError('Invalid normalization bounds')
        stats[group + '_min'], stats[group + '_max'] = lo, hi
    # Every goal sentence goes into the inference cache; Hanoi inference never computes embeddings on the fly.
    for prompt, embedding in zip(PROMPTS, load_prompt_embeddings(cfg.t5_text_embeddings_path)):
        cosmos_utils.t5_text_embeddings_cache[prompt] = embedding[None].to(device='cuda', dtype=torch.bfloat16)
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


def predict_play_actions(cfg, model, stats, image, state, prompt, *, seed=1, num_denoising_steps=None):
    """Return the 16 x 4 absolute chunk for one observation under the given goal sentence."""
    from cosmos_policy.experiments.robot.cosmos_utils import get_action
    validate_dense_config(cfg)
    goal = resolve_goal(prompt)
    observation = make_joint_observation(image, state)
    steps = cfg.num_denoising_steps_action if num_denoising_steps is None else int(num_denoising_steps)
    prediction = get_action(cfg, model, stats, observation, prompt_for_board(goal), seed=seed, randomize_seed=False,
                            num_denoising_steps_action=steps,
                            generate_future_state_and_value_in_parallel=False, batch_size=1)
    actions = threshold_jaw(prediction['actions'])
    if actions.shape[0] != cfg.chunk_size:
        raise ValueError('Predicted chunk length differs from the configured horizon')
    return actions


class HanoiPlayPolicy:
    """Adapter for the asynchronous chunk executor, without robot I/O. Every request names its goal board by sentence."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.model, self.stats, _, self.checkpoint_identity = load_play_policy(cfg)
        export = Path(cfg.ckpt_path).resolve()
        self.identity = {
            'cosmos_hanoi': {
                'contract': dict(DEPLOYMENT_CONTRACT_V7), 'contract_name': CONTRACT,
                'prompt_template': PROMPT_TEMPLATE, 'prompts_sha256': PROMPTS_SHA256, 'goal_sentences': len(PROMPTS),
                'boards': list(BOARDS), 'action_horizon': cfg.chunk_size,
                'export_sha256': sha256(export) if export.is_file() else None, 'checkpoint': str(export),
                'normalization_sha256': sha256(cfg.dataset_stats_path), 'embeddings_sha256': sha256(cfg.t5_text_embeddings_path),
                'num_steps': cfg.num_denoising_steps_action, 'config_name': cfg.config,
                'reference_rate_hz': REFERENCE_RATE_HZ, 'execution_prefix': EXECUTION_PREFIX,
                'horizon_cap_moves': self.checkpoint_identity.get('horizon_cap_moves'),
                'initial_weights': self.checkpoint_identity.get('initial_weights'),
                'initial_weights_sha256': self.checkpoint_identity.get('initial_weights_sha256'),
            }
        }

    def infer(self, observation, *, seed=1):
        if 'prompt' not in observation:
            raise ValueError('A play-policy request must carry the goal board sentence')
        goal = resolve_goal(observation['prompt'])
        actions = predict_play_actions(self.cfg, self.model, self.stats, observation['observation/image'],
                                       observation['observation/state'], prompt_for_board(goal), seed=seed)
        return {'actions': actions, 'reference_rate_hz': REFERENCE_RATE_HZ, 'execution_prefix': EXECUTION_PREFIX,
                'action_horizon': self.cfg.chunk_size, 'goal_board': goal, 'prompt': prompt_for_board(goal)}


__all__ = ['HanoiPlayInferenceConfig', 'HanoiPlayPolicy', 'load_play_policy', 'predict_play_actions', 'resolve_goal',
           'validate_checkpoint_contract']
