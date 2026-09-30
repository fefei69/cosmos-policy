"""hanoi_play_k5 experiment: language-goal hindsight labels over the play recording, dense 10 Hz chunks.

The six-task recipe (base Hanoi config, micro-batch 16 x 2, no block recompute,
sync-free metrics, v4 schedule shape, video-base or LIBERO initial weights,
32,000 updates, save/export every 2,000, held-out monitoring every 500) over
``HanoiPlayDataset``: dataset ``data/hanoi_cosmos/play_k5`` and the 81-sentence
embedding cache ``data/hanoi_cosmos/t5_embeddings_play.pkl``. Seven latent
slots as before: the goal enters only through the prompt embedding, so no
model geometry changes. Text dropout stays 0: the sentence is the only goal
signal.
"""
import json
import os
from pathlib import Path

from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L, LazyDict
from cosmos_policy.config.hanoi_config import make_config as base_config
from cosmos_policy.config.hanoi_dense_config import (
    ACTIVATION_CHECKPOINT_MODES, EFFECTIVE_BATCH, HELD_OUT_MONITOR_EXAMPLES, INIT_FORMATS, dense_schedule, microbatch_from_env,
)
from cosmos_policy.datasets.hanoi_play_data import DEFAULT_EMBEDDINGS, DEFAULT_METADATA
from cosmos_policy.datasets.hanoi_play_dataset import HanoiPlayDataset
from cosmos_policy.models.hanoi_dense_model import HanoiDensePolicyModel
from cosmos_policy.utils.hanoi_waypoint_training import HanoiWaypointTrainingMonitor

MAX_UPDATES = 32000
SAVE_EVERY = 2000
VALIDATION_EVERY = 500
EXPERIMENT = 'cosmos_predict2_2b_hanoi_play'


def embeddings_path(root):
    """The 81-sentence cache. HANOI_T5_EMBEDDINGS is deliberately ignored (env.sh points it at the two-prompt cache)."""
    return os.environ.get('HANOI_PLAY_EMBEDDINGS', str(root / DEFAULT_EMBEDDINGS))


def make_config():
    if os.environ.get('COSMOS_POLICY_PLATFORM') != 'hanoi_dense':
        raise ValueError('This experiment requires COSMOS_POLICY_PLATFORM=hanoi_dense (chunk 16 x 4, state 7)')
    if os.environ.get('HANOI_INIT_FORMAT', 'policy') not in INIT_FORMATS:
        raise ValueError(f'HANOI_INIT_FORMAT must be one of {INIT_FORMATS}')
    c = base_config()
    root = Path(__file__).resolve().parents[2]
    metadata = os.environ.get('HANOI_PLAY_METADATA', str(root / DEFAULT_METADATA))
    embeddings = embeddings_path(root)
    prepared = Path(metadata) / 'metadata.json'
    if prepared.exists():  # Training always has the prepared dataset; a deployment host loading an export may not.
        from cosmos_policy.constants import NUM_ACTIONS_CHUNK
        horizon = int(json.loads(prepared.read_text())['horizon'])
        if NUM_ACTIONS_CHUNK != horizon:
            raise ValueError(f'HANOI_DENSE_HORIZON={NUM_ACTIONS_CHUNK} but the prepared dataset at {metadata} has horizon {horizon}')
    micro = microbatch_from_env()

    def loader(split):
        return L(DataLoader)(dataset=L(HanoiPlayDataset)(metadata_dir=metadata, t5_text_embeddings_path=embeddings,
                                                         split=split, representative_order=split == 'val'),
                             batch_size=micro, drop_last=split == 'train', num_workers=4,
                             persistent_workers=True, pin_memory=True, timeout=120)
    c.dataloader_train, c.dataloader_val = loader('train'), loader('val')
    c.model = L(HanoiDensePolicyModel)(config=c.model.config, _recursive_=False)
    c.trainer.grad_accum_iter = EFFECTIVE_BATCH // micro
    c.trainer.max_val_iter = max(1, HELD_OUT_MONITOR_EXAMPLES // micro)
    checkpointing = os.environ.get('HANOI_DENSE_ACTIVATION_CHECKPOINT', 'none')
    if checkpointing not in ACTIVATION_CHECKPOINT_MODES:
        raise ValueError(f'Unknown activation checkpoint mode {checkpointing!r}')
    c.model.config.net.sac_config.mode = checkpointing
    c.trainer.max_iter = MAX_UPDATES
    c.trainer.validation_iter = VALIDATION_EVERY
    c.checkpoint.save_iter = SAVE_EVERY
    for key, value in dense_schedule(MAX_UPDATES).items():
        c.scheduler[key] = value
    c.trainer.callbacks['hanoi_metrics'] = L(HanoiWaypointTrainingMonitor)()
    c.job.name = os.environ.get('HANOI_PLAY_RUN_NAME', 'hanoi_cosmos_play_20260930_video_init')
    c.job.wandb_mode = os.environ.get('HANOI_DENSE_WANDB_MODE', 'online')
    store = ConfigStore.instance()
    store.store(group='experiment', package='_global_', name=EXPERIMENT, node=LazyDict({}))
    store.store(group='experiment', package='_global_', name=EXPERIMENT + '__inference',
                node=LazyDict({'model': {'config': {'initial_checkpoint': '', 'sde': {'sigma_max': 80, 'sigma_min': 4}}}}))
    return c
