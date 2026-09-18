"""waypoint_v4 experiment: consistent labels, larger micro-batches, no block recompute.

Built on the joint_v3 experiment (same network, optimizer, precision, loss and
latent-slot layout). Differences, all deliberate:

* dataset: ``HanoiWaypointDataset`` over ``data/hanoi_cosmos/waypoint_v4``;
* micro-batch 16 x 2 accumulation (effective batch 32 unchanged), tunable with
  ``HANOI_WAYPOINT_MICROBATCH`` for a memory fallback;
* no block-wise activation checkpointing (the joint run recomputed all 28 blocks
  while using 32 GB of an 80 GB card), tunable with
  ``HANOI_WAYPOINT_ACTIVATION_CHECKPOINT``;
* metrics are read from the GPU once per logging interval;
* 8,000 updates (about 71 passes over 3,589 training examples) with the ALOHA
  schedule shape compressed to that budget; save/export every 1,000.
"""
import os
from pathlib import Path

from hydra.core.config_store import ConfigStore

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L, LazyDict
from cosmos_policy.config.hanoi_joint_config import make_config as joint_config
from cosmos_policy.datasets.hanoi_waypoint_dataset import HanoiWaypointDataset
from cosmos_policy.utils.hanoi_waypoint_training import HanoiWaypointTrainingMonitor

EFFECTIVE_BATCH = 32
MAX_UPDATES = 8000
SAVE_EVERY = 1000
VALIDATION_EVERY = 500
HELD_OUT_MONITOR_EXAMPLES = 64
ACTIVATION_CHECKPOINT_MODES = ('none', 'mm_only', 'block_wise')


def waypoint_schedule(max_updates=MAX_UPDATES):
    """ALOHA's shape: 5% warm-up, linear decay to 0.3 over the budget, then a 0.06 hold."""
    if max_updates < 20:
        raise ValueError('Budget too small for a warm-up')
    return {
        'warm_up_steps': [max_updates // 20, 0],
        'cycle_lengths': [max_updates, 10**9],
        'f_start': [1e-6, 0.06],
        'f_max': [1.0, 0.06],
        'f_min': [0.3, 0.06],
    }


def microbatch_from_env():
    micro = int(os.environ.get('HANOI_WAYPOINT_MICROBATCH', '16'))
    if micro < 1 or EFFECTIVE_BATCH % micro:
        raise ValueError(f'Micro-batch must divide the effective batch of {EFFECTIVE_BATCH}')
    return micro


def make_config():
    c = joint_config()  # Requires COSMOS_POLICY_PLATFORM=hanoi_joint: same 7/8/4 dimensions.
    root = Path(__file__).resolve().parents[2]
    metadata = os.environ.get('HANOI_WAYPOINT_METADATA', str(root / 'data/hanoi_cosmos/waypoint_v4'))
    embeddings = os.environ.get('HANOI_T5_EMBEDDINGS', str(root / 'data/hanoi_cosmos/t5_embeddings.pkl'))
    micro = microbatch_from_env()
    for split, loader in (('train', c.dataloader_train), ('val', c.dataloader_val)):
        loader.dataset = L(HanoiWaypointDataset)(metadata_dir=metadata, t5_text_embeddings_path=embeddings,
                                                 split=split, representative_order=split == 'val')
        loader.batch_size = micro
    c.trainer.grad_accum_iter = EFFECTIVE_BATCH // micro
    c.trainer.max_val_iter = max(1, HELD_OUT_MONITOR_EXAMPLES // micro)
    checkpointing = os.environ.get('HANOI_WAYPOINT_ACTIVATION_CHECKPOINT', 'none')
    if checkpointing not in ACTIVATION_CHECKPOINT_MODES:
        raise ValueError(f'Unknown activation checkpoint mode {checkpointing!r}')
    c.model.config.net.sac_config.mode = checkpointing
    c.trainer.max_iter = MAX_UPDATES
    c.trainer.validation_iter = VALIDATION_EVERY
    c.checkpoint.save_iter = SAVE_EVERY
    for key, value in waypoint_schedule().items():
        c.scheduler[key] = value
    c.trainer.callbacks['hanoi_metrics'] = L(HanoiWaypointTrainingMonitor)()
    c.job.name = os.environ.get('HANOI_WAYPOINT_RUN_NAME', 'hanoi_cosmos_waypoint_v4_20260917')
    c.job.wandb_mode = os.environ.get('HANOI_WAYPOINT_WANDB_MODE', 'online')
    store = ConfigStore.instance()
    store.store(group='experiment', package='_global_', name='cosmos_predict2_2b_hanoi_waypoint', node=LazyDict({}))
    store.store(group='experiment', package='_global_', name='cosmos_predict2_2b_hanoi_waypoint__inference',
                node=LazyDict({'model': {'config': {'initial_checkpoint': '', 'sde': {'sigma_max': 80, 'sigma_min': 4}}}}))
    return c
