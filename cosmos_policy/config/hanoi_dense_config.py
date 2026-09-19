"""hanoi_dense_v5 experiment: dense 10 Hz chunks of absolute reference poses.

Built on the base Hanoi config (same network, optimizer, precision, loss and
latent-slot layout) with the waypoint_v4 training settings and these dense
decisions from docs/hanoi_dense_training_guide.md:

* platform ``hanoi_dense`` (chunk 16 x 4, state 7);
* dataset ``HanoiDenseDataset`` over ``data/hanoi_cosmos/dense_v5``;
* effective batch 32 as micro-batch 16 x 2, no block recompute, metrics read
  from the GPU once per logging interval (all tunable by environment);
* 16,000 updates, save/export every 1,000, held-out monitoring every 500;
* the v4 schedule shape compressed to the budget (5% warm-up, linear decay to
  0.3, then a 0.06 hold);
* initial weights from ``HANOI_INIT_CHECKPOINT`` with ``HANOI_INIT_FORMAT``
  ``video_base`` (run B) or ``policy`` (run A, the LIBERO checkpoint).
"""
import os
from pathlib import Path

from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L, LazyDict
from cosmos_policy.config.hanoi_config import make_config as base_config
from cosmos_policy.datasets.hanoi_dense_dataset import HanoiDenseDataset
from cosmos_policy.models.hanoi_dense_model import HanoiDensePolicyModel
from cosmos_policy.utils.hanoi_waypoint_training import HanoiWaypointTrainingMonitor

EFFECTIVE_BATCH = 32
MAX_UPDATES = 16000
SAVE_EVERY = 1000
VALIDATION_EVERY = 500
HELD_OUT_MONITOR_EXAMPLES = 64
ACTIVATION_CHECKPOINT_MODES = ('none', 'mm_only', 'block_wise')
INIT_FORMATS = ('policy', 'video_base')


def dense_schedule(max_updates=MAX_UPDATES):
    """waypoint_v4's shape: 5% warm-up, linear decay to 0.3 over the budget, then a 0.06 hold."""
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
    micro = int(os.environ.get('HANOI_DENSE_MICROBATCH', '16'))
    if micro < 1 or EFFECTIVE_BATCH % micro:
        raise ValueError(f'Micro-batch must divide the effective batch of {EFFECTIVE_BATCH}')
    return micro


def make_config():
    if os.environ.get('COSMOS_POLICY_PLATFORM') != 'hanoi_dense':
        raise ValueError('This experiment requires COSMOS_POLICY_PLATFORM=hanoi_dense')
    if os.environ.get('HANOI_INIT_FORMAT', 'policy') not in INIT_FORMATS:
        raise ValueError(f'HANOI_INIT_FORMAT must be one of {INIT_FORMATS}')
    c = base_config()
    root = Path(__file__).resolve().parents[2]
    metadata = os.environ.get('HANOI_DENSE_METADATA', str(root / 'data/hanoi_cosmos/dense_v5'))
    embeddings = os.environ.get('HANOI_T5_EMBEDDINGS', str(root / 'data/hanoi_cosmos/t5_embeddings.pkl'))
    micro = microbatch_from_env()

    def loader(split):
        return L(DataLoader)(dataset=L(HanoiDenseDataset)(metadata_dir=metadata, t5_text_embeddings_path=embeddings,
                                                          split=split, representative_order=split == 'val'),
                             batch_size=micro, drop_last=split == 'train', num_workers=4,
                             persistent_workers=True, pin_memory=True, timeout=120)
    c.dataloader_train, c.dataloader_val = loader('train'), loader('val')
    # Same model configuration; only the initial-weight loader differs.
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
    for key, value in dense_schedule().items():
        c.scheduler[key] = value
    c.trainer.callbacks['hanoi_metrics'] = L(HanoiWaypointTrainingMonitor)()
    c.job.name = os.environ.get('HANOI_DENSE_RUN_NAME', 'hanoi_cosmos_dense_20260919_libero_init')
    c.job.wandb_mode = os.environ.get('HANOI_DENSE_WANDB_MODE', 'online')
    store = ConfigStore.instance()
    store.store(group='experiment', package='_global_', name='cosmos_predict2_2b_hanoi_dense', node=LazyDict({}))
    store.store(group='experiment', package='_global_', name='cosmos_predict2_2b_hanoi_dense__inference',
                node=LazyDict({'model': {'config': {'initial_checkpoint': '', 'sde': {'sigma_max': 80, 'sigma_min': 4}}}}))
    return c
