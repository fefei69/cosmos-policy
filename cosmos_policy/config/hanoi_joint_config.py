"""Separate Cosmos experiment for seven measured inputs / eight sparse targets."""
import os
from pathlib import Path

from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L, LazyDict
from cosmos_policy.config.hanoi_config import make_config as base_config
from cosmos_policy.datasets.hanoi_joint_dataset import HanoiJointDataset
from cosmos_policy.utils.hanoi_schedule import aloha_training_schedule
from cosmos_policy.utils.hanoi_joint_training import HanoiJointTrainingMonitor


def make_config():
    if os.environ.get('COSMOS_POLICY_PLATFORM') != 'hanoi_joint':
        raise ValueError('This experiment requires COSMOS_POLICY_PLATFORM=hanoi_joint')
    c = base_config()
    root = Path(__file__).resolve().parents[2]
    metadata = os.environ.get('HANOI_JOINT_METADATA', str(root / 'data/hanoi_cosmos/joint_sparse_v3'))
    embeddings = os.environ.get('HANOI_T5_EMBEDDINGS', str(root / 'data/hanoi_cosmos/t5_embeddings.pkl'))
    def loader(split):
        return L(DataLoader)(dataset=L(HanoiJointDataset)(metadata_dir=metadata,
                            t5_text_embeddings_path=embeddings, split=split, representative_order=split == 'val'),
                            batch_size=2, drop_last=split == 'train', num_workers=4,
                            persistent_workers=True, pin_memory=True, timeout=120)
    c.dataloader_train, c.dataloader_val = loader('train'), loader('val')
    c.job.name = os.environ.get('HANOI_JOINT_RUN_NAME', 'hanoi_cosmos_joint_sparse_20260917')
    c.job.wandb_mode = os.environ.get('HANOI_JOINT_WANDB_MODE', 'online')
    c.trainer.callbacks['hanoi_metrics'] = L(HanoiJointTrainingMonitor)()
    c.trainer.max_iter = 30000
    c.trainer.grad_accum_iter = 16  # 2 x 16 x 1 H100 = global batch 32.
    c.trainer.validation_iter = 1000
    c.trainer.max_val_iter = 32
    c.checkpoint.save_iter = 2000
    for key, value in aloha_training_schedule().items():
        c.scheduler[key] = value
    # Native Cosmos FP32-master Adam, BF16, 1e-5 peak, and no EMA are retained.
    # The dataset/objective differs from OpenPI; its loss/LR/EMA are not copied.
    store = ConfigStore.instance()
    store.store(group='experiment', package='_global_', name='cosmos_predict2_2b_hanoi_joint', node=LazyDict({}))
    store.store(group='experiment', package='_global_', name='cosmos_predict2_2b_hanoi_joint__inference',
                node=LazyDict({'model': {'config': {'initial_checkpoint': '', 'sde': {'sigma_max': 80, 'sigma_min': 4}}}}))
    return c
