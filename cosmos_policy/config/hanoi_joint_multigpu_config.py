"""Isolated DDP qualification; the running single-H100 config is unchanged."""
import os

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy.config.hanoi_joint_config import make_config as joint_config
from cosmos_policy.utils.hanoi_multigpu_training import HanoiMultiGPUTrainingMonitor


def make_config():
    c = joint_config()
    size = int(os.environ['WORLD_SIZE'])
    if size not in (2, 4):
        raise ValueError('Qualification requires two or four GPUs on one node')
    c.trainer.distributed_parallelism = 'ddp'
    c.model.config.fsdp_shard_size = 1  # Replicated parameters; DDP averages gradients.
    c.trainer.ddp.static_graph = False  # Enable no_sync during accumulation.
    c.trainer.ddp.find_unused_parameters = False
    c.trainer.ddp.broadcast_buffers = False
    c.trainer.grad_accum_iter = 32 // (2 * size)
    c.trainer.max_val_iter = 32 // size  # 64 examples globally.
    c.dataloader_train.num_workers = 2
    c.dataloader_val.num_workers = 2
    c.trainer.callbacks.pop('hanoi_metrics')
    # Check gradients before the native clipping callback can sanitize NaNs.
    c.trainer.callbacks = {'hanoi_metrics': L(HanoiMultiGPUTrainingMonitor)(), **c.trainer.callbacks}
    c.job.wandb_mode = 'disabled'
    return c
