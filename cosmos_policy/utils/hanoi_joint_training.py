"""Publish scalar joint/sparse metrics alongside the local recovery journal."""
from cosmos_policy.utils.hanoi_training import HanoiTrainingMonitor


class HanoiJointTrainingMonitor(HanoiTrainingMonitor):
    def _write(self, record):
        super()._write(record)
        import wandb
        from cosmos_policy._src.imaginaire.utils import distributed
        if distributed.is_rank0() and wandb.run is not None and record.get('metrics'):
            prefix = 'validation' if record['event'] == 'validation' else 'train'
            wandb.log({f'hanoi_joint/{prefix}/{key}': value for key, value in record['metrics'].items()},
                      step=record['iteration'])
