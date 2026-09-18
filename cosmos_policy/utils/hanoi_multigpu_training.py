"""DDP qualification: rank-local recovery, reduced metrics and replica audits."""
import hashlib
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from cosmos_policy.utils.hanoi_checkpoint import read_rng_sidecar, validate_dcp_parts
from cosmos_policy.utils.hanoi_joint_training import HanoiJointTrainingMonitor


def tensor_digest(tensors):
    """Stream exact tensor bytes through CPU; never hold another model copy."""
    digest = hashlib.sha256()
    for name, tensor in tensors:
        if tensor is None:
            digest.update((name + ':none').encode())
            continue
        value = tensor.detach().cpu().contiguous()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f'Non-finite distributed tensor: {name}')
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def optimizer_digest(optimizer):
    tensors = []
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(group['params']):
            state = optimizer.state.get(parameter, {})
            for field in ('exp_avg', 'exp_avg_sq', 'master_param'):
                value = state.get(field)
                if value is None or value.dtype != torch.float32 or value.shape != parameter.shape:
                    raise RuntimeError(f'Missing full FP32 optimizer state: {field}')
                tensors.append((f'{group_index}/{parameter_index}/{field}', value))
        for field in ('step', 'lr'):
            value = group[field]
            tensors.append((f'{group_index}/{field}', value if torch.is_tensor(value) else torch.tensor(value)))
    return tensor_digest(tensors)


def select_rank_rng(state, rank, world_size):
    states = state.get('rank_states')
    if state.get('world_size') != world_size or not isinstance(states, list) or len(states) != world_size:
        raise RuntimeError('Checkpoint is missing the matching distributed RNG allocation')
    if not 0 <= rank < world_size or any(row.get('iteration') != state['iteration'] for row in states):
        raise RuntimeError('Distributed RNG rank/iteration mismatch')
    return states[rank]


def restore_rng(state):
    if len(state['torch_cuda']) != torch.cuda.device_count():
        raise RuntimeError('Checkpoint visible GPU count changed')
    previous = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    try:
        row = state['python']
        random.setstate((row[0], tuple(row[1]), row[2]))
        row = state['numpy']
        np.random.set_state((row[0], np.asarray(row[1], dtype=np.uint32), *row[2:]))
        torch.set_rng_state(torch.tensor(state['torch_cpu'], dtype=torch.uint8))
        torch.cuda.set_rng_state_all([torch.tensor(value, dtype=torch.uint8) for value in state['torch_cuda']])
    except Exception:
        random.setstate(previous[0])
        np.random.set_state(previous[1])
        torch.set_rng_state(previous[2])
        torch.cuda.set_rng_state_all(previous[3])
        raise


class HanoiMultiGPUTrainingMonitor(HanoiJointTrainingMonitor):
    def __init__(self):
        super().__init__()
        self._optimizer = None
        self._resume_audit = None
        self._audit_next_update = True
        self._timings = []
        self._microbatches = 0
        self._update_started = None

    @staticmethod
    def _average(records):
        keys = sorted(set().union(*(row.keys() for _, row in records)))
        sums = torch.tensor([[sum(n * row[key] for n, row in records if key in row),
                              sum(n for n, row in records if key in row)] for key in keys],
                            dtype=torch.float64, device='cuda')
        dist.all_reduce(sums)
        return {key: float(sums[index, 0] / sums[index, 1]) for index, key in enumerate(keys)}

    def _agree(self, kind, value, iteration):
        values = [None] * dist.get_world_size()
        dist.all_gather_object(values, value)
        if len(set(values)) != 1:
            raise RuntimeError(f'DDP {kind} differs across ranks at {iteration}: {values}')
        self._write({'event': 'replica_agreement', 'kind': kind, 'iteration': iteration,
                     'world_size': len(values), 'sha256': value})
        return value

    def on_training_step_start(self, model, data_batch, iteration=0):
        if self._microbatches == 0:
            self._update_started = time.perf_counter()
        self._microbatches += 1

    def on_before_optimizer_step(self, model_ddp, optimizer, scheduler, grad_scaler, iteration=0):
        self._optimizer = optimizer
        model = model_ddp.module
        if self._resume_audit is not None:
            actual = optimizer_digest(optimizer)
            if actual != self._resume_audit['optimizer_sha256']:
                raise RuntimeError('Reload changed FP32 optimizer moments, master weights, step or LR')
            self._agree('restored_optimizer', actual, iteration)
            self._write({'event': 'optimizer_restore_verified', 'iteration': iteration})
            self._resume_audit = None
        if self._audit_next_update:
            self._agree('averaged_gradients', tensor_digest(
                (name, parameter.grad) for name, parameter in model.net.named_parameters()), iteration)
            self._audit_next_update = False

    def on_before_zero_grad(self, model_ddp, optimizer, scheduler, iteration=0):
        torch.cuda.synchronize()
        # Skip process warmup/expensive initial audit. Checkpoint time is outside this hook.
        if iteration >= 8:
            elapsed = torch.tensor(time.perf_counter() - self._update_started, device='cuda')
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            self._timings.append(float(elapsed))
        self._microbatches = 0

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)
        stop = torch.tensor(int(getattr(self.trainer, 'stop_requested', False)), device='cuda')
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        self.trainer.stop_requested = bool(stop)

    def on_validation_end(self, model, iteration=0):
        if self._val:
            self._write({'event': 'validation', 'iteration': iteration,
                         'samples': sum(n for n, _ in self._val) * dist.get_world_size(),
                         'metrics': self._average(self._val),
                         'metric_definition': 'fixed-noise joint denoising loss across all ranks'})

    def on_save_checkpoint_start(self, model, iteration=0):
        model_hash = self._agree('parameters', tensor_digest(model.net.named_parameters()), iteration)
        optimizer_hash = self._agree('optimizer', optimizer_digest(self._optimizer), iteration)
        super().on_save_checkpoint_start(model, iteration)
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, self._checkpoint_rng)
        self._checkpoint_rng = {**self._checkpoint_rng, 'world_size': dist.get_world_size(),
                                'rank_states': states, 'model_sha256': model_hash,
                                'optimizer_sha256': optimizer_hash}

    def on_load_checkpoint_end(self, model, iteration=0, checkpoint_path=None):
        if not iteration:
            return
        validate_dcp_parts(checkpoint_path)
        state = read_rng_sidecar(checkpoint_path, iteration)
        local = select_rank_rng(state, dist.get_rank(), dist.get_world_size())
        actual = tensor_digest(model.net.named_parameters())
        if actual != state['model_sha256']:
            raise RuntimeError('Checkpoint reload changed policy parameters')
        self._agree('restored_parameters', actual, iteration)
        restore_rng(local)
        self._resume_audit = state
        self._write({'event': 'rank_rng_restored', 'iteration': iteration, 'world_size': dist.get_world_size()})

    def on_load_checkpoint_start(self, model):
        super().on_load_checkpoint_start(model)
        keys, path = self.trainer.checkpointer.keys_to_resume_during_load()
        if path is not None and 'model' in keys:
            select_rank_rng(read_rng_sidecar(path), dist.get_rank(), dist.get_world_size())

    def on_train_end(self, model, iteration=0):
        super().on_train_end(model, iteration)
        if self._timings:
            seconds = float(np.median(self._timings))
            self._write({'event': 'throughput', 'iteration': iteration, 'measured_updates': len(self._timings),
                         'median_update_seconds': seconds, 'examples_per_second': 32 / seconds,
                         'global_batch': 32, 'world_size': dist.get_world_size()})
