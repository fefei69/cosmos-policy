"""Failure-sensitive checks for the isolated distributed qualification."""
from types import SimpleNamespace

import pytest
import torch

from cosmos_policy.datasets.resumable_sampler import ResumableDistributedSampler
from cosmos_policy.utils.hanoi_multigpu_training import optimizer_digest, select_rank_rng, tensor_digest


def test_each_rank_recovers_its_own_rng_and_rejects_changed_allocation():
    state = {'iteration': 5, 'world_size': 4,
             'rank_states': [{'iteration': 5, 'torch_cpu': [rank]} for rank in range(4)]}
    assert [select_rank_rng(state, rank, 4)['torch_cpu'] for rank in range(4)] == [[0], [1], [2], [3]]
    with pytest.raises(RuntimeError, match='allocation'):
        select_rank_rng(state, 0, 2)
    state['rank_states'][2]['iteration'] = 4
    with pytest.raises(RuntimeError, match='iteration'):
        select_rank_rng(state, 0, 4)


def test_streamed_digest_detects_one_changed_weight_and_nonfinite():
    tensor = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    expected = tensor_digest([('weight', tensor)])
    assert tensor_digest([('weight', tensor.clone())]) == expected
    tensor[2, 3] += 1
    assert tensor_digest([('weight', tensor)]) != expected
    tensor[0, 0] = float('nan')
    with pytest.raises(FloatingPointError):
        tensor_digest([('weight', tensor)])


def test_optimizer_digest_includes_fp32_precision_and_step():
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    optimizer = SimpleNamespace(param_groups=[{'params': [parameter], 'lr': torch.tensor(1e-5), 'step': torch.tensor(5)}],
                                state={parameter: {key: torch.ones(2) for key in ('exp_avg', 'exp_avg_sq', 'master_param')}})
    before = optimizer_digest(optimizer)
    optimizer.state[parameter]['master_param'][0] += 1e-6
    assert optimizer_digest(optimizer) != before
    before = optimizer_digest(optimizer)
    optimizer.param_groups[0]['step'] += 1
    assert optimizer_digest(optimizer) != before
    optimizer.state[parameter]['exp_avg'] = torch.ones(2, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match='FP32'):
        optimizer_digest(optimizer)


@pytest.mark.parametrize('size', [2, 4])
def test_distributed_batches_cover_32_and_resume_across_epoch_boundary(size):
    dataset = list(range(5031))
    accumulation = 32 // (2 * size)
    rank_streams = []
    for rank in range(size):
        sampler = ResumableDistributedSampler(dataset, num_replicas=size, rank=rank, seed=0)
        rank_streams.append(list(sampler)[:2 * accumulation])
        # Exercise resume after a whole-epoch crossing with a partial accumulation window.
        sampler.resume_from_iteration(158, accumulation, 2, True)
        epoch = sampler.epoch
        offset = sampler.start_index
        reference = ResumableDistributedSampler(dataset, num_replicas=size, rank=rank, seed=0)
        reference.set_epoch(epoch)
        assert list(sampler) == list(reference)[offset:]
    combined = [index for stream in rank_streams for index in stream]
    assert len(combined) == len(set(combined)) == 32
