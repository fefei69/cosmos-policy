"""CPU data-order replay checks against uninterrupted stock distributed sampling."""

import copy

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from cosmos_policy.datasets.resumable_sampler import (
    ResumableDistributedSampler,
    check_data_order_contract,
    isolated_dataloader_generator,
    make_data_order_contract,
)


class CountingDataset(Dataset):
    resume_data_order = True
    resume_data_order_identity = {"source": "fixture", "indices_sha256": "fixed-index-order"}

    def __init__(self, size):
        self.size = size
        self.reads = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.reads.append(index)
        return index


def collect_batches(dataset, sampler, batch_size, drop_last, count):
    """Run actual DataLoaders, preserving the sampler's initial epoch/offset."""
    batches = []
    epoch = sampler.epoch
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        generator=isolated_dataloader_generator(dataset, seed=195),
    )
    while len(batches) < count:
        for batch in loader:
            batches.append(batch.tolist())
            if len(batches) == count:
                return batches
        epoch += 1
        sampler.set_epoch(epoch)
    return batches


@pytest.mark.parametrize("replicas", [1, 2, 3])
@pytest.mark.parametrize("sampler_drop_last", [False, True])
@pytest.mark.parametrize("loader_drop_last", [False, True])
@pytest.mark.parametrize("accumulation", [1, 3, 8])
def test_resume_matches_uninterrupted_shards_and_never_reads_skipped_prefix(
    replicas, sampler_drop_last, loader_drop_last, accumulation
):
    for rank in range(replicas):
        kwargs = dict(num_replicas=replicas, rank=rank, shuffle=True, seed=17, drop_last=sampler_drop_last)
        dataset = CountingDataset(29)
        reference = collect_batches(
            dataset, DistributedSampler(dataset, **kwargs), 3, loader_drop_last, 5 * accumulation + 7
        )
        for completed_steps in (0, 1, 2, 5):
            resumed_dataset = CountingDataset(29)
            sampler = ResumableDistributedSampler(resumed_dataset, **kwargs)
            sampler.resume_from_iteration(completed_steps, accumulation, batch_size=3, drop_last=loader_drop_last)
            actual = collect_batches(resumed_dataset, sampler, 3, loader_drop_last, 7)
            consumed_microbatches = completed_steps * accumulation
            assert actual == reference[consumed_microbatches : consumed_microbatches + 7]
            # No skipped prefix was fetched through __getitem__; this remains
            # true when a checkpoint falls after several epochs or across an
            # accumulation window that spans an epoch boundary.
            assert resumed_dataset.reads == [index for batch in actual for index in batch]


def test_remaining_length_full_epoch_length_and_exact_epoch_boundary():
    dataset = CountingDataset(11)
    sampler = ResumableDistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
    loader = DataLoader(dataset, sampler=sampler, batch_size=3, drop_last=True)
    assert sampler.full_epoch_batches(3, True) == 3
    assert sampler.full_epoch_batches(3, False) == 4
    assert sampler.resume_from_iteration(1, 2, 3, True) == (0, 2)
    assert sampler.start_index == 6
    assert len(sampler) == 5
    assert len(loader) == 1
    assert sampler.full_epoch_batches(3, True) == 3
    assert next(iter(loader)).tolist() == [6, 7, 8]
    # Exactly one complete epoch: start the next permutation at its beginning.
    assert sampler.resume_from_iteration(1, 3, 3, True) == (1, 0)
    assert len(sampler) == 11
    assert len(loader) == 3
    sampler.set_start_index(9)
    sampler.set_epoch(2)
    assert sampler.start_index == 0
    with pytest.raises(ValueError, match="offset"):
        sampler.set_start_index(12)
    with pytest.raises(ValueError, match="positive"):
        sampler.resume_from_iteration(0, 0, 3, True)
    empty = ResumableDistributedSampler(CountingDataset(1), num_replicas=2, rank=0, drop_last=True)
    with pytest.raises(ValueError, match="no batches"):
        empty.resume_from_iteration(1, 1, 3, True)


def test_isolated_train_and_validation_iterators_preserve_model_rng():
    dataset = CountingDataset(5)
    torch.manual_seed(1234)
    initial_rng = torch.get_rng_state().clone()
    train_generator = isolated_dataloader_generator(dataset, seed=195)
    val_generator = isolated_dataloader_generator(dataset, seed=195)
    assert train_generator is not val_generator
    train = DataLoader(dataset, batch_size=2, generator=train_generator)
    validation = DataLoader(dataset, batch_size=2, generator=val_generator)
    for loader in (train, validation, train):
        next(iter(loader))
    torch.testing.assert_close(torch.get_rng_state(), initial_rng)
    assert isolated_dataloader_generator(list(range(5)), seed=195) is None


def test_resume_contract_requires_unchanged_batch_accumulation_and_data(tmp_path):
    dataset = CountingDataset(29)
    sampler = ResumableDistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True, seed=17)
    loader = DataLoader(dataset, sampler=sampler, batch_size=3, drop_last=True)
    contract = make_data_order_contract(loader, grad_accum_iter=8, worker_seed=195)
    path = tmp_path / "experiment" / "data_order.json"
    with pytest.raises(ValueError, match="previous batch/order settings are unknown"):
        check_data_order_contract(path, contract, allow_create=False)
    check_data_order_contract(path, contract, allow_create=True)
    original = path.read_bytes()
    check_data_order_contract(path, contract, allow_create=False)
    for key, new_value in (
        ("batch_size", 2),
        ("grad_accum_iter", 4),
        ("num_replicas", 4),
        ("dataloader_drop_last", False),
        ("sampler_seed", 18),
        ("dataset_identity", {"source": "changed", "indices_sha256": "other-order"}),
    ):
        changed = copy.deepcopy(contract)
        changed[key] = new_value
        with pytest.raises(ValueError, match="changed data-order settings"):
            check_data_order_contract(path, changed, allow_create=False)
        assert path.read_bytes() == original
    # Offsets change the remaining length, not the persisted full-epoch shape.
    sampler.resume_from_iteration(1, 8, batch_size=3, drop_last=True)
    assert make_data_order_contract(loader, grad_accum_iter=8, worker_seed=195) == contract
