"""Resume deterministic distributed sampling without reading skipped examples.

Offsets count samples on one data-parallel rank. ``len(sampler)`` and
``len(dataloader)`` describe the remaining part of the selected epoch;
``full_epoch_batches`` always describes a complete epoch, independent of offset.
Checkpoints must be saved after completed optimizer/gradient-accumulation steps.
"""

import hashlib
import json
import os
import tempfile
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DistributedSampler


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ResumableDistributedSampler(DistributedSampler):
    """Stock epoch permutation/sharding, with an offset into a rank's indices."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_index = 0

    def __iter__(self):
        # Only sampler indices are skipped. DataLoader never requests these
        # examples, so expensive HDF5 image decoding is not replayed on resume.
        return islice(super().__iter__(), self.start_index, None)

    def __len__(self):
        return self.num_samples - self.start_index

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self.start_index = 0

    def set_start_index(self, start_index: int) -> None:
        if not 0 <= start_index <= self.num_samples:
            raise ValueError(f"Sample offset must be between 0 and {self.num_samples}, got {start_index}")
        self.start_index = start_index

    def full_epoch_batches(self, batch_size: int, drop_last: bool) -> int:
        """Number of complete-epoch DataLoader batches, including its partial tail if enabled."""
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        batches, remainder = divmod(self.num_samples, batch_size)
        return batches + int(bool(remainder) and not drop_last)

    def resume_from_iteration(
        self, iteration: int, grad_accum_iter: int, batch_size: int, drop_last: bool
    ) -> tuple[int, int]:
        """Set the next epoch/sample index from completed optimizer steps.

        Accumulation may span epoch boundaries. Counting microbatches, rather
        than rounded optimizer steps per epoch, preserves those boundaries.
        Returns ``(epoch, batch_offset_within_epoch)`` for logging.
        """
        if iteration < 0 or grad_accum_iter < 1:
            raise ValueError("Iteration must be nonnegative and gradient accumulation must be positive")
        batches_per_epoch = self.full_epoch_batches(batch_size, drop_last)
        if batches_per_epoch == 0:
            raise ValueError("Training DataLoader has no batches on each rank; reduce batch size or disable drop_last")
        epoch, batch_offset = divmod(iteration * grad_accum_iter, batches_per_epoch)
        self.set_epoch(epoch)
        self.set_start_index(batch_offset * batch_size)
        return epoch, batch_offset


def make_data_order_contract(dataloader, grad_accum_iter: int, worker_seed: int) -> dict:
    """Capture only settings which affect sample order or gradient-step position."""
    sampler = dataloader.sampler
    identity = getattr(dataloader.dataset, "resume_data_order_identity", None)
    if not isinstance(sampler, ResumableDistributedSampler) or identity is None:
        raise ValueError("Resumable datasets must provide a sampler and resume_data_order_identity")
    return {
        "format_version": 1,
        "dataset_identity": identity,
        "dataset_size": len(dataloader.dataset),
        "num_replicas": sampler.num_replicas,
        "samples_per_rank": sampler.num_samples,
        "sampler_drop_last": sampler.drop_last,
        "sampler_shuffle": sampler.shuffle,
        "sampler_seed": sampler.seed,
        "torch_version": str(torch.__version__),
        "batch_size": dataloader.batch_size,
        "dataloader_drop_last": dataloader.drop_last,
        "grad_accum_iter": grad_accum_iter,
        "worker_seed": worker_seed,
        "full_epoch_batches": sampler.full_epoch_batches(dataloader.batch_size, dataloader.drop_last),
    }


def check_data_order_contract(path: str | Path, contract: dict, *, allow_create: bool) -> None:
    """Persist once on a fresh run; reject unknown or changed resume ordering.

    Rank zero may create this file before a distributed barrier. Other ranks
    validate the resulting file without writing. Training duration, checkpoint
    cadence, and validation frequency are deliberately absent from the contract.
    """
    path = Path(path)
    # Normalize tuples/numpy-free primitive containers through the same JSON
    # representation used on disk before comparison.
    expected = json.loads(json.dumps(contract, sort_keys=True))
    if not path.exists():
        if not allow_create:
            raise ValueError(
                f"Missing data-order contract for resumed training: {path}; previous batch/order settings are unknown"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(expected, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            # Do not replace a contract from a concurrent launch of this run.
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
    recorded = json.loads(path.read_text())
    if recorded != expected:
        changed = sorted(key for key in recorded.keys() | expected.keys() if recorded.get(key) != expected.get(key))
        raise ValueError(
            f"Cannot resume with changed data-order settings ({', '.join(changed)}); "
            "restore the recorded dataset, sharding, batch size and accumulation, or start a separate experiment"
        )


def isolated_dataloader_generator(dataset, seed: int):
    """Keep Hanoi iterator/worker base seeds from consuming model-noise RNG."""
    if getattr(dataset, "resume_data_order", False):
        return torch.Generator().manual_seed(seed)
    return None
