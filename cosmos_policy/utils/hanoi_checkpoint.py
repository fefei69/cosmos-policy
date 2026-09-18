"""Check local resume artifacts without loading model/optimizer tensors."""

import json
import re
from pathlib import Path


def checkpoint_iteration(checkpoint):
    checkpoint = Path(checkpoint)
    match = re.fullmatch(r"iter_(\d{9})", checkpoint.name)
    if match is None:
        raise RuntimeError(f"Invalid Hanoi checkpoint directory: {checkpoint}")
    return int(match.group(1))


def validate_dcp_parts(checkpoint):
    """Require every metadata-referenced byte range in all four DCP parts.

    Checking only .metadata is insufficient: a failed save can leave metadata
    alongside missing or truncated shards. This reads metadata and file sizes,
    not the multi-gigabyte model or optimizer tensors.
    """
    from torch.distributed.checkpoint import FileSystemReader

    checkpoint = Path(checkpoint)
    for part in ("model", "optim", "scheduler", "trainer"):
        directory = (checkpoint / part).resolve()
        try:
            metadata = FileSystemReader(directory).read_metadata()
            if not metadata.state_dict_metadata or not metadata.storage_data:
                raise ValueError("empty checkpoint metadata")
            for storage in metadata.storage_data.values():
                shard = (directory / storage.relative_path).resolve()
                if not shard.is_relative_to(directory):
                    raise ValueError("checkpoint shard escapes its component directory")
                if storage.offset < 0 or storage.length <= 0 or shard.stat().st_size < storage.offset + storage.length:
                    raise ValueError(f"missing/truncated checkpoint shard: {shard}")
        except Exception as error:
            raise RuntimeError(f"Incomplete Hanoi checkpoint component {directory}: {error}") from error


def read_rng_sidecar(checkpoint, iteration=None):
    checkpoint = Path(checkpoint)
    expected = checkpoint_iteration(checkpoint) if iteration is None else iteration
    path = checkpoint / "hanoi_rng.json"
    try:
        state = json.loads(path.read_text())
        if not isinstance(state, dict) or state.get("iteration") != expected:
            raise ValueError("RNG iteration does not match checkpoint")
        if not all(
            isinstance(state.get(key), list) and state[key] for key in ("python", "numpy", "torch_cpu", "torch_cuda")
        ):
            raise ValueError("missing or empty RNG state")
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"Invalid or missing Hanoi RNG sidecar {path}: {error}") from error
    return state


def latest_complete_checkpoint(run):
    """Resolve only this run's checkpoint, refusing a partial resume."""
    run = Path(run).resolve()
    marker = run / "checkpoints/latest_checkpoint.txt"
    try:
        value = marker.read_text().strip()
    except OSError as error:
        raise RuntimeError(f"No complete training checkpoint marker: {marker}") from error
    checkpoint = Path(value)
    if not checkpoint.is_absolute():
        checkpoint = marker.parent / checkpoint
    checkpoint = checkpoint.resolve()
    if checkpoint.parent != marker.parent or not value:
        raise RuntimeError(f"Checkpoint must belong to this Hanoi run: {checkpoint}")
    iteration = checkpoint_iteration(checkpoint)
    validate_dcp_parts(checkpoint)
    read_rng_sidecar(checkpoint, iteration)
    try:
        contract = json.loads((run / "data_order.json").read_text())
        if contract.get("format_version") != 1 or "dataset_identity" not in contract:
            raise ValueError("missing data-order identity")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise RuntimeError(f"Missing or invalid Hanoi data-order contract in {run}: {error}") from error
    return checkpoint


def validate_model_metadata(model_directory, expected_state_dict, *, require_dtype=True):
    """Reject partial or incompatible DCP model weights before tensor loading.

    All parameters and ordinary buffers must match exactly. Only the reserved
    ``._extra_state`` entries used for Transformer Engine's version-dependent
    bookkeeping may be absent or additional. No tensor values are read here.
    """
    import torch
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    def optional_extra_state(key):
        return key == "_extra_state" or key.endswith("._extra_state")

    directory = Path(model_directory)
    expected = {key: value for key, value in expected_state_dict.items() if not optional_extra_state(key)}
    if not expected or any(not isinstance(value, torch.Tensor) for value in expected.values()):
        raise ValueError("Expected a nonempty policy tensor state dictionary for checkpoint validation")
    recorded = {
        key: value
        for key, value in FileSystemReader(directory).read_metadata().state_dict_metadata.items()
        if not optional_extra_state(key)
    }
    missing = sorted(expected.keys() - recorded.keys())
    unexpected = sorted(recorded.keys() - expected.keys())
    mismatched_shapes = []
    mismatched_dtypes = []
    for key in expected.keys() & recorded.keys():
        entry = recorded[key]
        if not isinstance(entry, TensorStorageMetadata) or tuple(entry.size) != tuple(expected[key].shape):
            mismatched_shapes.append(key)
        elif require_dtype and entry.properties.dtype != expected[key].dtype:
            mismatched_dtypes.append(key)
    if missing or unexpected or mismatched_shapes or mismatched_dtypes:
        raise RuntimeError(
            f"DCP model does not match the complete policy network: {directory}; "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"mismatched_shapes={sorted(mismatched_shapes)[:8]}, "
            f"mismatched_dtypes={sorted(mismatched_dtypes)[:8]}"
        )
    return {"validated_model_tensors": len(expected)}


def validate_optimizer_metadata(optimizer_directory, model_net):
    """Require full FP32 Adam moments and master weights before DCP restore.

    The upstream permissive planner can leave newly initialized destination
    tensors in place when a checkpoint omits them. Inspect the saved metadata,
    before that happens, for every trainable parameter of the single-GPU Hanoi
    network. State-dictionary names remove activation-checkpoint wrapper names
    in the same way as PyTorch's optimizer FQNs.
    """
    import torch
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    trainable_ids = {id(parameter) for parameter in model_net.parameters() if parameter.requires_grad}
    parameters = {
        name: parameter
        for name, parameter in model_net.state_dict(prefix="net.", keep_vars=True).items()
        if id(parameter) in trainable_ids
    }
    if not trainable_ids or {id(parameter) for parameter in parameters.values()} != trainable_ids:
        raise ValueError("Cannot identify all trainable Hanoi parameters in the model state dictionary")
    directory = Path(optimizer_directory)
    metadata = FileSystemReader(directory).read_metadata().state_dict_metadata
    missing, mismatched_shapes, mismatched_dtypes = [], [], []
    for name, parameter in parameters.items():
        for field in ("master_param", "exp_avg", "exp_avg_sq"):
            key = f"state.{name}.{field}"
            if key not in metadata:
                missing.append(key)
                continue
            entry = metadata[key]
            if not isinstance(entry, TensorStorageMetadata) or tuple(entry.size) != tuple(parameter.shape):
                mismatched_shapes.append(key)
            elif entry.properties.dtype != torch.float32:
                mismatched_dtypes.append(key)
    if missing or mismatched_shapes or mismatched_dtypes:
        raise RuntimeError(
            f"DCP optimizer lacks complete FP32 Hanoi master weights/moments: {directory}; "
            f"missing={missing[:8]}, mismatched_shapes={mismatched_shapes[:8]}, "
            f"mismatched_dtypes={mismatched_dtypes[:8]}"
        )
    return {"validated_optimizer_parameters": len(parameters)}
