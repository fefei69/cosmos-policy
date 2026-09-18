r"""Consolidate a completed Hanoi DCP model checkpoint on CPU for desktop inference.

Run once on a CPU node with enough host RAM for the complete policy weights:
    python examples/hanoi/export_checkpoint.py --checkpoint /run/checkpoints/iter_000001000 \
        --output /run/export/hanoi.pt

Only the model directory is read. Optimizer/trainer state is not exported.
The destination is published atomically and existing files are never replaced.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path


def export_checkpoint(checkpoint: str | Path, output: str | Path) -> dict:
    import torch
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    source = Path(checkpoint).expanduser().resolve()
    destination = Path(output).expanduser().absolute()
    if (source / "model" / ".metadata").is_file():
        source = source / "model"
    if not (source / ".metadata").is_file():
        raise ValueError("Provide a completed DCP model directory, or an iteration directory containing model/.metadata")
    if destination.suffix != ".pt":
        raise ValueError("Output must have the .pt extension")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if destination.resolve().is_relative_to(source):
        raise ValueError("Export outside the source model checkpoint directory")

    metadata = FileSystemReader(str(source)).read_metadata().state_dict_metadata
    if not metadata or any(not key.startswith(("net.", "net_ema.")) for key in metadata):
        raise ValueError("Source must contain flat policy model weights only, not optimizer or trainer state")
    if not any(key.startswith("net.") for key in metadata):
        raise ValueError("Source checkpoint has no regular policy network weights")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".pt", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        # This PyTorch utility loads DCP shards on CPU with no distributed group.
        dcp_to_torch_save(str(source), str(temporary))
        state = torch.load(str(temporary), map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(state, dict) or state.keys() != metadata.keys():
            raise ValueError("Consolidated checkpoint keys differ from DCP metadata")
        tensor_count, element_count = 0, 0
        for key, entry in metadata.items():
            if not isinstance(entry, TensorStorageMetadata):
                continue
            tensor = state[key]
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(entry.size):
                raise ValueError(f"Consolidation changed tensor shape for {key}")
            if tensor.dtype != entry.properties.dtype or tensor.device.type != "cpu":
                raise ValueError(f"Consolidation changed tensor dtype/device for {key}")
            tensor_count += 1
            element_count += tensor.numel()
            if tensor.is_floating_point() and tensor.numel():
                flat = tensor.reshape(-1)
                # An explicit spot check avoids reading several billion values.
                if not torch.isfinite(flat[:16]).all() or not torch.isfinite(flat[-16:]).all():
                    raise ValueError(f"Non-finite values in sampled weights for {key}")
        del state
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        # A same-directory hard link is atomic and fails if destination exists,
        # including a competing export that finished while consolidation ran.
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {
            "source_model_directory": str(source),
            "output": str(destination),
            "file_bytes": destination.stat().st_size,
            "tensor_count": tensor_count,
            "element_count": element_count,
            "validation": "All keys/shapes/dtypes verified; first/last 16 values of each floating tensor checked finite",
        }
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
