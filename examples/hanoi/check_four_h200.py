"""Bounded four-H200 access, BF16 arithmetic, and NCCL communication check.

This diagnostic does not load a policy, dataset, or training checkpoint.
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4 or int(os.environ["LOCAL_WORLD_SIZE"]) != 4:
        raise RuntimeError("This diagnostic requires four processes on one node")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("Expected exactly four visible CUDA devices")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    props = torch.cuda.get_device_properties(device)
    if "H200" not in props.name:
        raise RuntimeError(f"Expected H200, found {props.name}")
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=90), device_id=device)
    try:
        # Each rank contributes a different value. All four must participate.
        collective_sizes = [1, 262144, 4194304]
        for count in collective_sizes:
            values = torch.full((count,), float(rank + 1), device=device)
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            if not torch.all(values == 10).item():
                raise RuntimeError(f"Incorrect NCCL sum for {count} elements")
            del values

        # An exactly representable BF16 reference checks arithmetic on each GPU.
        width = 4096
        left = torch.ones((width, width), device=device, dtype=torch.bfloat16)
        right = torch.full_like(left, 1.0 / width)
        result = left @ right
        torch.cuda.synchronize(device)
        if not torch.all(result == 1).item():
            raise RuntimeError("BF16 matrix multiplication failed its known-answer check")

        # Small real forward/backward updates exercise distributed gradients.
        torch.manual_seed(195)
        layer = torch.nn.Linear(1024, 256, bias=False).to(device=device, dtype=torch.bfloat16)
        model = torch.nn.parallel.DistributedDataParallel(layer, device_ids=[local_rank])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        torch.manual_seed(195 + rank)
        losses = []
        for _ in range(3):
            inputs = torch.randn((32, 1024), device=device, dtype=torch.bfloat16)
            optimizer.zero_grad(set_to_none=True)
            loss = model(inputs).float().square().mean()
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite diagnostic loss")
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        weights = layer.weight.detach().float()
        low, high = weights.clone(), weights.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN)
        dist.all_reduce(high, op=dist.ReduceOp.MAX)
        if not torch.isfinite(weights).all().item() or not torch.equal(low, high):
            raise RuntimeError("Diagnostic model weights differ across GPUs after updates")

        local = {
            "rank": rank, "local_rank": local_rank, "hostname": socket.gethostname(),
            "gpu_name": props.name, "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
            "total_memory_bytes": props.total_memory, "diagnostic_losses": losses,
            "bf16_known_answer_passed": True, "ddp_updates_passed": 3,
            "nccl_sum_elements_passed": collective_sizes,
        }
        records = [None] * world_size
        dist.all_gather_object(records, local)
        if len({r["hostname"] for r in records}) != 1:
            raise RuntimeError("Expected all four GPUs on the same node")
        uuids = [r["gpu_uuid"] for r in records]
        if "unavailable" not in uuids and len(set(uuids)) != 4:
            raise RuntimeError("Workers are not assigned to four distinct GPUs")
        if rank == 0:
            report = {
                "status": "passed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "job_id": os.environ.get("SLURM_JOB_ID"), "world_size": world_size,
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "nccl_version": torch.cuda.nccl.version(), "devices": records,
                "scope": "GPU allocation, BF16 arithmetic, NCCL collectives, and three tiny DDP updates",
                "cosmos_training_or_checkpoint_migration_validated": False,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(args.output)
            print(json.dumps(report, allow_nan=False), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
