"""Local metrics and a checkpoint-safe time limit for the Hanoi experiment."""

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from cosmos_policy._src.imaginaire.utils import distributed
from cosmos_policy._src.imaginaire.utils.callback import Callback
from cosmos_policy.utils.hanoi_checkpoint import read_rng_sidecar, validate_dcp_parts


class HanoiTrainingMonitor(Callback):
    def __init__(self):
        super().__init__()
        self._train = []
        self._val = []
        self._started = time.time()
        self.stop_at = float(os.environ.get("HANOI_STOP_AT_EPOCH", "inf"))
        self._checkpoint_rng = None
        self._last_step_at = None

    def _write(self, record):
        if not distributed.is_rank0():
            return
        path = Path(self.config.job.path_local)
        path.mkdir(parents=True, exist_ok=True)
        record = {"time": time.time(), **record}
        with (path / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print("HANOI_METRICS " + json.dumps(record, allow_nan=False), flush=True)

    @staticmethod
    def _metrics(output, loss):
        metrics = {"loss": float(loss.detach())}
        for name in (
            "demo_sample_action_l1_loss",
            "demo_sample_action_mse_loss",
            "demo_sample_future_proprio_l1_loss",
            "demo_sample_future_image_l1_loss",
            "demo_sample_value_l1_loss",
        ):
            if name in output:
                value = float(output[name].detach())
                if math.isfinite(value):
                    metrics[name] = value
        if not math.isfinite(metrics["loss"]):
            raise FloatingPointError(f"Non-finite Hanoi objective: {metrics['loss']}")
        if "demo_sample_action_l1_loss" not in metrics:
            raise FloatingPointError("Hanoi demonstration action loss is missing or non-finite")
        return metrics

    @staticmethod
    def _average(records):
        keys = set().union(*(record.keys() for _, record in records))
        return {
            key: sum(n * row[key] for n, row in records if key in row) / sum(n for n, row in records if key in row)
            for key in keys
        }

    def on_train_start(self, model, iteration=0):
        self._last_step_at = time.time()
        self._write(
            {
                "event": "train_start",
                "iteration": iteration,
                "gpu": torch.cuda.get_device_name(),
                "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }
        )

    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration=0):
        self._train.append((len(data_batch["actions"]), self._metrics(output_batch, loss)))

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if iteration % self.config.trainer.logging_iter == 0 and self._train:
            self._write(
                {
                    "event": "train",
                    "iteration": iteration,
                    "metrics": self._average(self._train),
                    "elapsed_seconds": time.time() - self._started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )
            self._train.clear()
        now = time.time()
        step_seconds = 0 if self._last_step_at is None else now - self._last_step_at
        self._last_step_at = now
        # Stop before starting an optimizer step likely to cross the deadline.
        if now + step_seconds >= self.stop_at:
            self.trainer.stop_requested = True
            self._write({"event": "time_budget_reached", "iteration": iteration})

    def on_validation_start(self, model, dataloader_val, iteration=0):
        self._val.clear()

    def on_validation_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        self._val.append((len(data_batch["actions"]), self._metrics(output_batch, loss)))

    def on_validation_end(self, model, iteration=0):
        if self._val:
            self._write(
                {
                    "event": "validation",
                    "iteration": iteration,
                    "samples": sum(n for n, _ in self._val),
                    "metrics": self._average(self._val),
                    "metric_definition": "fixed-noise joint denoising loss, not rollout success",
                }
            )

    def on_save_checkpoint_start(self, model, iteration=0):
        numpy_state = np.random.get_state()
        self._checkpoint_rng = {
            "iteration": iteration,
            "python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch_cpu": torch.get_rng_state().tolist(),
            "torch_cuda": [state.tolist() for state in torch.cuda.get_rng_state_all()],
        }

    def on_save_checkpoint_success(self, iteration=0, elapsed_time=0):
        if not distributed.is_rank0():
            return
        directory = Path(self.config.job.path_local) / "checkpoints" / f"iter_{iteration:09}"
        # The upstream callback is also invoked in a finally block on errors.
        # Never report or publish RNG state for an incomplete checkpoint.
        try:
            validate_dcp_parts(directory)
            if self._checkpoint_rng is None or self._checkpoint_rng["iteration"] != iteration:
                raise RuntimeError("No matching RNG snapshot was captured at checkpoint start")
        except RuntimeError as error:
            self._write({"event": "checkpoint_incomplete", "iteration": iteration, "error": str(error)})
            return
        target = directory / "hanoi_rng.json"
        temporary = target.with_suffix(".json.tmp")
        with temporary.open("w") as stream:
            json.dump(self._checkpoint_rng, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        self._write({"event": "checkpoint_saved", "iteration": iteration, "seconds": elapsed_time})

    def on_load_checkpoint_start(self, model):
        from cosmos_policy.utils.hanoi_checkpoint import validate_model_metadata, validate_optimizer_metadata

        resume_keys, path = self.trainer.checkpointer.keys_to_resume_during_load()
        if path is not None and "model" in resume_keys:
            # Stock DCP uses a permissive load planner even with strict_resume.
            # Reject missing/wrong policy tensors before any state is mutated.
            validate_dcp_parts(path)
            read_rng_sidecar(path)
            validate_model_metadata(Path(path) / "model", model.state_dict())
        if path is not None and "optim" in resume_keys:
            validate_optimizer_metadata(Path(path) / "optim", model.net)

    def on_load_checkpoint_end(self, model, iteration=0, checkpoint_path=None):
        if not iteration:
            return
        validate_dcp_parts(checkpoint_path)
        state = read_rng_sidecar(checkpoint_path, iteration)
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Checkpoint RNG state does not match this iteration/GPU allocation")
        previous = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        try:
            python_state = state["python"]
            random.setstate((python_state[0], tuple(python_state[1]), python_state[2]))
            numpy_state = state["numpy"]
            np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
            torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))
            torch.cuda.set_rng_state_all([torch.tensor(value, dtype=torch.uint8) for value in state["torch_cuda"]])
        except Exception as error:
            # A malformed later field must not leave the earlier RNGs changed.
            random.setstate(previous[0])
            np.random.set_state(previous[1])
            torch.set_rng_state(previous[2])
            torch.cuda.set_rng_state_all(previous[3])
            raise RuntimeError(f"Invalid Hanoi checkpoint RNG state: {error}") from error
        self._write({"event": "rng_restored", "iteration": iteration})

    def on_train_end(self, model, iteration=0):
        self._write(
            {
                "event": "train_end",
                "iteration": iteration,
                "stopped_for_time_budget": bool(getattr(self.trainer, "stop_requested", False)),
            }
        )
