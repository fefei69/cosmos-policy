"""Joint training monitor without per-micro-batch host synchronisation.

The joint_v3 monitor converted the loss and five metric tensors to Python floats
after every micro-batch, draining the CUDA queue up to 96 times per optimizer
update. This subclass keeps detached scalars on the GPU and reads them back once
per logging interval. A non-finite loss is therefore detected up to
``trainer.logging_iter`` updates late instead of immediately; the native gradient
clipping callback still guards each update.
"""
import math

import torch

from cosmos_policy.utils.hanoi_joint_training import HanoiJointTrainingMonitor

METRIC_NAMES = (
    "demo_sample_action_l1_loss",
    "demo_sample_action_mse_loss",
    "demo_sample_future_proprio_l1_loss",
    "demo_sample_future_image_l1_loss",
    "demo_sample_value_l1_loss",
)


def records_to_floats(pending):
    """One device-to-host copy for a list of (sample_count, {name: 0-d tensor})."""
    if not pending:
        return []
    names = ["loss", *METRIC_NAMES]
    rows = []
    for _, record in pending:
        rows.append(torch.stack([
            record[name].reshape(()).float() if name in record else torch.full((), math.nan, device=record["loss"].device)
            for name in names]))
    values = torch.stack(rows).tolist()
    converted = []
    for (count, record), row in zip(pending, values):
        metrics = {}
        for name, value in zip(names, row):
            if name in record and math.isfinite(value):
                metrics[name] = value
        if "loss" not in metrics:
            raise FloatingPointError(f"Non-finite Hanoi objective: {row[0]}")
        if "demo_sample_action_l1_loss" not in metrics:
            raise FloatingPointError("Hanoi demonstration action loss is missing or non-finite")
        converted.append((count, metrics))
    return converted


class HanoiWaypointTrainingMonitor(HanoiJointTrainingMonitor):
    def __init__(self):
        super().__init__()
        self._pending = []

    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration=0):
        record = {"loss": loss.detach()}
        for name in METRIC_NAMES:
            if name in output_batch:
                record[name] = output_batch[name].detach()
        self._pending.append((len(data_batch["actions"]), record))

    def flush_pending(self):
        self._train.extend(records_to_floats(self._pending))
        self._pending.clear()

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if iteration % self.config.trainer.logging_iter == 0:
            self.flush_pending()
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration=iteration)

    def on_train_end(self, model, iteration=0):
        self._pending.clear()
        super().on_train_end(model, iteration=iteration)
