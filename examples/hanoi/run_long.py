"""Bounded position/jaw training with checkpoint evaluations and loss criteria.

No job submissions or retries: all stages share a single Slurm allocation.
"""

import fcntl
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from cosmos_policy.utils.hanoi_checkpoint import checkpoint_iteration, latest_complete_checkpoint
from cosmos_policy.utils.hanoi_schedule import aloha_training_schedule
from cosmos_policy.datasets.hanoi_data import FORMAT_VERSION, PROPRIO_COLUMNS, PROPRIO_SCHEMA
from examples.hanoi.run_pilot import execute_before, validate_inference_report


STAGES = (10000, 20000, 35000, 50000)
ALLOCATION_SECONDS = 36 * 3600
CHECKPOINT_EVERY = 2000
TARGET_L1 = 0.01
WINDOW_STEPS = 500


def scratch_headroom():
    output = subprocess.check_output(["/share/apps/local/bin/myquota"], text=True, timeout=20)
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    row = next(line for line in output.splitlines() if line.startswith("/scratch"))
    match = re.search(r"(\d+(?:\.\d+)?)TB/\S+\s+(\d+(?:\.\d+)?)TB\((\d+(?:\.\d+)?)%\)", row)
    if not match:
        raise RuntimeError(f"Cannot verify scratch quota: {row}")
    total, used, percent = map(float, match.groups())
    # Use the larger usage estimate and leave an extra 10 GB for display rounding.
    remaining = total - max(used, total * percent / 100) - 0.01
    return {"quota_tb": total, "used_tb": used, "used_percent": percent, "free_bytes_lower_bound": remaining * 10**12}


def assess_training(run, iteration, report, baseline):
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    # Logging occurs every ten optimizer steps. Require a full contiguous window,
    # and use the last occurrence if an interrupted attempt repeated a step.
    end = iteration // 10 * 10
    recent = {row["iteration"]: row for row in rows if row.get("event") == "train"}
    expected = list(range(end - WINDOW_STEPS + 10, end + 1, 10))
    complete = all(step in recent for step in expected)
    mean_l1 = None
    if complete:
        losses = [recent[step]["metrics"]["demo_sample_action_l1_loss"] for step in expected]
        if not all(math.isfinite(x) for x in losses):
            raise FloatingPointError("Non-finite training action L1")
        mean_l1 = sum(losses) / len(losses)
    metrics, reference = report["mean_metrics"], baseline["mean_metrics"]
    # Same 100 held-out anchors, inference settings and seed as the pilot.
    nonregression = all(metrics[key] <= reference[key] for key in reference if "xyz_l2_mm" in key)
    nonregression &= all(metrics[key] >= reference[key] - 0.02 for key in reference if "jaw_accuracy" in key)
    return {
        "iteration": iteration,
        "training_window_steps": WINDOW_STEPS,
        "training_window_complete": complete,
        "training_action_l1_mean": mean_l1,
        "training_action_l1_target": TARGET_L1,
        "training_target_met": complete and mean_l1 <= TARGET_L1,
        "offline_nonregression_vs_pilot": nonregression,
        "ready_for_robot_deployment": False,
        "note": "Offline criteria do not establish robot task success; deployment hardware and full task rollouts still need evaluation.",
    }


def main():
    root = Path(__file__).resolve().parents[2]
    name = os.environ["HANOI_RUN_NAME"]
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("HANOI_RUN_NAME must be one directory name")
    run = Path(os.environ["IMAGINAIRE_OUTPUT_ROOT"]) / "cosmos_policy/hanoi" / name
    metadata = json.loads((Path(os.environ["HANOI_METADATA_DIR"]) / "metadata.json").read_text())
    if metadata.get("format_version") != FORMAT_VERSION or metadata.get("proprio_columns") != list(PROPRIO_COLUMNS):
        raise RuntimeError("Long training requires the position/jaw metadata; velocity is excluded")
    if (run / "data_order.json").exists():
        contract = json.loads((run / "data_order.json").read_text())
        if contract.get("dataset_identity", {}).get("proprio_schema") != PROPRIO_SCHEMA:
            raise RuntimeError("Cannot resume a velocity-trained run with the new state layout; use a new run name")
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint = latest_complete_checkpoint(run) if (run / "checkpoints/latest_checkpoint.txt").exists() else None
    # Reference only the old pilot's offline action metrics. Its weights and
    # velocity-containing observations never enter this fresh training run.
    baseline_path = Path(os.environ["HANOI_BASELINE_REPORT"])
    baseline = validate_inference_report(baseline_path, 100)
    started = time.time()
    # Leave 60 seconds before Slurm's allocation limit and 20 minutes for the
    # final checkpoint/export/evaluation. Every stage shares these deadlines.
    deadline = started + ALLOCATION_SECONDS - 60
    stop_at = started + ALLOCATION_SECONDS - 1200
    os.environ["HANOI_STOP_AT_EPOCH"] = str(stop_at)
    os.environ.pop("HANOI_CONTINUATION_ANCHOR", None)
    os.environ["HANOI_TRAINING_SCHEDULE"] = "aloha"
    state = {
        "run_name": name, "job_id": os.environ.get("SLURM_JOB_ID"), "mode": "long_training_position_only",
        "started": started, "training_stop_at": stop_at, "pipeline_deadline": deadline,
        "resume_checkpoint": str(checkpoint) if checkpoint else None, "stages": list(STAGES), "events": [],
        "schedule": aloha_training_schedule(), "target_training_l1": TARGET_L1,
        "proprio_schema": PROPRIO_SCHEMA, "proprio_columns": list(PROPRIO_COLUMNS),
        "baseline_report": str(baseline_path),
        "effective_batch_size": 16, "checkpoint_every_steps": CHECKPOINT_EVERY,
    }
    python = str(root / ".venv/bin/python")

    def record(phase, **extra):
        state["phase"] = phase
        state["events"].append({"phase": phase, "time": time.time(), **extra})
        temporary = run / "pipeline.json.tmp"
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
        temporary.replace(run / "pipeline.json")
        print("HANOI_PIPELINE " + json.dumps(state["events"][-1], allow_nan=False), flush=True)

    def execute(phase, args, phase_deadline=deadline):
        record(phase, argv=args)
        execute_before(args, root, phase_deadline)

    def evaluate(current):
        export = run / "exports" / f"{current.name}.pt"
        if not export.exists():
            execute("export_cpu", [python, "examples/hanoi/export_checkpoint.py", "--checkpoint", str(current), "--output", str(export)])
        report_path = run / f"{current.name}_validation.json"
        execute("validation_actions", [
            python, "-m", "cosmos_policy.experiments.robot.hanoi.run_hanoi_eval",
            "--metadata-dir", os.environ["HANOI_METADATA_DIR"],
            "--data-dir", os.environ.get("HANOI_DATA_ROOT", "/scratch/cw5167/datasets"),
            "--embeddings", os.environ["HANOI_T5_EMBEDDINGS"],
            "--checkpoint", str(export), "--samples", "100", "--output", str(report_path),
        ])
        report = validate_inference_report(report_path, 100, export)
        if report.get("proprio_schema") != PROPRIO_SCHEMA:
            raise RuntimeError("Inference did not use the position/jaw observation schema")
        assessment = assess_training(run, checkpoint_iteration(current), report, baseline)
        record("stage_evaluated", checkpoint=str(current), exported_policy=str(export), validation_report=str(report_path), assessment=assessment)
        return assessment

    try:
        record("training_start")
        train = [
            python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
            "-m", "cosmos_policy.scripts.train", "--config=cosmos_policy/config/hanoi_config.py", "--",
            "experiment=cosmos_predict2_2b_hanoi",
        ]
        if checkpoint is None:
            quota = scratch_headroom()
            if quota["free_bytes_lower_bound"] < 200_000_000_000:
                raise RuntimeError("Insufficient scratch headroom for initial qualification")
            execute("qualify_train_and_save", train + ["trainer.max_iter=2", "checkpoint.save_iter=2"], min(stop_at + 900, time.time() + 1200))
            checkpoint = latest_complete_checkpoint(run)
        qualification = run / "qualification_inference.json"
        if not qualification.exists():
            execute("qualify_reload_and_inference", [
                python, "-m", "cosmos_policy.experiments.robot.hanoi.run_hanoi_eval",
                "--metadata-dir", os.environ["HANOI_METADATA_DIR"],
                "--data-dir", os.environ.get("HANOI_DATA_ROOT", "/scratch/cw5167/datasets"),
                "--embeddings", os.environ["HANOI_T5_EMBEDDINGS"],
                "--checkpoint", str(checkpoint), "--samples", "2", "--output", str(qualification),
            ], min(deadline, time.time() + 1200))
        qualified = validate_inference_report(qualification, 2)
        if qualified.get("proprio_schema") != PROPRIO_SCHEMA or Path(qualified["checkpoint"]).resolve().parent != (run / "checkpoints").resolve():
            raise RuntimeError("Qualification must use this run's position/jaw checkpoint")
        assessment = None
        stop_reason = "maximum_steps"
        for target in STAGES:
            current_step = checkpoint_iteration(checkpoint)
            if current_step >= target:
                continue
            if time.time() >= stop_at:
                stop_reason = "time_budget"
                break
            quota = scratch_headroom()
            # Bound this stage's full checkpoints, its final checkpoint and export,
            # and retain an additional 150 GB of shared scratch headroom.
            saves = target // CHECKPOINT_EVERY - current_step // CHECKPOINT_EVERY + 1
            required_bytes = saves * 28_000_000_000 + 5_000_000_000 + 150_000_000_000
            record("stage_preflight", target_step=target, quota=quota, required_free_bytes=required_bytes)
            if quota["free_bytes_lower_bound"] < required_bytes:
                stop_reason = "scratch_headroom"
                break
            execute("fine_tune_resume", train + [f"trainer.max_iter={target}",
                f"checkpoint.save_iter={CHECKPOINT_EVERY}", "trainer.validation_iter=250",
            ], stop_at + 900)
            checkpoint = latest_complete_checkpoint(run)
            if checkpoint_iteration(checkpoint) <= current_step:
                raise RuntimeError("Training returned without advancing the checkpoint")
            assessment = evaluate(checkpoint)
            if assessment["training_target_met"] and assessment["offline_nonregression_vs_pilot"]:
                stop_reason = "loss_target_and_offline_checks"
                break
            if checkpoint_iteration(checkpoint) < target:
                stop_reason = "time_budget"
                break
        if assessment is None:
            assessment = evaluate(checkpoint)
        record("training_complete", checkpoint=str(checkpoint), exported_policy=str(run / "exports" / f"{checkpoint.name}.pt"), stop_reason=stop_reason, assessment=assessment)
    except Exception as error:
        record("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
