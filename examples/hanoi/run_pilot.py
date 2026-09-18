"""Bounded single-H100 qualification, fine-tuning, export, and validation.

Invoked by train.sbatch. All child processes finish within the same allocation;
the training callback reserves time for checkpoint export and final evaluation.
"""

import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from cosmos_policy.utils.hanoi_checkpoint import latest_complete_checkpoint


def execute_before(args, root, deadline):
    """Bound the entire child process group, including torchrun workers."""
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("Hanoi pilot deadline has been reached")
    process = subprocess.Popen(args, cwd=root, start_new_session=True)
    try:
        returncode = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=min(10, max(0, deadline - time.time())))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise TimeoutError(f"Hanoi pilot phase exceeded its deadline: {args}") from error
    if returncode:
        raise subprocess.CalledProcessError(returncode, args)


def validate_inference_report(path, samples, checkpoint=None):
    """Do not mistake a partial JSON write for completed GPU qualification."""
    try:
        report = json.loads(Path(path).read_text())
        if report["num_samples"] != samples or len(report["samples"]) != samples or report["split"] != "val":
            raise ValueError("wrong validation split or incomplete sample count")
        if checkpoint is not None and Path(report["checkpoint"]).resolve() != Path(checkpoint).resolve():
            raise ValueError("inference report refers to a different checkpoint")
        metrics = report["mean_metrics"]
        if not metrics or not all(math.isfinite(float(value)) for value in metrics.values()):
            raise ValueError("missing or non-finite inference metrics")
        if not math.isfinite(report["latency_mean_seconds"]) or report["latency_mean_seconds"] <= 0:
            raise ValueError("invalid inference timing")
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RuntimeError(f"Incomplete Hanoi inference report {path}: {error}") from error
    return report


def main():
    root = Path(__file__).resolve().parents[2]
    name = os.environ["HANOI_RUN_NAME"]
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("HANOI_RUN_NAME must be one directory name")
    run = Path(os.environ["IMAGINAIRE_OUTPUT_ROOT"]) / "cosmos_policy/hanoi" / name
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    python = str(root / ".venv/bin/python")
    state = {
        "run_name": name,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "started": time.time(),
        "events": [],
        "direction": os.environ.get("HANOI_DIRECTION"),
    }
    stop_at = min(float(os.environ["HANOI_STOP_AT_EPOCH"]), state["started"] + 6600)
    if not math.isfinite(stop_at):
        raise ValueError("HANOI_STOP_AT_EPOCH must be a finite Unix timestamp")
    os.environ["HANOI_STOP_AT_EPOCH"] = str(stop_at)
    deadline = min(stop_at + 600, state["started"] + 7200)
    state.update(training_stop_at=stop_at, pipeline_deadline=deadline)

    def record(phase, **extra):
        state["phase"] = phase
        state["events"].append({"phase": phase, "time": time.time(), **extra})
        temporary = run / "pipeline.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(run / "pipeline.json")
        print("HANOI_PIPELINE " + json.dumps(state["events"][-1]), flush=True)

    def execute(phase, args, phase_deadline=None):
        record(phase, argv=args)
        execute_before(args, root, min(deadline, phase_deadline or deadline))

    train = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        "-m",
        "cosmos_policy.scripts.train",
        "--config=cosmos_policy/config/hanoi_config.py",
        "--",
        "experiment=cosmos_predict2_2b_hanoi",
    ]
    metadata = os.environ["HANOI_METADATA_DIR"]
    evaluation = [
        python,
        "-m",
        "cosmos_policy.experiments.robot.hanoi.run_hanoi_eval",
        "--metadata-dir",
        metadata,
        "--data-dir",
        os.environ.get("HANOI_DATA_ROOT", "/scratch/cw5167/datasets"),
        "--embeddings",
        os.environ.get("HANOI_T5_EMBEDDINGS", str(Path(metadata) / "t5_embeddings.pkl")),
    ]
    try:
        # Use the same batch/accumulation settings for qualification and training
        # so optimizer, sampler offset, and noise-state resumption are meaningful.
        if not (run / "checkpoints/latest_checkpoint.txt").is_file():
            execute(
                "qualify_train_and_save",
                train + sys.argv[1:] + ["trainer.max_iter=2", "checkpoint.save_iter=2"],
                stop_at,
            )
        checkpoint = latest_complete_checkpoint(run)
        qualification = run / "qualification_inference.json"
        try:
            previous = validate_inference_report(qualification, 2)
            if Path(previous["checkpoint"]).resolve().parent != (run / "checkpoints").resolve():
                raise RuntimeError("Qualification checkpoint belongs to another run")
        except RuntimeError:
            execute(
                "qualify_reload_and_inference",
                evaluation
                + [
                    "--checkpoint",
                    str(checkpoint),
                    "--samples",
                    "2",
                    "--output",
                    str(qualification),
                ],
                stop_at,
            )
            validate_inference_report(qualification, 2, checkpoint)
        if time.time() < stop_at:
            execute("fine_tune_resume", train + sys.argv[1:])
        else:
            record("training_budget_exhausted_after_qualification")
        checkpoint = latest_complete_checkpoint(run)
        export = run / "exports" / f"{checkpoint.name}.pt"
        if not export.exists():
            execute(
                "export_cpu",
                [
                    python,
                    "examples/hanoi/export_checkpoint.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(export),
                ],
            )
        validation = run / f"{checkpoint.name}_validation.json"
        execute(
            "validation_actions",
            evaluation
            + [
                "--checkpoint",
                str(export),
                "--samples",
                "100",
                "--output",
                str(validation),
            ],
        )
        validate_inference_report(validation, 100, export)
        record(
            "pilot_complete",
            checkpoint=str(checkpoint),
            exported_policy=str(export),
            note="Pilot completed; offline metrics do not establish task success or convergence.",
        )
    except Exception as error:
        record("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
