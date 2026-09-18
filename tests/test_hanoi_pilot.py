"""Test pilot orchestration and deadlines without launching training or CUDA."""

import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_checkpoint_path = Path(__file__).resolve().parents[1] / "cosmos_policy/utils/hanoi_checkpoint.py"
_checkpoint_spec = importlib.util.spec_from_file_location("_hanoi_pilot_checkpoint_test", _checkpoint_path)
_checkpoint_module = importlib.util.module_from_spec(_checkpoint_spec)
_checkpoint_spec.loader.exec_module(_checkpoint_module)
# The helper is real; bypass only the unrelated logging/Hub imports in utils' __init__.
with patch.dict(sys.modules, {"cosmos_policy.utils.hanoi_checkpoint": _checkpoint_module}):
    from examples.hanoi import run_pilot


def report(checkpoint, samples):
    return {
        "checkpoint": str(checkpoint),
        "num_samples": samples,
        "samples": [{}] * samples,
        "split": "val",
        "mean_metrics": {"chunk63_xyz_rmse_mm": 1.0},
        "latency_mean_seconds": 0.2,
    }


def test_pipeline_qualifies_identical_batch_settings_before_resume(monkeypatch, tmp_path):
    monkeypatch.setenv("HANOI_RUN_NAME", "test_run")
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("HANOI_METADATA_DIR", str(tmp_path / "metadata"))
    monkeypatch.setenv("HANOI_T5_EMBEDDINGS", str(tmp_path / "custom_embeddings.pkl"))
    monkeypatch.setenv("HANOI_DATA_ROOT", str(tmp_path / "raw"))
    monkeypatch.setenv("HANOI_STOP_AT_EPOCH", str(run_pilot.time.time() + 6600))
    overrides = ["dataloader_train.batch_size=1", "trainer.grad_accum_iter=4", "trainer.max_iter=100"]
    monkeypatch.setattr(sys, "argv", ["run_pilot.py", *overrides])
    run = tmp_path / "cosmos_policy/hanoi/test_run"
    current = [run / "checkpoints/iter_000000002"]
    monkeypatch.setattr(run_pilot, "latest_complete_checkpoint", lambda path: current[0])
    commands = []

    def execute(args, root, deadline):
        commands.append(args)
        assert deadline <= run_pilot.time.time() + 7200
        if "cosmos_policy.scripts.train" in args:
            if len(commands) > 1:
                current[0] = run / "checkpoints/iter_000000003"
        elif "cosmos_policy.experiments.robot.hanoi.run_hanoi_eval" in args:
            output = Path(args[args.index("--output") + 1])
            checkpoint = args[args.index("--checkpoint") + 1]
            samples = int(args[args.index("--samples") + 1])
            output.write_text(json.dumps(report(checkpoint, samples)))
        else:
            output = Path(args[args.index("--output") + 1])
            output.parent.mkdir(parents=True)
            output.write_bytes(b"export fixture")

    monkeypatch.setattr(run_pilot, "execute_before", execute)
    run_pilot.main()
    assert len(commands) == 5
    assert commands[0][-len(overrides) - 2 :] == [*overrides, "trainer.max_iter=2", "checkpoint.save_iter=2"]
    assert commands[2][-len(overrides) :] == overrides
    for command in (commands[1], commands[4]):
        assert command[command.index("--data-dir") + 1] == str(tmp_path / "raw")
        assert command[command.index("--embeddings") + 1] == str(tmp_path / "custom_embeddings.pkl")
    summary = json.loads((run / "pipeline.json").read_text())
    assert summary["phase"] == "pilot_complete"
    assert [row["phase"] for row in summary["events"]][:3] == [
        "qualify_train_and_save",
        "qualify_reload_and_inference",
        "fine_tune_resume",
    ]


@pytest.mark.parametrize(
    "bad",
    [
        "{",
        {},
        {**report("weights.pt", 2), "num_samples": 1},
        {**report("weights.pt", 2), "mean_metrics": {"loss": float("nan")}},
    ],
)
def test_partial_or_invalid_inference_reports_are_not_qualification(tmp_path, bad):
    path = tmp_path / "qualification.json"
    path.write_text(bad if isinstance(bad, str) else json.dumps(bad))
    with pytest.raises(RuntimeError, match="Incomplete Hanoi inference report"):
        run_pilot.validate_inference_report(path, 2)


def test_inference_report_must_match_requested_checkpoint(tmp_path):
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(report(tmp_path / "one.pt", 2)))
    with pytest.raises(RuntimeError, match="different checkpoint"):
        run_pilot.validate_inference_report(path, 2, tmp_path / "two.pt")


def test_deadline_refuses_launch_and_stops_only_its_child_group(monkeypatch, tmp_path):
    monkeypatch.setattr(run_pilot.time, "time", lambda: 100)
    launched = []
    signals = []

    class Child:
        pid = 12345
        waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("owned command", timeout)
            return -15

    def popen(args, **kwargs):
        launched.append(kwargs)
        return Child()

    monkeypatch.setattr(run_pilot.subprocess, "Popen", popen)
    monkeypatch.setattr(run_pilot.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(TimeoutError, match="deadline has been reached"):
        run_pilot.execute_before(["owned command"], tmp_path, 99)
    assert not launched
    with pytest.raises(TimeoutError, match="phase exceeded"):
        run_pilot.execute_before(["owned command"], tmp_path, 101)
    assert launched == [{"cwd": tmp_path, "start_new_session": True}]
    assert signals == [(12345, signal.SIGTERM)]
