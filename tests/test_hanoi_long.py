"""Check long-run resume rates, deadline handling, and held-out stopping rules."""

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed.checkpoint as dcp

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_long_schedule_test", _root / "cosmos_policy/utils/hanoi_schedule.py")
_schedule = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_schedule)
continuation_schedule = _schedule.continuation_schedule

# Exercise the real LR scheduler without importing unrelated GPU-only Megatron
# and Transformer Engine modules on CPU test hosts.
import cosmos_policy._src.imaginaire.utils as _utils

_spec = importlib.util.spec_from_file_location("_long_lr_test", _root / "cosmos_policy/_src/imaginaire/functional/lr_scheduler.py")
_lr = importlib.util.module_from_spec(_spec)
with patch.object(_utils, "distributed", types.SimpleNamespace(), create=True), patch.object(_utils, "log", types.SimpleNamespace(), create=True):
    _spec.loader.exec_module(_lr)
LambdaLinearScheduler = _lr.LambdaLinearScheduler


def make_scheduler(schedule):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    function = LambdaLinearScheduler(**schedule)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, function.schedule)
    return optimizer, scheduler


def test_real_dcp_resume_preserves_rate_and_extended_schedule(tmp_path):
    pilot = dict(warm_up_steps=[100], cycle_lengths=[10000], f_start=[1e-6], f_max=[1.0], f_min=[0.1])
    old_optimizer, old = make_scheduler(pilot)
    for _ in range(3341):
        old_optimizer.step()
        old.step()
    dcp.save(old.state_dict(), checkpoint_id=tmp_path / "pilot")
    optimizer, resumed = make_scheduler(continuation_schedule(3341))
    state = resumed.state_dict()
    dcp.load(state, checkpoint_id=tmp_path / "pilot")
    resumed.load_state_dict(state)
    optimizer.load_state_dict(old_optimizer.state_dict())
    assert resumed.last_epoch == 3341
    assert resumed.get_last_lr() == old.get_last_lr() == [optimizer.param_groups[0]["lr"]]
    rate = resumed.lr_lambdas[0]
    assert rate(3341) * 1e-5 == pytest.approx(old.get_last_lr()[0])
    assert rate(3342) < rate(3341)
    assert rate(10001) > 0.3
    assert rate(20000) == pytest.approx(0.3)
    assert rate(20001) == pytest.approx(0.06)
    assert rate(50000) == pytest.approx(0.06)
    for _ in range(20001 - 3341):
        optimizer.step()
        resumed.step()
    dcp.save(resumed.state_dict(), checkpoint_id=tmp_path / "long")
    _, again = make_scheduler(continuation_schedule(3341))
    state = again.state_dict()
    dcp.load(state, checkpoint_id=tmp_path / "long")
    again.load_state_dict(state)
    assert again.last_epoch == 20001
    assert again.get_last_lr()[0] == pytest.approx(6e-7)
    assert again.lr_lambdas[0](50000) == pytest.approx(0.06)


@pytest.fixture
def long_module():
    # Like the pilot tests, retain the actual checkpoint implementation while
    # bypassing unrelated Hub imports in the package initializer.
    path = Path(__file__).resolve().parents[1] / "cosmos_policy/utils/hanoi_checkpoint.py"
    spec = importlib.util.spec_from_file_location("_long_checkpoint_test", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with patch.dict(sys.modules, {"cosmos_policy.utils.hanoi_checkpoint": helper, "cosmos_policy.utils.hanoi_schedule": _schedule}):
        from examples.hanoi import run_long
    return run_long


def inference_report(checkpoint, error=1.0):
    return {
        "proprio_schema": "xyz_measured_jaw_v1",
        "checkpoint": str(checkpoint), "num_samples": 100, "samples": [{}] * 100,
        "split": "val", "mean_metrics": {"first_xyz_l2_mm": error, "chunk63_xyz_l2_mm": error, "first_jaw_accuracy": 0.98},
        "latency_mean_seconds": 0.25,
    }


def write_metrics(run, end, loss):
    with (run / "metrics.jsonl").open("a") as stream:
        for step in range(end - 490, end + 1, 10):
            stream.write(json.dumps({"event": "train", "iteration": step, "metrics": {"demo_sample_action_l1_loss": loss}}) + "\n")


def test_loss_target_requires_complete_window_and_offline_check(long_module, tmp_path):
    baseline = inference_report("baseline.pt")
    write_metrics(tmp_path, 10000, 0.009)
    good = long_module.assess_training(tmp_path, 10000, baseline, baseline)
    assert good["training_target_met"] and good["offline_nonregression_vs_pilot"]
    worse = long_module.assess_training(tmp_path, 10000, inference_report("new.pt", error=2), baseline)
    assert worse["training_target_met"] and not worse["offline_nonregression_vs_pilot"]
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    (tmp_path / "metrics.jsonl").write_text("\n".join(lines[1:]) + "\n")
    assert not long_module.assess_training(tmp_path, 10000, baseline, baseline)["training_target_met"]


@pytest.mark.parametrize("time_stop", [False, True])
@pytest.mark.parametrize("fresh", [False, True])
def test_long_pipeline_resumes_evaluates_and_respects_shared_budget(long_module, monkeypatch, tmp_path, time_stop, fresh):
    run = tmp_path / "cosmos_policy/hanoi/run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "exports").mkdir()
    baseline_path = run / "exports/iter_000003341.pt"
    baseline_path.write_bytes(b"baseline")
    (run / "iter_000003341_validation.json").write_text(json.dumps(inference_report(baseline_path)))
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "metadata.json").write_text(json.dumps({"format_version": 2, "proprio_columns": [0, 1, 2, 6]}))
    for key, value in {"HANOI_RUN_NAME": "run", "IMAGINAIRE_OUTPUT_ROOT": str(tmp_path), "HANOI_METADATA_DIR": str(metadata), "HANOI_T5_EMBEDDINGS": "embeddings", "HANOI_BASELINE_REPORT": str(run / "iter_000003341_validation.json")}.items():
        monkeypatch.setenv(key, value)
    # A inherited legacy schedule must never survive the position-only launch.
    monkeypatch.setenv("HANOI_CONTINUATION_ANCHOR", "3341")
    monkeypatch.setattr(long_module.time, "time", lambda: 1000)
    monkeypatch.setattr(long_module, "scratch_headroom", lambda: {"free_bytes_lower_bound": 2e12})
    current = [run / "checkpoints/iter_000002000"]
    if not fresh:
        (run / "checkpoints/latest_checkpoint.txt").write_text(current[0].name)
        (run / "data_order.json").write_text(json.dumps({"dataset_identity": {"proprio_schema": "xyz_measured_jaw_v1"}}))
        qualified = inference_report(current[0])
        qualified.update(num_samples=2, samples=[{}, {}])
        (run / "qualification_inference.json").write_text(json.dumps(qualified))
    monkeypatch.setattr(long_module, "latest_complete_checkpoint", lambda _: current[0])
    calls = []

    def execute(args, root, deadline):
        calls.append(args)
        assert deadline <= 1000 + long_module.ALLOCATION_SECONDS - 60
        if "cosmos_policy.scripts.train" in args:
            assert float(long_module.os.environ["HANOI_STOP_AT_EPOCH"]) == 1000 + 36 * 3600 - 1200
            assert "HANOI_CONTINUATION_ANCHOR" not in long_module.os.environ
            assert long_module.os.environ["HANOI_TRAINING_SCHEDULE"] == "aloha"
            if "trainer.max_iter=2" in args:
                assert fresh
                current[0] = run / "checkpoints/iter_000000002"
                return
            assert "trainer.max_iter=10000" in args
            step = 9000 if time_stop else 10000
            current[0] = run / "checkpoints" / f"iter_{step:09d}"
            write_metrics(run, step, 0.02 if time_stop else 0.009)
        elif "cosmos_policy.experiments.robot.hanoi.run_hanoi_eval" in args:
            output = Path(args[args.index("--output") + 1])
            checkpoint = args[args.index("--checkpoint") + 1]
            samples = int(args[args.index("--samples") + 1])
            report = inference_report(checkpoint)
            report.update(num_samples=samples, samples=[{}] * samples)
            output.write_text(json.dumps(report))
        else:
            Path(args[args.index("--output") + 1]).write_bytes(b"export")

    monkeypatch.setattr(long_module, "execute_before", execute)
    long_module.main()
    assert len(calls) == (5 if fresh else 3)
    result = json.loads((run / "pipeline.json").read_text())
    assert result["phase"] == "training_complete"
    assert result["events"][-1]["stop_reason"] == ("time_budget" if time_stop else "loss_target_and_offline_checks")


def test_legacy_velocity_run_cannot_be_resumed(long_module, monkeypatch, tmp_path):
    run = tmp_path / "cosmos_policy/hanoi/old_run"
    run.mkdir(parents=True)
    (run / "data_order.json").write_text(json.dumps({"dataset_identity": {"dataset": "cosmos_hanoi_v1"}}))
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "metadata.json").write_text(json.dumps({"format_version": 2, "proprio_columns": [0, 1, 2, 6]}))
    monkeypatch.setenv("HANOI_RUN_NAME", "old_run")
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("HANOI_METADATA_DIR", str(metadata))
    with pytest.raises(RuntimeError, match="Cannot resume a velocity-trained run"):
        long_module.main()
    assert not (run / "pipeline.json").exists()


def test_fresh_aloha_schedule_and_checkpoint_resume(tmp_path):
    optimizer, scheduler = make_scheduler(_schedule.aloha_training_schedule())
    rate = scheduler.lr_lambdas[0]
    assert rate(0) == pytest.approx(1e-6)
    assert rate(2000) == pytest.approx(1.0)
    assert rate(20000) == pytest.approx(0.3)
    assert rate(20001) == pytest.approx(0.06)
    assert rate(50000) == pytest.approx(0.06)
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    dcp.save(scheduler.state_dict(), checkpoint_id=tmp_path / "fresh")
    _, resumed = make_scheduler(_schedule.aloha_training_schedule())
    state = resumed.state_dict()
    dcp.load(state, checkpoint_id=tmp_path / "fresh")
    resumed.load_state_dict(state)
    assert resumed.last_epoch == 2 and resumed.get_last_lr() == scheduler.get_last_lr()


def test_quota_uses_conservative_display_bounds(long_module, monkeypatch):
    monkeypatch.setattr(long_module.subprocess, "check_output", lambda *a, **k: "/scratch $SCRATCH NO/YES 5.00TB/5.00M 3.27TB(65.49%)/569729(11.00%)\n")
    assert long_module.scratch_headroom()["free_bytes_lower_bound"] == pytest.approx(1.7155e12)
