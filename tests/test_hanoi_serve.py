"""CPU tests for the waypoint policy server: codec, validation, and the wire protocol."""
import json
import threading

import numpy as np
import pytest
from websockets.sync.client import connect

from cosmos_policy.experiments.robot.hanoi.serve_waypoint import (
    PROMPT,
    WaypointPolicyServer,
    build_metadata,
    packb,
    unpackb,
    validate_observation,
    validate_reply,
)


def observation(**overrides):
    base = {
        "observation/image": np.full((224, 224, 3), 7, np.uint8),
        "observation/state": np.arange(7, dtype=np.float32) / 10,
        "observation/cartesian_position": np.array([0.49, -0.05, 0.19], np.float32),
        "prompt": PROMPT,
    }
    base.update(overrides)
    return base


class FakePolicy:
    def __init__(self):
        self.calls = []

    def infer(self, obs, *, seed, dream=False):
        self.calls.append((obs, seed))
        actions = np.tile(np.r_[obs["observation/cartesian_position"], 1.0], (8, 1)).astype(np.float32)
        actions[:, 0] += np.arange(8, dtype=np.float32) * 0.001
        reply = {"actions": actions, "commit_count": 1, "reference_rate_hz": None}
        if dream:
            reply["future_image"] = np.full((224, 224, 3), 9, np.uint8)
            reply["value"] = 0.5
        return reply


def test_codec_roundtrips_arrays_scalars_and_nested_dicts():
    payload = {
        "image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "state": np.array([0.1, -0.2], np.float32),
        "flag": np.bool_(True),
        "scalar": np.float64(2.5),
        "nested": {"text": "abc", "list": [1, 2.5, "x"]},
    }
    restored = unpackb(packb(payload))
    np.testing.assert_array_equal(restored["image"], payload["image"])
    assert restored["image"].dtype == np.uint8 and restored["state"].dtype == np.float32
    np.testing.assert_array_equal(restored["state"], payload["state"])
    assert restored["flag"] == np.bool_(True) and restored["scalar"] == 2.5
    assert restored["nested"] == payload["nested"]
    with pytest.raises(ValueError, match="Unsupported dtype"):
        packb({"bad": np.array([object()])})


def test_observation_validation():
    clean = validate_observation(observation())
    assert clean["observation/state"].dtype == np.float32 and clean["prompt"] == PROMPT
    clean = validate_observation(observation(prompt=PROMPT.encode()))
    assert clean["prompt"] == PROMPT
    cases = {
        "contract crop": observation(**{"observation/image": np.zeros((480, 640, 3), np.uint8)}),
        "contract crop ": observation(**{"observation/image": np.zeros((224, 224, 3), np.float32)}),
        "joint angles": observation(**{"observation/state": np.zeros(4, np.float32)}),
        "joint angles ": observation(**{"observation/state": np.r_[np.zeros(6), np.nan]}),
        "measured XYZ": observation(**{"observation/cartesian_position": np.zeros(2)}),
        "AAAA to CCCC": observation(prompt="Move all four rings from peg C to peg A following Tower of Hanoi rules."),
    }
    for message, bad in cases.items():
        with pytest.raises(ValueError, match=message.strip()):
            validate_observation(bad)
    with pytest.raises(ValueError, match="Missing observation key"):
        validate_observation({"observation/image": np.zeros((224, 224, 3), np.uint8)})


def test_reply_validation():
    good = validate_reply({"actions": np.zeros((8, 4)), "commit_count": 1})
    assert good["actions"].dtype == np.float32 and good["commit_count"] == 1
    with pytest.raises(ValueError, match="eight finite"):
        validate_reply({"actions": np.zeros((63, 4))})
    with pytest.raises(ValueError, match="exactly one destination"):
        validate_reply({"actions": np.zeros((8, 4)), "commit_count": 2})
    assert "future_image" not in good
    dreamed = validate_reply({"actions": np.zeros((8, 4)), "future_image": np.zeros((224, 224, 3), np.uint8), "value": 0.25})
    assert dreamed["future_image"].shape == (224, 224, 3) and dreamed["value"] == 0.25
    with pytest.raises(ValueError, match="uint8 image"):
        validate_reply({"actions": np.zeros((8, 4)), "future_image": np.zeros((224, 224, 3), np.float32), "value": 0.5})
    with pytest.raises(ValueError, match="value must lie"):
        validate_reply({"actions": np.zeros((8, 4)), "future_image": np.zeros((224, 224, 3), np.uint8), "value": 1.5})


def test_metadata_carries_contract_and_hashes(tmp_path):
    run = tmp_path / "run"
    (run / "exports").mkdir(parents=True)
    export = run / "exports/iter_000000001.pt"
    export.write_bytes(b"weights")
    (run / "joint_contract.json").write_text(json.dumps(
        {"contract": "hanoi_waypoint_v4_cosmos_v1", "statistics_sha256": "s", "raw_sha256": "r", "max_updates": 8000}))
    metadata_dir = tmp_path / "waypoint_v4"
    metadata_dir.mkdir()
    (metadata_dir / "dataset_statistics.json").write_text("{}")
    (metadata_dir / "metadata.json").write_text(json.dumps(
        {"contract": "hanoi_waypoint_v4_cosmos_v1", "deployment": {"version": 4, "action_horizon": 8}}))
    actions = np.zeros((3, 8, 4), np.float32)
    actions[:, :, :3] = [0.4961, -0.0572, 0.0877]
    actions[1, 2, :3] = [0.4926, -0.0562, 0.19110004]  # rounds onto the same 0.1 mm grid point
    actions[2, 5, :3] = [0.4926, -0.0562, 0.1911]
    np.savez(metadata_dir / "train.npz", actions=actions)
    identity = build_metadata(export, metadata_dir, seed=1, denoising_steps=5, gpu="test")["cosmos_hanoi"]
    assert identity["contract"] == {"version": 4, "action_horizon": 8}
    assert identity["destinations"] == [[0.4926, -0.0562, 0.1911], [0.4961, -0.0572, 0.0877]]
    assert identity["export_sha256"] == __import__("hashlib").sha256(b"weights").hexdigest()
    assert identity["statistics_sha256"] == __import__("hashlib").sha256(b"{}").hexdigest()
    assert identity["training_identity"]["max_updates"] == 8000 and identity["commit_count"] == 1
    assert identity["dream"] is False
    assert build_metadata(export, metadata_dir, seed=1, denoising_steps=5, gpu="test", dream=True)["cosmos_hanoi"]["dream"] is True


@pytest.fixture
def server():
    policy = FakePolicy()
    instance = WaypointPolicyServer(policy, {"cosmos_hanoi": {"contract": {"version": 4}}}, port=0, seed=3)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    assert instance.ready.wait(5)
    yield instance, policy
    instance.stop()
    thread.join(5)


def test_wire_protocol_roundtrip_and_error_close(server):
    instance, policy = server
    uri = f"ws://127.0.0.1:{instance.bound_port}"
    with connect(uri, compression=None, open_timeout=5) as ws:
        metadata = unpackb(ws.recv(timeout=5))
        assert metadata == {"cosmos_hanoi": {"contract": {"version": 4}}}
        ws.send(packb(observation()))
        reply = unpackb(ws.recv(timeout=5))
        assert reply["actions"].shape == (8, 4) and reply["actions"].dtype == np.float32
        assert reply["commit_count"] == 1 and reply["server_timing"]["infer_ms"] >= 0
        assert "future_image" not in reply and "value" not in reply
        np.testing.assert_allclose(reply["actions"][0, :3], [0.49, -0.05, 0.19])
        ws.send(packb(observation()))
        second = unpackb(ws.recv(timeout=5))
        assert "prev_total_ms" in second["server_timing"]
        assert policy.calls[0][1] == 3 and policy.calls[0][0]["observation/state"].dtype == np.float32
        # A bad request returns the traceback as text, then the server closes the connection.
        ws.send(packb(observation(**{"observation/state": np.zeros(4, np.float32)})))
        error = ws.recv(timeout=5)
        assert isinstance(error, str) and "joint angles" in error
        with pytest.raises(Exception):
            ws.recv(timeout=5)
    # The server stays up for the next client.
    with connect(uri, compression=None, open_timeout=5) as ws:
        assert unpackb(ws.recv(timeout=5))["cosmos_hanoi"]["contract"]["version"] == 4


def test_dreaming_server_returns_the_future_frame_and_value():
    instance = WaypointPolicyServer(FakePolicy(), {"cosmos_hanoi": {"dream": True}}, port=0, seed=3, dream=True)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    assert instance.ready.wait(5)
    try:
        with connect(f"ws://127.0.0.1:{instance.bound_port}", compression=None, open_timeout=5) as ws:
            assert unpackb(ws.recv(timeout=5))["cosmos_hanoi"]["dream"] is True
            ws.send(packb(observation()))
            reply = unpackb(ws.recv(timeout=5))
            assert reply["actions"].shape == (8, 4) and reply["value"] == 0.5
            assert reply["future_image"].shape == (224, 224, 3) and reply["future_image"].dtype == np.uint8
            assert int(reply["future_image"][0, 0, 0]) == 9
    finally:
        instance.stop()
        thread.join(5)


def test_dense_server_play_identity_and_validators():
    """The WebSocket server's play mode: contract seven, 81 board sentences, goal board echoed."""
    from cosmos_policy.datasets.hanoi_play_data import DEPLOYMENT_CONTRACT_V7, PROMPT_TEMPLATE, PROMPTS_SHA256
    from cosmos_policy.experiments.robot.hanoi import serve_dense

    class FakePolicy:
        identity = {"cosmos_hanoi": {
            "action_horizon": 16, "contract": dict(DEPLOYMENT_CONTRACT_V7), "contract_name": "hanoi_play_k5_cosmos_v1",
            "checkpoint": "run/exports/iter_000032000.pt", "prompt_template": PROMPT_TEMPLATE, "prompts_sha256": PROMPTS_SHA256,
            "export_sha256": "a" * 64, "normalization_sha256": "b" * 64, "embeddings_sha256": "c" * 64, "num_steps": 5,
            "horizon_cap_moves": 5, "initial_weights_sha256": "d" * 64}}

    metadata = serve_dense.build_play_metadata(FakePolicy(), seed=1, gpu="test")
    block = metadata["hanoi_play"]
    assert block["model"] == "cosmos_play" and block["config_name"] == "cosmos_hanoi_play_k5_h16"
    assert block["contract"]["version"] == 7 and block["contract"]["execution_prefix"] == 8 and block["contract"]["action_horizon"] == 16
    assert block["prompt"] is None and len(block["prompts"]) == 81 and block["boards"][0] == "AAAA" and block["horizon_cap_moves"] == 5
    assert block["prompts"]["BAAA"] == "Goal: peg A holds rings 2, 3 and 4, peg B holds ring 1, peg C is empty."
    assert block["training_contract"] == "hanoi_play_k5_cosmos_v1" and metadata["cosmos_hanoi"] is FakePolicy.identity["cosmos_hanoi"]
    json.dumps(metadata)  # the identity must serialise for the client's server_metadata.json

    prompts = tuple(block["prompts"].values())
    request = {"observation/image": np.zeros((224, 224, 3), np.uint8), "observation/state": np.zeros(7, np.float32)}
    assert serve_dense.validate_observation({**request, "prompt": prompts[5]}, prompts)["prompt"] == prompts[5]
    for bad in ({}, {"prompt": "Move all four rings from peg A to peg C following Tower of Hanoi rules."}):
        with pytest.raises(ValueError, match="81 trained prompts"):
            serve_dense.validate_observation({**request, **bad}, prompts)
    reply = serve_dense.make_reply_validator(16, 8)({"actions": np.zeros((16, 4), np.float32), "goal_board": "BAAA", "execution_prefix": 8})
    assert reply["goal_board"] == "BAAA" and reply["actions"].shape == (16, 4) and "task_direction" not in reply
