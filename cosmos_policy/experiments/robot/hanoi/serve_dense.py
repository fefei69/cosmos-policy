"""Serve a hanoi_dense_v5 export over the OpenPI WebSocket protocol for the dense client.

Same framing as ``serve_waypoint`` (and the OpenPI dense server): metadata on connect, then one
msgpack request per reply. The OpenPI client ``examples/hanoi/deployment/dense_client.py`` drives
the arm; it reads the chunk length and the execution prefix from the ``hanoi_dense`` identity and
selects the export by ``config_name``. Run from the repository root in the Cosmos environment::

    source examples/hanoi/env.sh
    .venv/bin/python -m cosmos_policy.experiments.robot.hanoi.serve_dense --port 8001

Request: ``observation/image`` (224, 224, 3) uint8, ``observation/state`` (7,) joint angles and jaw
stroke, optional ``prompt``. Reply: ``actions`` (H, 4) float32 absolute XYZ and binary jaw intent at
10 Hz, ``reference_rate_hz`` 10, ``execution_prefix`` 8, ``action_horizon`` H, ``server_timing``.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import time

import numpy as np

DEFAULT_RUN = Path("data/hanoi_cosmos/runs/cosmos_policy/hanoi/hanoi_cosmos_dense_20260919_video_init_cycle2")
DEFAULT_STATS = Path("data/hanoi_cosmos/dense_v5/dataset_statistics.json")
DEFAULT_EMBEDDINGS = Path("data/hanoi_cosmos/t5_embeddings.pkl")
CONFIG_NAME = "cosmos_hanoi_dense_v5_h{horizon}"
PROMPT = "Move all four rings from peg A to peg C following Tower of Hanoi rules."


def validate_observation(observation: dict) -> dict:
    if not isinstance(observation, dict):
        raise ValueError("Observation must be a mapping")
    try:
        image = np.asarray(observation["observation/image"])
        state = np.asarray(observation["observation/state"], dtype=np.float32)
    except KeyError as error:
        raise ValueError(f"Missing observation key {error}") from error
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError("observation/image must be the (224, 224, 3) uint8 contract crop")
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("observation/state must be six joint angles (rad) and jaw stroke (m)")
    prompt = observation.get("prompt", PROMPT)
    if isinstance(prompt, bytes):
        prompt = prompt.decode()
    if prompt != PROMPT:
        raise ValueError("This policy was trained for AAAA to CCCC only")
    return {"observation/image": image, "observation/state": state, "prompt": PROMPT}


def make_reply_validator(horizon: int, execution_prefix: int):
    def validate_reply(result: dict) -> dict:
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (horizon, 4) or not np.isfinite(actions).all():
            raise ValueError(f"Policy must return {horizon} finite XYZ/jaw references")
        if not np.isin(actions[:, 3], (0, 1)).all():
            raise ValueError("Jaw intent must be thresholded to 0/1")
        if int(result.get("execution_prefix", execution_prefix)) != execution_prefix:
            raise ValueError("Policy changed the execution prefix")
        return {"actions": actions, "reference_rate_hz": int(result.get("reference_rate_hz", 10)),
                "execution_prefix": execution_prefix, "action_horizon": horizon}
    return validate_reply


def build_metadata(policy, *, seed: int, gpu: str) -> dict:
    identity = policy.identity["cosmos_hanoi"]
    horizon = int(identity["action_horizon"])
    contract = {**identity["contract"], "prompt": PROMPT, "joint_order": "trossen_arm_driver_arm_indices_0_to_5"}
    dense = {
        "model": "cosmos_dense",
        "config_name": CONFIG_NAME.format(horizon=horizon),
        "checkpoint": identity["checkpoint"],
        "contract": contract,
        "prompt": PROMPT,
        "export_sha256": identity["export_sha256"],
        "normalization_sha256": identity["normalization_sha256"],
        "num_steps": identity["num_steps"],
        "seed": seed,
        "sampling": "fixed seed per request",
        "gpu": gpu,
        "initial_weights_sha256": identity.get("initial_weights_sha256"),
        "training_contract": identity.get("contract_name"),
    }
    return {"hanoi_dense": dense, "cosmos_hanoi": identity}


def warm_up(policy, seed: int) -> float:
    observation = {"observation/image": np.zeros((224, 224, 3), np.uint8), "observation/state": np.zeros(7, np.float32),
                   "prompt": PROMPT}
    started = time.monotonic()
    policy.infer(observation, seed=seed)
    return time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN / "exports/iter_000016000.pt")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--denoising-steps", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run = args.checkpoint.resolve().parent.parent  # run/exports/iter_*.pt
    horizon = int(json.loads((run / "joint_contract.json").read_text()).get("horizon", 16))
    # Platform constants bind at import time; set them before any cosmos_policy import.
    os.environ["COSMOS_POLICY_PLATFORM"] = "hanoi_dense"
    os.environ["HANOI_DENSE_HORIZON"] = str(horizon)
    os.environ.setdefault("IMAGINAIRE_OUTPUT_ROOT", str(Path("data/hanoi_cosmos/runs").resolve()))
    import torch

    from cosmos_policy.experiments.robot.hanoi.dense_policy import HanoiDenseInferenceConfig, HanoiDensePolicy
    from cosmos_policy.experiments.robot.hanoi.serve_waypoint import WaypointPolicyServer

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to serve the policy")
    cfg = HanoiDenseInferenceConfig(str(args.checkpoint), str(args.stats), str(args.embeddings),
                                    num_denoising_steps_action=args.denoising_steps, chunk_size=horizon)
    logging.info("Loading %s (%d-step chunks)", args.checkpoint, horizon)
    policy = HanoiDensePolicy(cfg)
    metadata = build_metadata(policy, seed=args.seed, gpu=torch.cuda.get_device_name())
    identity = metadata["hanoi_dense"]
    logging.info("Export SHA-256 %s, config %s", identity["export_sha256"], identity["config_name"])
    logging.info("Warm-up inference took %.2f s", warm_up(policy, args.seed))
    WaypointPolicyServer(
        policy, metadata, host=args.host, port=args.port, seed=args.seed,
        validate_observation=validate_observation,
        validate_reply=make_reply_validator(horizon, int(identity["contract"]["execution_prefix"])),
        name=f"Cosmos dense Hanoi policy ({horizon}-step chunks)",
    ).serve_forever()


if __name__ == "__main__":
    main()
