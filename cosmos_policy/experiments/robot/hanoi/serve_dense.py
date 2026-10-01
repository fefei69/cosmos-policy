"""Serve a hanoi_dense_v5, hanoi_multitask_v6 or hanoi_play_k5 export over the OpenPI WebSocket protocol for the dense client.

Same framing as ``serve_waypoint`` (and the OpenPI dense server): metadata on connect, then one
msgpack request per reply. The OpenPI client ``examples/hanoi/deployment/dense_client.py`` drives
the arm; it reads the chunk length and the execution prefix from the ``hanoi_dense`` identity and
selects the export by ``config_name``. Run from the repository root in the Cosmos environment::

    source examples/hanoi/env.sh
    .venv/bin/python -m cosmos_policy.experiments.robot.hanoi.serve_dense --port 8001              # single-task dense
    .venv/bin/python -m cosmos_policy.experiments.robot.hanoi.serve_dense --port 8001 --multitask  # six tasks, prompt required
    .venv/bin/python -m cosmos_policy.experiments.robot.hanoi.serve_dense --port 8001 --play       # play_k5, goal board sentence required

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
MULTITASK_CONFIG_NAME = "cosmos_hanoi_multitask_v6_h{horizon}{cycle}"
MULTITASK_CONTRACT = "hanoi_multitask_v6_cosmos_v1"
DEFAULT_MULTITASK_RUN = Path("data/hanoi_cosmos/runs/cosmos_policy/hanoi/hanoi_cosmos_multitask_20260926_video_init")
DEFAULT_MULTITASK_STATS = Path("data/hanoi_cosmos/multitask_v6/dataset_statistics.json")
DEFAULT_MULTITASK_EMBEDDINGS = Path("data/hanoi_cosmos/t5_embeddings_multitask.pkl")
PLAY_CONFIG_NAME = "cosmos_hanoi_play_k5_h{horizon}"
PLAY_CONTRACT = "hanoi_play_k5_cosmos_v1"
DEFAULT_PLAY_RUN = Path("data/hanoi_cosmos/runs/cosmos_policy/hanoi/hanoi_cosmos_play_20260930_video_init")
DEFAULT_PLAY_STATS = Path("data/hanoi_cosmos/play_k5/dataset_statistics.json")
DEFAULT_PLAY_EMBEDDINGS = Path("data/hanoi_cosmos/t5_embeddings_play.pkl")
PROMPT = "Move all four rings from peg A to peg C following Tower of Hanoi rules."


def validate_observation(observation: dict, prompts=(PROMPT,)) -> dict:
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
    prompt = observation.get("prompt", prompts[0] if len(prompts) == 1 else None)
    if isinstance(prompt, bytes):
        prompt = prompt.decode()
    if prompt not in prompts:
        raise ValueError("This policy was trained for AAAA to CCCC only" if len(prompts) == 1 else
                         f"The request must carry one of the {len(prompts)} trained prompts verbatim; there is no default")
    return {"observation/image": image, "observation/state": state, "prompt": prompt}


def make_reply_validator(horizon: int, execution_prefix: int):
    def validate_reply(result: dict) -> dict:
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (horizon, 4) or not np.isfinite(actions).all():
            raise ValueError(f"Policy must return {horizon} finite XYZ/jaw references")
        if not np.isin(actions[:, 3], (0, 1)).all():
            raise ValueError("Jaw intent must be thresholded to 0/1")
        if int(result.get("execution_prefix", execution_prefix)) != execution_prefix:
            raise ValueError("Policy changed the execution prefix")
        reply = {"actions": actions, "reference_rate_hz": int(result.get("reference_rate_hz", 10)),
                 "execution_prefix": execution_prefix, "action_horizon": horizon}
        if "task" in result:  # six-task policy: echo the task the prompt resolved to, in the client's field names
            reply["task_direction"] = str(result["task"])
            reply["task"] = int(result.get("task_index", -1))
            reply["goal_peg"] = str(result.get("goal_peg", ""))
        if "goal_board" in result:  # play policy: echo the goal board the sentence resolved to
            reply["goal_board"] = str(result["goal_board"])
        return reply
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


def build_multitask_metadata(policy, *, seed: int, gpu: str, cycle: str) -> dict:
    """The hanoi_multitask identity the OpenPI dense client checks: contract six with the task table and prompts."""
    from cosmos_policy.datasets.hanoi_multitask_data import TASKS

    identity = policy.identity["cosmos_hanoi"]
    horizon = int(identity["action_horizon"])
    contract = {**identity["contract"], "version": 6, "action_horizon": horizon, "joint_order": "trossen_arm_driver_arm_indices_0_to_5",
                "conditioning": "instruction string only; one of six verbatim prompts is required on every request, no default task",
                "tasks": [{"index": t.index, "direction": t.direction, "start_peg": t.start, "goal_peg": t.goal, "prompt": t.prompt} for t in TASKS]}
    block = {
        "model": "cosmos_multitask",
        "config_name": MULTITASK_CONFIG_NAME.format(horizon=horizon, cycle=cycle),
        "checkpoint": identity["checkpoint"],
        "contract": contract,
        "prompt": None,
        "prompts": list(identity["prompts"]),
        "export_sha256": identity["export_sha256"],
        "normalization_sha256": identity["normalization_sha256"],
        "embeddings_sha256": identity.get("embeddings_sha256"),
        "num_steps": identity["num_steps"],
        "seed": seed,
        "sampling": "fixed seed per request",
        "gpu": gpu,
        "initial_weights_sha256": identity.get("initial_weights_sha256"),
        "training_contract": identity.get("contract_name"),
    }
    return {"hanoi_multitask": block, "cosmos_hanoi": identity}


def build_play_metadata(policy, *, seed: int, gpu: str) -> dict:
    """The hanoi_play identity the OpenPI dense client checks: contract seven with the 81 boards and their sentences."""
    from cosmos_policy.datasets.hanoi_play_data import BOARDS, PROMPTS

    identity = policy.identity["cosmos_hanoi"]
    horizon = int(identity["action_horizon"])
    contract = {**identity["contract"], "action_horizon": horizon, "joint_order": "trossen_arm_driver_arm_indices_0_to_5",
                "conditioning": "goal board sentence only; one of 81 verbatim sentences is required on every request, no default goal"}
    block = {
        "model": "cosmos_play",
        "config_name": PLAY_CONFIG_NAME.format(horizon=horizon),
        "checkpoint": identity["checkpoint"],
        "contract": contract,
        "prompt": None,
        "boards": list(BOARDS),
        "prompts": dict(zip(BOARDS, PROMPTS)),  # board (peg per ring, ring 1 first) -> its sentence
        "prompt_template": identity["prompt_template"],
        "prompts_sha256": identity["prompts_sha256"],
        "horizon_cap_moves": identity.get("horizon_cap_moves"),
        "export_sha256": identity["export_sha256"],
        "normalization_sha256": identity["normalization_sha256"],
        "embeddings_sha256": identity.get("embeddings_sha256"),
        "num_steps": identity["num_steps"],
        "seed": seed,
        "sampling": "fixed seed per request",
        "gpu": gpu,
        "initial_weights_sha256": identity.get("initial_weights_sha256"),
        "training_contract": identity.get("contract_name"),
    }
    return {"hanoi_play": block, "cosmos_hanoi": identity}


def warm_up(policy, seed: int, prompt: str = PROMPT) -> float:
    observation = {"observation/image": np.zeros((224, 224, 3), np.uint8), "observation/state": np.zeros(7, np.float32),
                   "prompt": prompt}
    started = time.monotonic()
    policy.infer(observation, seed=seed)
    return time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--multitask", action="store_true",
                        help="serve the six-task contract-six policy (defaults below switch to its export, stats and embeddings)")
    parser.add_argument("--play", action="store_true",
                        help="serve the play-trained goal-conditioned policy (contract seven; the goal board sentence is required)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="default: the selected dense, six-task or play export")
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--denoising-steps", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.multitask and args.play:
        raise ValueError("Choose --multitask or --play, not both")
    if args.checkpoint is None:
        args.checkpoint = ((DEFAULT_PLAY_RUN / "exports/iter_000032000.pt") if args.play else
                           (DEFAULT_MULTITASK_RUN / "exports/iter_000032000.pt") if args.multitask else (DEFAULT_RUN / "exports/iter_000016000.pt"))
    run = args.checkpoint.resolve().parent.parent  # run/exports/iter_*.pt
    run_identity = json.loads((run / "joint_contract.json").read_text())
    play = args.play or run_identity.get("contract") == PLAY_CONTRACT
    multitask = not play and (args.multitask or run_identity.get("contract") == MULTITASK_CONTRACT)
    args.stats = args.stats or (DEFAULT_PLAY_STATS if play else DEFAULT_MULTITASK_STATS if multitask else DEFAULT_STATS)
    args.embeddings = args.embeddings or (DEFAULT_PLAY_EMBEDDINGS if play else DEFAULT_MULTITASK_EMBEDDINGS if multitask else DEFAULT_EMBEDDINGS)
    horizon = int(run_identity.get("horizon", 16))
    # Platform constants bind at import time; set them before any cosmos_policy import.
    os.environ["COSMOS_POLICY_PLATFORM"] = "hanoi_dense"
    os.environ["HANOI_DENSE_HORIZON"] = str(horizon)
    os.environ.setdefault("IMAGINAIRE_OUTPUT_ROOT", str(Path("data/hanoi_cosmos/runs").resolve()))
    import torch

    from cosmos_policy.experiments.robot.hanoi.dense_policy import HanoiDenseInferenceConfig, HanoiDensePolicy
    from cosmos_policy.experiments.robot.hanoi.serve_waypoint import WaypointPolicyServer

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to serve the policy")
    if play:
        from cosmos_policy.experiments.robot.hanoi.play_policy import HanoiPlayInferenceConfig, HanoiPlayPolicy

        cfg = HanoiPlayInferenceConfig(str(args.checkpoint), str(args.stats), str(args.embeddings),
                                       num_denoising_steps_action=args.denoising_steps, chunk_size=horizon)
        logging.info("Loading play %s (%d-step chunks)", args.checkpoint, horizon)
        policy = HanoiPlayPolicy(cfg)
        metadata = build_play_metadata(policy, seed=args.seed, gpu=torch.cuda.get_device_name())
        identity = metadata["hanoi_play"]
        prompts = tuple(identity["prompts"].values())
    elif multitask:
        from cosmos_policy.experiments.robot.hanoi.multitask_policy import HanoiMultitaskInferenceConfig, HanoiMultitaskPolicy

        cfg = HanoiMultitaskInferenceConfig(str(args.checkpoint), str(args.stats), str(args.embeddings),
                                            num_denoising_steps_action=args.denoising_steps, chunk_size=horizon)
        logging.info("Loading six-task %s (%d-step chunks)", args.checkpoint, horizon)
        policy = HanoiMultitaskPolicy(cfg)
        cycle = "_cycle2" if "cycle2" in run.name else ""
        metadata = build_multitask_metadata(policy, seed=args.seed, gpu=torch.cuda.get_device_name(), cycle=cycle)
        identity = metadata["hanoi_multitask"]
        prompts = tuple(identity["prompts"])
    else:
        cfg = HanoiDenseInferenceConfig(str(args.checkpoint), str(args.stats), str(args.embeddings),
                                        num_denoising_steps_action=args.denoising_steps, chunk_size=horizon)
        logging.info("Loading %s (%d-step chunks)", args.checkpoint, horizon)
        policy = HanoiDensePolicy(cfg)
        metadata = build_metadata(policy, seed=args.seed, gpu=torch.cuda.get_device_name())
        identity = metadata["hanoi_dense"]
        prompts = (PROMPT,)
    logging.info("Export SHA-256 %s, config %s", identity["export_sha256"], identity["config_name"])
    logging.info("Warm-up inference took %.2f s", warm_up(policy, args.seed, prompts[0]))
    if len(prompts) > 1:
        logging.info("%s server: every request must carry one of %d verbatim prompts", "Play" if play else "Six-task", len(prompts))
    WaypointPolicyServer(
        policy, metadata, host=args.host, port=args.port, seed=args.seed,
        validate_observation=lambda observation: validate_observation(observation, prompts),
        validate_reply=make_reply_validator(horizon, int(identity["contract"]["execution_prefix"])),
        name=f"Cosmos {'play' if play else 'six-task' if multitask else 'dense'} Hanoi policy ({horizon}-step chunks)",
    ).serve_forever()


if __name__ == "__main__":
    main()
