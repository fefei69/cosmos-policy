"""Serve the Hanoi waypoint_v4 policy over the OpenPI WebSocket protocol.

Run inside the Cosmos environment (``uv sync --extra cu128`` plus
``examples/hanoi/requirements.txt``), from the repository root::

    source examples/hanoi/env.sh && export COSMOS_POLICY_PLATFORM=hanoi_joint
    .venv/bin/python -m cosmos_policy.experiments.robot.hanoi.serve_waypoint --port 8001

The robot-side client is ``examples/hanoi/deployment/cosmos_client.py`` in the OpenPI
checkout; it runs in the ROS/Trossen environment, which cannot import this package.

Protocol (identical framing to OpenPI's ``WebsocketPolicyServer``):

1. On connect the server sends msgpack metadata ``{"cosmos_hanoi": {...}}`` carrying the
   waypoint_v4 deployment contract, the export and statistics SHA-256, and sampling settings.
2. Each request is a msgpack dict with ``observation/image`` (224, 224, 3) uint8,
   ``observation/state`` (7,) joint angles + jaw stroke, ``observation/cartesian_position`` (3,)
   measured XYZ, and an optional ``prompt`` that must equal the trained instruction.
3. Each reply is ``{"actions": (8, 4) float32 absolute XYZ + jaw intent, "commit_count": 1,
   "server_timing": {...}}``. A text reply is a traceback; the connection then closes.
   With ``--dream`` each reply also carries ``future_image`` (224, 224, 3) uint8, the frame the
   model expects after the whole chunk, and ``value`` in [0, 1]; the actions are unchanged.

The server calls ``HanoiWaypointPolicy.infer`` with a fixed seed, the exact adapter the
offline serving-parity check validates, and nothing else.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import http
import json
import logging
import os
import threading
import time
import traceback
from pathlib import Path

import msgpack
import numpy as np

PROMPT = "Move all four rings from peg A to peg C following Tower of Hanoi rules."
DEFAULT_RUN = Path("data/hanoi_cosmos/runs/cosmos_policy/hanoi/hanoi_cosmos_waypoint_v4_20260917")
MAX_MESSAGE_BYTES = 16 * 1024 * 1024

# ---- msgpack with NumPy arrays; adapted from openpi_client.msgpack_numpy (Apache-2.0) ----


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ---- request validation ----


def validate_observation(observation: dict) -> dict:
    """Return a clean observation for ``HanoiWaypointPolicy.infer`` or raise ``ValueError``."""
    if not isinstance(observation, dict):
        raise ValueError("Observation must be a mapping")
    try:
        image = np.asarray(observation["observation/image"])
        state = np.asarray(observation["observation/state"], dtype=np.float32)
        xyz = np.asarray(observation["observation/cartesian_position"], dtype=np.float32)
    except KeyError as error:
        raise ValueError(f"Missing observation key {error}") from error
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError("observation/image must be the (224, 224, 3) uint8 contract crop")
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("observation/state must be six joint angles (rad) and jaw stroke (m)")
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("observation/cartesian_position must be measured XYZ in metres")
    prompt = observation.get("prompt", PROMPT)
    if isinstance(prompt, bytes):
        prompt = prompt.decode()
    if prompt != PROMPT:
        raise ValueError("This policy was trained for AAAA to CCCC only")
    return {
        "observation/image": image,
        "observation/state": state,
        "observation/cartesian_position": xyz,
        "prompt": PROMPT,
    }


def validate_reply(result: dict) -> dict:
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.shape != (8, 4) or not np.isfinite(actions).all():
        raise ValueError("Policy must return eight finite XYZ/jaw destinations")
    if int(result.get("commit_count", 1)) != 1:
        raise ValueError("The waypoint policy commits exactly one destination per observation")
    reply = {"actions": actions, "commit_count": 1}
    if "future_image" in result:
        future = np.asarray(result["future_image"])
        if future.shape != (224, 224, 3) or future.dtype != np.uint8:
            raise ValueError("Predicted future frame must be a 224 x 224 x 3 uint8 image")
        value = float(result.get("value", np.nan))
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Predicted value must lie in [0, 1]")
        reply["future_image"], reply["value"] = future, value
    return reply


# ---- server ----


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


class WaypointPolicyServer:
    """One policy, any number of sequential clients; inference runs on the event-loop thread."""

    def __init__(self, policy, metadata: dict, *, host: str = "127.0.0.1", port: int = 8001, seed: int = 1,
                 dream: bool = False):
        self.policy = policy
        self.dream = dream
        self.metadata = metadata
        self.host, self.port, self.seed = host, port, seed
        self.bound_port = None
        self.ready = threading.Event()
        self._loop = None
        self._server = None

    def serve_forever(self) -> None:
        try:
            asyncio.run(self.run())
        except asyncio.CancelledError:
            pass  # stop() closed the server from another thread.

    async def run(self) -> None:
        from websockets.asyncio.server import serve

        self._loop = asyncio.get_running_loop()
        async with serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=MAX_MESSAGE_BYTES,
            process_request=_health_check,
        ) as server:
            self._server = server
            sockets = getattr(server, "sockets", None) or server.server.sockets
            self.bound_port = sockets[0].getsockname()[1]
            self.ready.set()
            logging.info("Serving Hanoi waypoint policy on ws://%s:%d", self.host, self.bound_port)
            await server.serve_forever()

    def stop(self) -> None:
        if self._loop is not None and self._server is not None:
            self._loop.call_soon_threadsafe(self._server.close)

    async def _handler(self, websocket) -> None:
        import websockets

        peer = websocket.remote_address
        logging.info("Connection from %s opened", peer)
        await websocket.send(packb(self.metadata))
        previous_total = None
        while True:
            try:
                message = await websocket.recv()
            except websockets.ConnectionClosed:
                logging.info("Connection from %s closed", peer)
                return
            started = time.monotonic()
            try:
                observation = validate_observation(unpackb(message))
                infer_started = time.monotonic()
                reply = validate_reply(self.policy.infer(observation, seed=self.seed, **({"dream": True} if self.dream else {})))
                reply["server_timing"] = {"infer_ms": (time.monotonic() - infer_started) * 1000}
                if previous_total is not None:
                    reply["server_timing"]["prev_total_ms"] = previous_total * 1000
                await websocket.send(packb(reply))
                previous_total = time.monotonic() - started
            except Exception:
                logging.exception("Inference request from %s failed", peer)
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                return


# ---- loading the real policy (GPU) ----


def sha256_file(path: Path) -> str:
    from cosmos_policy.datasets.hanoi_joint_data import sha256

    return sha256(path)


def destination_grid(metadata_dir: Path) -> list:
    """Every recorded destination in the training labels (18 points for waypoint_v4), 0.1 mm resolution.

    The client snaps each committed destination onto this grid: the policy picks the right
    point but carries a few millimetres of bias, and the relative XYZ encoding would otherwise
    let that bias accumulate from one waypoint to the next.
    """
    with np.load(metadata_dir / "train.npz", allow_pickle=False) as archive:
        xyz = archive["actions"][:, :, :3].reshape(-1, 3).astype(np.float64)
    points = np.unique(np.round(xyz, 4), axis=0)
    if not 2 <= len(points) <= 64 or not np.isfinite(points).all():
        raise ValueError(f"Unexpected destination grid: {len(points)} points")
    return points.tolist()


def build_metadata(checkpoint: Path, metadata_dir: Path, *, seed: int, denoising_steps: int, gpu: str,
                   dream: bool = False) -> dict:
    """Identity the client verifies before moving: contract, hashes, sampling settings, destination grid."""
    prepared = json.loads((metadata_dir / "metadata.json").read_text())
    run = checkpoint.resolve().parent.parent
    training = json.loads((run / "joint_contract.json").read_text())
    return {
        "cosmos_hanoi": {
            "contract": prepared["deployment"],
            "dataset_contract": prepared["contract"],
            "prompt": PROMPT,
            "checkpoint": str(checkpoint.resolve()),
            "export_sha256": sha256_file(checkpoint),
            "statistics_sha256": sha256_file(metadata_dir / "dataset_statistics.json"),
            "training_identity": {
                key: training[key] for key in ("contract", "statistics_sha256", "raw_sha256", "max_updates")
            },
            "destinations": destination_grid(metadata_dir),
            "seed": seed,
            "num_denoising_steps": denoising_steps,
            "commit_count": 1,
            "dream": dream,
            "gpu": gpu,
        }
    }


def load_policy(checkpoint: Path, metadata_dir: Path, embeddings: Path, *, denoising_steps: int):
    os.environ["COSMOS_POLICY_PLATFORM"] = "hanoi_joint"
    os.environ.setdefault("IMAGINAIRE_OUTPUT_ROOT", str(Path("data/hanoi_cosmos/runs").resolve()))
    import torch

    from cosmos_policy.experiments.robot.hanoi.waypoint_policy import (
        HanoiWaypointInferenceConfig,
        HanoiWaypointPolicy,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to serve the policy")
    cfg = HanoiWaypointInferenceConfig(
        str(checkpoint), str(metadata_dir / "dataset_statistics.json"), str(embeddings)
    )
    cfg.num_denoising_steps_action = denoising_steps
    return HanoiWaypointPolicy(cfg), torch.cuda.get_device_name()


def warm_up(policy, seed: int, *, dream: bool = False) -> float:
    """One discarded inference so the first robot request does not pay CUDA warm-up."""
    observation = {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.zeros(7, np.float32),
        "observation/cartesian_position": np.zeros(3, np.float32),
        "prompt": PROMPT,
    }
    started = time.monotonic()
    validate_reply(policy.infer(observation, seed=seed, **({"dream": True} if dream else {})))
    return time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN / "exports/iter_000008000.pt")
    parser.add_argument("--metadata", type=Path, default=Path("data/hanoi_cosmos/waypoint_v4"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/hanoi_cosmos/t5_embeddings.pkl"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--denoising-steps", type=int, default=5)
    parser.add_argument("--dream", action="store_true",
                        help="return the predicted future frame and value with every reply so the client "
                             "can save them (one extra VAE decode per request; actions unchanged)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    policy, gpu = load_policy(args.checkpoint, args.metadata, args.embeddings, denoising_steps=args.denoising_steps)
    logging.info("Policy loaded on %s; hashing export for the identity handshake", gpu)
    metadata = build_metadata(
        args.checkpoint, args.metadata, seed=args.seed, denoising_steps=args.denoising_steps, gpu=gpu,
        dream=args.dream,
    )
    logging.info("Export SHA-256 %s", metadata["cosmos_hanoi"]["export_sha256"])
    logging.info("Warm-up inference took %.2f s", warm_up(policy, args.seed, dream=args.dream))
    WaypointPolicyServer(policy, metadata, host=args.host, port=args.port, seed=args.seed,
                         dream=args.dream).serve_forever()


if __name__ == "__main__":
    main()
