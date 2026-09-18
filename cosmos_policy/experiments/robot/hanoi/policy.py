"""Hanoi observation/action contract shared by offline evaluation and serving.

This module does not connect to a robot. Inputs are stored RGB224 images and
four measured state values (XYZ and jaw); returned actions are absolute XYZ references and
binary jaw intent (0=close, 1=open) at the recorded 30 Hz rate.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class HanoiInferenceConfig:
    ckpt_path: str
    dataset_stats_path: str
    t5_text_embeddings_path: str
    config: str = "cosmos_predict2_2b_hanoi__inference"
    config_file: str = "cosmos_policy/config/hanoi_config.py"
    suite: str = "hanoi"
    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = False
    num_wrist_images: int = 0
    use_proprio: bool = True
    normalize_proprio: bool = True
    unnormalize_actions: bool = True
    action_dim: int = 4
    chunk_size: int = 63
    use_jpeg_compression: bool = False
    trained_with_image_aug: bool = False
    use_variance_scale: bool = False
    num_denoising_steps_action: int = 5


def validate_config(cfg: HanoiInferenceConfig) -> None:
    required = {
        "suite": "hanoi",
        "use_third_person_image": True,
        "num_third_person_images": 1,
        "use_wrist_image": False,
        "num_wrist_images": 0,
        "use_proprio": True,
        "normalize_proprio": True,
        "unnormalize_actions": True,
        "action_dim": 4,
        "chunk_size": 63,
        "use_jpeg_compression": False,
        "trained_with_image_aug": False,
    }
    for name, expected in required.items():
        if getattr(cfg, name) != expected:
            raise ValueError(f"Hanoi requires {name}={expected!r}")
    if cfg.num_denoising_steps_action < 1:
        raise ValueError("num_denoising_steps_action must be positive")


def validate_dataset_stats(stats: dict) -> dict:
    """Check train-derived bounds before placing any model on a GPU."""
    converted = {}
    for group, dimension in (("actions", 4), ("proprio", 4)):
        for bound in ("min", "max"):
            key = f"{group}_{bound}"
            values = np.asarray(stats[key], dtype=np.float32)
            if values.shape != (dimension,) or not np.isfinite(values).all():
                raise ValueError(f"{key} must contain {dimension} finite values")
            converted[key] = values
        if np.any(converted[f"{group}_max"] <= converted[f"{group}_min"]):
            raise ValueError(f"{group} bounds must have positive span; regenerate training statistics")
    return converted


def make_observation(pixels: np.ndarray, measured_state: np.ndarray) -> dict:
    """Accept stored RGB and measured XYZ/jaw; reject velocity-bearing state."""
    pixels = np.asarray(pixels)
    measured_state = np.asarray(measured_state, dtype=np.float32)
    if pixels.shape != (224, 224, 3) or pixels.dtype != np.uint8:
        raise ValueError("Expected the stored uint8 RGB image with shape (224,224,3); do not crop it again")
    if measured_state.shape != (4,) or not np.isfinite(measured_state).all():
        raise ValueError("Expected four finite measured values: XYZ and jaw stroke; exclude velocity and commanded jaw")
    return {"primary_image": pixels, "proprio": measured_state.copy()}


def to_absolute_actions(relative_actions: np.ndarray, measured_xyz: np.ndarray) -> np.ndarray:
    """Add one observation anchor to every XYZ; jaw intent is never accumulated."""
    actions = np.array(relative_actions, dtype=np.float32, copy=True)
    anchor = np.asarray(measured_xyz, dtype=np.float32)
    if actions.shape != (63, 4) or not np.isfinite(actions).all():
        raise ValueError("Expected a finite (63,4) action chunk in unnormalized dataset units")
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError("Expected finite measured XYZ from the current observation")
    actions[:, :3] += anchor[None, :]
    actions[:, 3] = (actions[:, 3] >= 0.5).astype(np.float32)
    return actions


def load_hanoi_weights(model, state_dict: dict) -> None:
    """Require every network tensor to match before copying checkpoint weights.

    The generic Cosmos non-strict loader logs and skips incompatible tensors.
    Serving must instead fail rather than retain randomly initialized weights.
    Transformer Engine's optional extra-state blobs are version-dependent and
    may be absent; all network parameters and ordinary buffers are required.
    """
    import torch

    def optional_extra_state(key):
        return key == "_extra_state" or key.endswith("._extra_state")

    expected = {key: value for key, value in model.net.state_dict().items() if not optional_extra_state(key)}
    regular = {
        key.removeprefix("net."): value
        for key, value in state_dict.items()
        if key.startswith("net.") and not optional_extra_state(key)
    }
    missing = sorted(expected.keys() - regular.keys())
    unexpected = sorted(regular.keys() - expected.keys())
    mismatched = [
        key for key in expected.keys() & regular.keys()
        if not isinstance(regular[key], torch.Tensor) or regular[key].shape != expected[key].shape
    ]
    if missing or unexpected or mismatched:
        raise ValueError(
            "Hanoi checkpoint does not match the complete policy network: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, mismatched_shapes={mismatched[:8]}"
        )
    result = model.net.load_state_dict(regular, strict=False)
    missing_after_load = [key for key in result.missing_keys if not optional_extra_state(key)]
    unexpected_after_load = [key for key in result.unexpected_keys if not optional_extra_state(key)]
    if missing_after_load or unexpected_after_load:
        raise RuntimeError(f"Incomplete Hanoi weight load: {missing_after_load=}, {unexpected_after_load=}")


def load_hanoi_policy(cfg: HanoiInferenceConfig):
    """Load a local full policy .pt or a training DCP model directory and cached T5.

    Dependency and checkpoint preparation happen before GPU allocation. A missing
    file or direction embedding fails before model initialization; this function
    never computes a T5 embedding or downloads a policy checkpoint.
    """
    import json
    import pickle

    import torch

    from cosmos_policy.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
    from cosmos_policy.datasets.hanoi_data import PROMPTS
    from cosmos_policy.experiments.robot import cosmos_utils

    validate_config(cfg)
    if (ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM) != (4, 63, 4):
        raise ValueError("Set COSMOS_POLICY_PLATFORM=hanoi before importing Cosmos Policy")
    checkpoint = Path(cfg.ckpt_path).expanduser().resolve()
    if not cfg.ckpt_path or not checkpoint.exists():
        raise FileNotFoundError(f"Provide an existing local Hanoi checkpoint: {cfg.ckpt_path!r}")
    if checkpoint.is_dir() and (checkpoint / "model" / ".metadata").is_file():
        checkpoint = checkpoint / "model"
    is_dcp = checkpoint.is_dir() and (checkpoint / ".metadata").is_file()
    if not is_dcp and not (checkpoint.is_file() and checkpoint.suffix == ".pt"):
        raise ValueError("Checkpoint must be a .pt file or a DCP model directory containing .metadata")

    with open(cfg.dataset_stats_path) as handle:
        stats = validate_dataset_stats(json.load(handle))
    with open(cfg.t5_text_embeddings_path, "rb") as handle:
        embeddings = pickle.load(handle)
    # Validate all directions together so an absent reverse prompt fails at startup.
    checked_embeddings = {}
    for prompt in PROMPTS.values():
        if prompt not in embeddings:
            raise KeyError(f"Missing precomputed direction prompt {prompt!r}; prepare the T5 cache first")
        embedding = torch.as_tensor(embeddings[prompt])
        if embedding.shape == (512, 1024):
            embedding = embedding.unsqueeze(0)
        if embedding.shape != (1, 512, 1024) or not torch.isfinite(embedding).all():
            raise ValueError(f"Expected a finite (1,512,1024) T5 embedding for {prompt!r}")
        checked_embeddings[prompt] = embedding.to(dtype=torch.bfloat16, device="cuda")
    cosmos_utils.t5_text_embeddings_cache.update(checked_embeddings)

    model, model_config = cosmos_utils.load_model_from_checkpoint(
        experiment_name=cfg.config,
        s3_checkpoint_dir=str(checkpoint),
        config_file=cfg.config_file,
        instantiate_ema=False,
        load_ema_to_reg=False,
        skip_load_model=True,
    )
    if is_dcp:
        # The repository's generic local loader assumes a consolidated file.
        from torch.distributed.checkpoint import FileSystemReader

        from cosmos_policy._src.predict2.checkpointer.dcp import DefaultLoadPlanner, ModelWrapper, dcp_load_state_dict
        from cosmos_policy.utils.hanoi_checkpoint import validate_model_metadata

        wrapper = ModelWrapper(model)
        state_dict = wrapper.state_dict()
        validate_model_metadata(checkpoint, state_dict)
        dcp_load_state_dict(state_dict, FileSystemReader(str(checkpoint)), DefaultLoadPlanner(allow_partial_load=False))
        wrapper.load_state_dict(state_dict)
        del state_dict, wrapper
    else:
        # mmap avoids a second full host-memory checkpoint copy. Loading the
        # network directly lets us inspect missing keys and shape mismatches.
        state_dict = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(state_dict, dict):
            raise ValueError("Expected a consolidated policy state dictionary")
        load_hanoi_weights(model, state_dict)
        del state_dict
    if (model.config.state_t, model.config.min_num_conditional_frames) != (7, 3):
        raise ValueError("Hanoi requires the seven-latent inference model config with three conditioning frames")
    model.eval()
    model.to("cuda")
    torch.cuda.empty_cache()
    return model, stats, model_config


def predict_hanoi_actions(
    cfg: HanoiInferenceConfig,
    model,
    dataset_stats: dict,
    pixels: np.ndarray,
    measured_state: np.ndarray,
    direction: str,
    *,
    seed: int = 1,
) -> np.ndarray:
    """Predict one chunk without decoding future RGB or retaining GPU intermediates."""
    from cosmos_policy.datasets.hanoi_data import PROMPTS
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    validate_config(cfg)
    if direction not in PROMPTS:
        raise ValueError(f"Unknown Hanoi direction: {direction!r}")
    observation = make_observation(pixels, measured_state)
    prediction = get_action(
        cfg,
        model,
        dataset_stats,
        observation,
        PROMPTS[direction],
        seed=seed,
        randomize_seed=False,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=False,
        batch_size=1,
    )
    return to_absolute_actions(prediction["actions"], observation["proprio"][:3])
