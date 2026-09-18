"""Lazy, episode-bounded Hanoi demonstrations for the stock Cosmos trainer."""

from __future__ import annotations

import json
import os
import pickle
from itertools import zip_longest
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_policy.datasets.hanoi_data import (
    ACTION_HORIZON,
    DIRECTIONS,
    FORMAT_VERSION,
    PROPRIO_COLUMNS,
    PROPRIO_SCHEMA,
    PROMPTS,
    normalize,
    source_path,
    validate_source_scope,
)
from cosmos_policy.datasets.resumable_sampler import file_sha256


class HanoiDataset(Dataset):
    """One eligible observation and a 63-reference target per sample.

    Seven latent slots follow the stock policy layout without wrist cameras:
    blank, measured proprio, RGB, actions, future proprio, future RGB, value.
    HDF5 handles are opened independently in each DataLoader worker.
    """

    resume_data_order = True

    def __init__(
        self,
        data_dir: str,
        t5_text_embeddings_path: str,
        metadata_dir: str = "data/hanoi_cosmos/aaaa_to_cccc_pos_only",
        split: str = "train",
        chunk_size: int = ACTION_HORIZON,
        final_image_size: int = 224,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        gamma: float = 0.9995,
        num_duplicates_per_image: int = 4,
        use_image_aug: bool = False,
        representative_order: bool = False,
        expected_direction: str | None = None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        if chunk_size != 63 or final_image_size != 224 or num_duplicates_per_image != 4:
            raise ValueError("Hanoi requires horizon 63, stored RGB 224x224, and four raw frames per latent")
        if use_image_aug:
            raise ValueError("Hanoi baseline preserves stored crops; image augmentation is disabled")
        if representative_order and split != "val":
            raise ValueError("Representative ordering is only used for the validation split")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        self.data_dir = str(Path(data_dir).resolve())
        self.metadata_dir = str(Path(metadata_dir).resolve())
        self.split, self.chunk_size = split, chunk_size
        self.final_image_size = final_image_size
        self.normalize_actions, self.normalize_proprio = normalize_actions, normalize_proprio
        self.gamma = gamma
        metadata = json.loads((Path(metadata_dir) / "metadata.json").read_text())
        if metadata["format_version"] != FORMAT_VERSION or metadata["chunk_size"] != chunk_size:
            raise ValueError("Prepared Hanoi metadata has an incompatible format or horizon")
        if metadata.get("proprio_columns") != list(PROPRIO_COLUMNS) or metadata.get("proprio_schema") != PROPRIO_SCHEMA:
            raise ValueError("Hanoi requires position/jaw metadata with velocity excluded")
        for source in metadata["sources"]:
            path = source_path(self.data_dir, source["direction"])
            identity = path.stat()
            if identity.st_size != source["size_bytes"] or identity.st_mtime_ns != source["mtime_ns"]:
                raise ValueError(f"Source changed after indexing: {path}; prepare metadata again")
        with np.load(Path(metadata_dir) / f"{split}_indices.npz", allow_pickle=False) as indices:
            self.source_indices = indices["source_index"].copy()
            self.episode_indices = indices["episode_index"].copy()
            self.row_indices = indices["row_index"].copy()
        if len(self.row_indices) != metadata["eligible_counts"][split]:
            raise ValueError("Prepared Hanoi split count does not match metadata")
        selected_sources = validate_source_scope(metadata, self.source_indices, expected_direction)
        self.prompts = {direction: PROMPTS[direction] for direction in selected_sources.values()}
        if representative_order:
            self._order_validation_anchors()
        self.episodes = {(e["source_index"], e["episode_index"]): e for e in metadata["episodes"]}
        self.dataset_stats = json.loads((Path(metadata_dir) / "dataset_statistics.json").read_text())
        self.resume_data_order_identity = {
            "dataset": "cosmos_hanoi_v2",
            "proprio_columns": list(PROPRIO_COLUMNS),
            "proprio_schema": PROPRIO_SCHEMA,
            "format_version": metadata["format_version"],
            "split": split,
            "sources": metadata["sources"],
            "indices_sha256": file_sha256(Path(metadata_dir) / f"{split}_indices.npz"),
            "representative_order": representative_order,
            "chunk_size": chunk_size,
            "normalize_actions": normalize_actions,
            "normalize_proprio": normalize_proprio,
            "statistics": self.dataset_stats,
            "prompts": self.prompts,
            "gamma": gamma,
        }
        with open(t5_text_embeddings_path, "rb") as stream:
            text_embeddings = pickle.load(stream)
        self.t5_text_embeddings = {}
        for prompt in self.prompts.values():
            if prompt not in text_embeddings:
                raise ValueError(f"Missing precomputed T5 embedding for {prompt!r}")
            embedding = torch.as_tensor(text_embeddings[prompt]).detach().cpu()
            if embedding.shape == (1, 512, 1024):
                embedding = embedding.squeeze(0)
            if embedding.shape != (512, 1024):
                raise ValueError(f"Expected T5 embedding (512, 1024), got {tuple(embedding.shape)}")
            self.t5_text_embeddings[prompt] = embedding.to(torch.bfloat16)
        self.unique_commands = set(self.prompts.values())
        self._handles = {}
        self._pid = None

    def _order_validation_anchors(self) -> None:
        """Make even short validation prefixes cover episodes and trajectory time.

        Visit each held-out episode/direction group in round-robin
        order. Within each group, rotate between shuffled early/middle/late thirds.
        This only reorders the existing split; it neither removes nor adds rows.
        """
        rng = np.random.default_rng(195)
        groups = sorted(set(zip(self.source_indices.tolist(), self.episode_indices.tolist(), strict=True)))
        rng.shuffle(groups)
        grouped_orders = []
        for source, episode in groups:
            positions = np.flatnonzero((self.source_indices == source) & (self.episode_indices == episode))
            positions = positions[np.argsort(self.row_indices[positions])]
            thirds = [rng.permutation(part).tolist() for part in np.array_split(positions, 3)]
            grouped_orders.append([i for chunk in zip_longest(*thirds) for i in chunk if i is not None])
        order = np.asarray(
            [i for chunk in zip_longest(*grouped_orders) for i in chunk if i is not None], dtype=np.int64
        )
        self.source_indices = self.source_indices[order]
        self.episode_indices = self.episode_indices[order]
        self.row_indices = self.row_indices[order]

    def __len__(self) -> int:
        return len(self.row_indices)

    def _file(self, source_index: int) -> h5py.File:
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if source_index not in self._handles:
            self._handles[source_index] = h5py.File(source_path(self.data_dir, DIRECTIONS[source_index]), "r")
        return self._handles[source_index]

    def close(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            handle.close()
        self._handles = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handles"], state["_pid"] = {}, None
        return state

    def __del__(self):
        self.close()

    def _actions(self, handle: h5py.File, row: int, end: int, measured_xyz: np.ndarray) -> np.ndarray:
        targets = handle["action_abs"][row : min(row + self.chunk_size, end), :].astype(np.float32)
        if len(targets) < self.chunk_size:
            targets = np.concatenate((targets, np.repeat(targets[-1:], self.chunk_size - len(targets), axis=0)))
        targets[:, :3] -= measured_xyz
        return normalize(targets, self.dataset_stats, "actions") if self.normalize_actions else targets

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_index = int(self.source_indices[index])
        episode = self.episodes[(source_index, int(self.episode_indices[index]))]
        row, end = int(self.row_indices[index]), episode["start"] + episode["length"]
        if episode["split"] != self.split or not episode["start"] <= row < end:
            raise ValueError("Anchor does not belong to its recorded episode/split")
        future_row = min(row + self.chunk_size, end - 1)
        handle = self._file(source_index)
        # Exclude velocity (3:6) and commanded jaw (7) from both current inputs
        # and the auxiliary future-state target.
        measured = handle["proprio"][row, list(PROPRIO_COLUMNS)].astype(np.float32)
        future_measured = handle["proprio"][future_row, list(PROPRIO_COLUMNS)].astype(np.float32)
        actions = self._actions(handle, row, end, measured[:3])
        next_actions = self._actions(handle, future_row, end, future_measured[:3])
        current_rgb = handle["pixels"][row]
        future_rgb = handle["pixels"][future_row]
        blank = np.zeros_like(current_rgb)
        frames = [blank[None]]
        for frame in (blank, current_rgb, blank, blank, future_rgb, blank):
            frames.append(np.repeat(frame[None], 4, axis=0))
        video = torch.from_numpy(np.concatenate(frames).transpose(3, 0, 1, 2).copy())
        proprio = normalize(measured, self.dataset_stats, "proprio") if self.normalize_proprio else measured
        future_proprio = (
            normalize(future_measured, self.dataset_stats, "proprio") if self.normalize_proprio else future_measured
        )
        # Stock demonstration return: terminal success reward, rescaled to [-1, 1].
        value = np.float32(2 * self.gamma ** (end - 1 - future_row) - 1)
        next_future_row = min(future_row + self.chunk_size, end - 1)
        next_value = np.float32(2 * self.gamma ** (end - 1 - next_future_row) - 1)
        return {
            "video": video,
            "actions": actions,
            "proprio": proprio,
            "future_proprio": future_proprio,
            "t5_text_embeddings": self.t5_text_embeddings[self.prompts[DIRECTIONS[source_index]]],
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            # This is the pretrained synthetic video-slot conditioning convention;
            # it does not change the 30 Hz robot reference timeline.
            "fps": 16,
            "padding_mask": torch.zeros(1, 224, 224),
            "image_size": torch.full((4,), 224.0),
            "__key__": index,
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "current_proprio_latent_idx": 1,
            "current_image_latent_idx": 2,
            "action_latent_idx": 3,
            "future_proprio_latent_idx": 4,
            "future_image_latent_idx": 5,
            "value_latent_idx": 6,
            "current_wrist_image_latent_idx": -1,
            "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1,
            "future_wrist_image_latent_idx": -1,
            "future_wrist_image2_latent_idx": -1,
            "future_image2_latent_idx": -1,
            "value_function_return": value,
            "next_action_chunk": next_actions,
            "next_value_function_return": next_value,
        }
