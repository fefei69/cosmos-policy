"""Local Hanoi fine-tuning config; importing it never downloads checkpoints.

This preserves the released 2B policy network and joint action/future-state/value
objective. The sequence has seven latent slots because Hanoi has no wrist camera.
"""

import copy
import os
from pathlib import Path
from typing import Any

import attrs
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from cosmos_policy._src.imaginaire.config import Config as BaseConfig
from cosmos_policy._src.imaginaire.configs.lr_scheduler import LambdaLinearSchedulerConfig
from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.predict2.checkpointer.dcp import DistributedCheckpointer
from cosmos_policy._src.predict2.configs.common.defaults.callbacks import BASIC_CALLBACKS
from cosmos_policy._src.predict2.configs.common.defaults.optimizer import FusedAdamWConfig
from cosmos_policy._src.predict2.configs.video2world.defaults.net import COSMOS_V1_2B_NET_MININET
from cosmos_policy._src.predict2.networks.minimal_v4_dit import SACConfig
from cosmos_policy.config.conditioner.video2world_conditioner import VideoPredictionConditioner
from cosmos_policy.datasets.hanoi_dataset import HanoiDataset
from cosmos_policy.models.hanoi_model import HanoiPolicyModel, HanoiPolicyVideo2WorldConfig
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE
from cosmos_policy.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from cosmos_policy.trainer import CosmosPolicyTrainer
from cosmos_policy.utils.hanoi_optimizer import get_hanoi_optimizer
from cosmos_policy.utils.hanoi_schedule import aloha_training_schedule, continuation_schedule
from cosmos_policy.utils.hanoi_training import HanoiTrainingMonitor


@attrs.define(slots=False)
class Config(BaseConfig):
    defaults: list[Any] = attrs.field(factory=lambda: ["_self_", {"experiment": None}])


def make_config():
    root = Path(__file__).resolve().parents[2]
    metadata = os.environ.get("HANOI_METADATA_DIR", str(root / "data/hanoi_cosmos/aaaa_to_cccc_pos_only"))
    embeddings = os.environ.get("HANOI_T5_EMBEDDINGS", str(root / "data/hanoi_cosmos/t5_embeddings.pkl"))
    net = copy.deepcopy(COSMOS_V1_2B_NET_MININET)
    net.rope_enable_fps_modulation = False
    # Same positional settings as the released LIBERO/Predict2-2B checkpoint.
    net.rope_h_extrapolation_ratio = 3.0
    net.rope_w_extrapolation_ratio = 3.0
    net.rope_t_extrapolation_ratio = 1.0
    net.sac_config = SACConfig(mode="block_wise")
    conditioner = copy.deepcopy(VideoPredictionConditioner)
    conditioner.text.dropout_rate = 0.0
    conditioner.use_video_condition.dropout_rate = 0.0
    initial_checkpoint = os.environ.get(
        "HANOI_INIT_CHECKPOINT", str(root / "checkpoints/public/Cosmos-Policy-LIBERO-Predict2-2B.pt")
    )
    model_config = HanoiPolicyVideo2WorldConfig(
        initial_checkpoint=initial_checkpoint,
        net=net,
        conditioner=conditioner,
        tokenizer=L(Wan2pt1VAEInterface)(
            vae_pth=os.environ.get("HANOI_VAE_PATH", str(root / "checkpoints/public/Wan2.1_VAE.pth")),
            chunk_duration=25,
            temporal_window=16,
            load_mean_std=False,
            is_parallel=False,
        ),
        sde=L(HybridEDMSDE)(
            hybrid_sigma_distribution=True,
            p_mean=1.3862943611198906,
            p_std=1.2,
            sigma_max=200,
            sigma_min=0.01,
            uniform_lower=1.0,
            uniform_upper=85.0,
        ),
        state_t=7,
        resolution="224",
        resize_online=True,
        scaling="rectified_flow",
        sigma_data=1.0,
        precision="bfloat16",
        fsdp_shard_size=1,
        min_num_conditional_frames=3,
        max_num_conditional_frames=3,
        sigma_conditional=0.0,
        conditioning_strategy="frame_replace",
        denoise_replace_gt_frames=True,
        high_sigma_strategy="none",
    )
    model_config.ema = copy.deepcopy(model_config.ema)
    model_config.ema.enabled = False
    optimizer = copy.deepcopy(FusedAdamWConfig)
    optimizer._target_ = get_hanoi_optimizer
    optimizer.lr = 1e-5
    scheduler = copy.deepcopy(LambdaLinearSchedulerConfig)
    scheduler.warm_up_steps = [100]
    scheduler.cycle_lengths = [10000]
    scheduler.f_max = [1.0]
    scheduler.f_min = [0.1]
    if os.environ.get("HANOI_TRAINING_SCHEDULE") == "aloha":
        for key, value in aloha_training_schedule().items():
            scheduler[key] = value
    elif os.environ.get("HANOI_CONTINUATION_ANCHOR"):
        for key, value in continuation_schedule(int(os.environ["HANOI_CONTINUATION_ANCHOR"])).items():
            scheduler[key] = value

    def loader(split):
        return L(DataLoader)(
            dataset=L(HanoiDataset)(
                data_dir=os.environ.get("HANOI_DATA_ROOT", "/scratch/cw5167/datasets"),
                metadata_dir=metadata,
                t5_text_embeddings_path=embeddings,
                split=split,
                chunk_size=63,
                gamma=0.9995,
                use_image_aug=False,
                representative_order=split == "val",
                expected_direction=os.environ.get("HANOI_DIRECTION", "aaaa_to_cccc"),
            ),
            batch_size=2,
            drop_last=split == "train",
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            pin_memory_device="",
            timeout=120,
        )

    c = Config(
        model=L(HanoiPolicyModel)(config=model_config, _recursive_=False),
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader_train=loader("train"),
        dataloader_val=loader("val"),
    )
    c.job.project = "cosmos_policy"
    c.job.group = "hanoi"
    c.job.name = os.environ.get("HANOI_RUN_NAME", "hanoi_cosmos_pilot")
    c.job.wandb_mode = "disabled"
    c.trainer.type = CosmosPolicyTrainer
    # The model handles parallelism; shard size one is a normal single GPU.
    c.trainer.distributed_parallelism = "fsdp"
    c.trainer.max_iter = 10000
    c.trainer.grad_accum_iter = 8
    c.trainer.logging_iter = 10
    c.trainer.seed = 195
    c.trainer.run_validation = True
    c.trainer.run_validation_on_start = True
    c.trainer.validation_iter = 100
    c.trainer.max_val_iter = 16
    c.trainer.timeout_period = 600
    c.trainer.callbacks = copy.deepcopy(BASIC_CALLBACKS)
    c.trainer.callbacks["compile_tokenizer"].enabled = False
    c.trainer.callbacks["hanoi_metrics"] = L(HanoiTrainingMonitor)()
    c.model_parallel.context_parallel_size = 1
    c.checkpoint.type = L(DistributedCheckpointer)()
    c.checkpoint.dcp_async_mode_enabled = False
    c.checkpoint.load_path = initial_checkpoint
    c.checkpoint.load_training_state = False
    c.checkpoint.load_ema_to_reg = False
    c.checkpoint.strict_resume = True
    c.checkpoint.save_iter = 100
    c.checkpoint.save_to_object_store.enabled = False
    c.checkpoint.load_from_object_store.enabled = False
    c.upload_reproducible_setup = False

    cs = ConfigStore.instance()
    cs.store(group="experiment", package="_global_", name="cosmos_predict2_2b_hanoi", node=LazyDict({}))
    cs.store(
        group="experiment",
        package="_global_",
        name="cosmos_predict2_2b_hanoi__inference",
        node=LazyDict({"model": {"config": {"initial_checkpoint": "", "sde": {"sigma_max": 80, "sigma_min": 4}}}}),
    )
    return c
