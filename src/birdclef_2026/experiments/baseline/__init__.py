import itertools
import shutil
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, model_validator
from wm import Experiment

from birdclef_2026.data.loaders import build_combined_dataloaders
from birdclef_2026.data.transforms import (
    FrequencyMask,
    GaussianNoise,
    TimeMask,
    build_spectrogram_pipeline,
)
from birdclef_2026.experiments.baseline.model import build_model, build_vit_model
from birdclef_2026.experiments.baseline.trainer import train_n_steps


class BirdCLEFBaseline(Experiment):
    name = "birdclef_baseline"
    gpu = "A10G"
    ephemeral_disk = (
        524288  # 512 GB — minimum allowed, needed to copy 74 GB memmap to local disk
    )

    class Config(BaseModel):
        lr: float = 1e-3
        batch_size: int = 32
        total_steps: int | None = None
        total_val_rounds: int | None = None
        val_every_epochs: float = 1.0
        val_fraction: float = 0.1
        max_samples_per_split: int | None = None  # set small (e.g. 64) for a smoke run
        soundscape_repeat: int = 5
        mixup_repeat: int = 1
        dropout: float = 0.0
        hidden: int = 512
        weight_decay: float = 0.0
        unfreeze_blocks: int = 0
        backbone: Literal["efficientnet_b3", "vit_base"] = "efficientnet_b3"
        backbone_lr: float = 1e-5
        resume_from: str | None = None
        use_augmentation: bool = False

        @model_validator(mode="after")
        def _exactly_one_stopping_criterion(self):
            if (self.total_steps is None) == (self.total_val_rounds is None):
                raise ValueError("Exactly one of total_steps or total_val_rounds must be set")
            return self

    @staticmethod
    def run(config: "BirdCLEFBaseline.Config", wandb_run, run_dir: Path) -> None:
        print("Copying data to local disk...")
        shutil.copy("/data/audio.npy", "/tmp/audio.npy")
        shutil.copy("/data/index.parquet", "/tmp/index.parquet")
        shutil.copy("/data/soundscape_audio.npy", "/tmp/soundscape_audio.npy")
        shutil.copy("/data/soundscape_index.parquet", "/tmp/soundscape_index.parquet")
        print("Done.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader, val_loaders, label2idx = build_combined_dataloaders(
            sl_audio_path="/tmp/audio.npy",
            sl_index_path="/tmp/index.parquet",
            ss_audio_path="/tmp/soundscape_audio.npy",
            ss_index_path="/tmp/soundscape_index.parquet",
            taxonomy_path="/data/taxonomy.csv",
            batch_size=config.batch_size,
            val_fraction=config.val_fraction,
            soundscape_repeat=config.soundscape_repeat,
            mixup_repeat=config.mixup_repeat,
            max_samples_per_split=config.max_samples_per_split,
        )
        n_classes = len(label2idx)

        augmentations = (
            nn.Sequential(
                FrequencyMask(max_mask_size=20, num_masks=2, p=0.5),
                TimeMask(max_mask_size=40, num_masks=2, p=0.5),
                GaussianNoise(min_std=0.0, max_std=0.05, p=0.5),
            )
            if config.use_augmentation
            else nn.Identity()
        )
        transform = nn.Sequential(build_spectrogram_pipeline(), augmentations).to(device)

        if config.backbone == "efficientnet_b3":
            model = build_model(n_classes, hidden=config.hidden, dropout=config.dropout, unfreeze_blocks=config.unfreeze_blocks).to(device)
        elif config.backbone == "vit_base":
            model = build_vit_model(n_classes, hidden=config.hidden, dropout=config.dropout, unfreeze_blocks=config.unfreeze_blocks).to(device)
        else:
            raise ValueError(f"Unknown backbone: {config.backbone}")

        param_groups = [{"params": model[1].parameters(), "lr": config.lr}]
        if config.unfreeze_blocks > 0:
            backbone_params = [p for p in model[0].parameters() if p.requires_grad]
            param_groups.append({"params": backbone_params, "lr": config.backbone_lr})
        optimizer = torch.optim.Adam(param_groups, weight_decay=config.weight_decay)

        if config.resume_from:
            print(f"Resuming from {config.resume_from}")
            ckpt = torch.load(config.resume_from, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        criterion = nn.BCEWithLogitsLoss()

        wandb_run.define_metric("batch_step")
        wandb_run.define_metric("batch_loss", step_metric="batch_step")
        wandb_run.define_metric("mean_pos_logit", step_metric="batch_step")
        wandb_run.define_metric("mean_neg_logit", step_metric="batch_step")
        wandb_run.define_metric("val_step")
        for prefix in ["val_single_label", "val_soundscape"]:
            wandb_run.define_metric(f"{prefix}_loss", step_metric="val_step")
            wandb_run.define_metric(f"{prefix}_roc_auc", step_metric="val_step")

        val_every = max(1, int(len(train_loader) * config.val_every_epochs))
        print(f"val_every={val_every} steps ({config.val_every_epochs} epoch equivalent)")

        train_n_steps(
            model,
            itertools.cycle(train_loader),
            val_loaders,
            optimizer,
            criterion,
            transform,
            device,
            config.batch_size,
            config.total_steps,
            config.total_val_rounds,
            val_every,
            wandb_run,
            run_dir,
        )
