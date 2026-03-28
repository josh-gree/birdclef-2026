import itertools
import shutil
from pathlib import Path

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
from birdclef_2026.experiments.attn_sed.model import build_attn_sed_model
from birdclef_2026.experiments.attn_sed.trainer import train_attn_sed


class BirdCLEFAttnSED(Experiment):
    name = "birdclef_attn_sed"
    gpu = "A10G"
    ephemeral_disk = 524288

    class Config(BaseModel):
        lr: float = 5e-4
        weight_decay: float = 1e-4
        batch_size: int = 32
        total_steps: int | None = None
        total_val_rounds: int | None = None
        val_every_epochs: float = 1.0
        val_fraction: float = 0.1
        max_samples_per_split: int | None = None
        soundscape_repeat: int = 5
        mixup_repeat: int = 1
        dropout: float = 0.2
        backbone: str = "hgnetv2_b0.ssld_stage2_ft_in1k"
        # Mel config (notebook defaults)
        n_fft: int = 2048
        hop_length: int = 313
        n_mels: int = 256
        # AttnSED loss
        timewise_weight: float = 0.5
        temperature: float = 0.1
        warmup_fraction: float = 0.25
        use_augmentation: bool = False

        @model_validator(mode="after")
        def _exactly_one_stopping_criterion(self):
            if (self.total_steps is None) == (self.total_val_rounds is None):
                raise ValueError("Exactly one of total_steps or total_val_rounds must be set")
            return self

    @staticmethod
    def run(config: "BirdCLEFAttnSED.Config", wandb_run, run_dir: Path) -> None:
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
        transform = nn.Sequential(
            build_spectrogram_pipeline(
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                fmin=20.0,
            ),
            augmentations,
        ).to(device)

        model = build_attn_sed_model(n_classes, dropout=config.dropout, backbone_name=config.backbone).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        val_every = max(1, int(len(train_loader) * config.val_every_epochs))
        total_steps = config.total_steps or (config.total_val_rounds * val_every)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.lr,
            total_steps=total_steps,
            pct_start=config.warmup_fraction,
            anneal_strategy="cos",
            div_factor=25,
            final_div_factor=1e4,
        )

        wandb_run.define_metric("batch_step")
        for m in ["batch_loss", "mean_pos_logit", "mean_neg_logit", "lr"]:
            wandb_run.define_metric(m, step_metric="batch_step")
        wandb_run.define_metric("val_step")
        for prefix in ["val_single_label", "val_soundscape"]:
            wandb_run.define_metric(f"{prefix}_loss", step_metric="val_step")
            wandb_run.define_metric(f"{prefix}_roc_auc", step_metric="val_step")

        print(f"val_every={val_every} steps, total_steps={total_steps}")

        train_attn_sed(
            model=model,
            train_iter=itertools.cycle(train_loader),
            val_loaders=val_loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            transform=transform,
            device=device,
            batch_size=config.batch_size,
            total_steps=total_steps,
            total_val_rounds=config.total_val_rounds,
            val_every=val_every,
            wandb_run=wandb_run,
            run_dir=run_dir,
            timewise_weight=config.timewise_weight,
            temperature=config.temperature,
        )
