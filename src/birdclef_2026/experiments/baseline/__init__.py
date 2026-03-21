import itertools
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from pydantic import BaseModel, model_validator
from wm import Experiment

from birdclef_2026.data.loaders import build_dataloaders
from birdclef_2026.data.transforms import build_spectrogram_pipeline
from birdclef_2026.experiments.baseline.model import (
    build_frozen_efficientnet_b3_backbone,
    build_head,
)
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
        dropout: float = 0.0
        weight_decay: float = 0.0
        balance_train: bool = False
        resume_from: str | None = None

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
        audio_path, index_path = "/tmp/audio.npy", "/tmp/index.parquet"
        print("Done.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader, val_loader, label2idx = build_dataloaders(
            audio_path,
            index_path,
            batch_size=config.batch_size,
            val_fraction=config.val_fraction,
            max_samples_per_split=config.max_samples_per_split,
            balance_train=config.balance_train,
        )
        n_classes = len(label2idx)

        transform = build_spectrogram_pipeline().to(device)

        backbone = build_frozen_efficientnet_b3_backbone()
        head = build_head(backbone.num_features, n_classes, config.dropout)
        model = nn.Sequential(backbone, head).to(device)

        optimizer = torch.optim.Adam(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        if config.resume_from:
            print(f"Resuming from {config.resume_from}")
            ckpt = torch.load(config.resume_from, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        criterion = nn.CrossEntropyLoss()

        wandb_run.define_metric("batch_step")
        wandb_run.define_metric("batch_loss", step_metric="batch_step")
        wandb_run.define_metric("val_step")
        wandb_run.define_metric("val_loss", step_metric="val_step")
        wandb_run.define_metric("val_acc1", step_metric="val_step")
        wandb_run.define_metric("val_acc5", step_metric="val_step")
        wandb_run.define_metric("val_roc_auc", step_metric="val_step")

        val_every = max(1, int(len(train_loader) * config.val_every_epochs))
        print(f"val_every={val_every} steps ({config.val_every_epochs} epoch equivalent)")

        train_n_steps(
            model,
            itertools.cycle(train_loader),
            val_loader,
            optimizer,
            criterion,
            transform,
            device,
            label2idx,
            config.batch_size,
            config.total_steps,
            config.total_val_rounds,
            val_every,
            wandb_run,
            run_dir,
        )
