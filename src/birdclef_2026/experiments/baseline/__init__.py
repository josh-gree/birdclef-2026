import shutil

import torch
import torch.nn as nn
from pydantic import BaseModel
from wm import Experiment

from birdclef_2026.data.loaders import build_dataloaders
from birdclef_2026.data.transforms import build_spectrogram_pipeline
from birdclef_2026.experiments.baseline.losses import build_loss
from birdclef_2026.experiments.baseline.model import (
    build_frozen_efficientnet_b3_backbone,
    build_mlp_head,
)
from birdclef_2026.experiments.baseline.trainer import eval_one_epoch, train_one_epoch


class BirdCLEFBaseline(Experiment):
    name = "birdclef_baseline"
    gpu = "A10G"
    ephemeral_disk = (
        524288  # 512 GB — minimum allowed, needed to copy 74 GB memmap to local disk
    )

    class Config(BaseModel):
        lr: float = 1e-3
        batch_size: int = 32
        epochs: int = 10
        val_fraction: float = 0.1
        max_samples_per_split: int | None = None  # set small (e.g. 64) for a smoke run
        hidden_size: int = 512

    @staticmethod
    def run(config: "BirdCLEFBaseline.Config", wandb_run) -> None:
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
        )
        n_classes = len(label2idx)

        transform = build_spectrogram_pipeline().to(device)

        backbone = build_frozen_efficientnet_b3_backbone()
        head = build_mlp_head(backbone.num_features, n_classes, config.hidden_size)
        model = nn.Sequential(backbone, head).to(device)

        optimizer = torch.optim.Adam(head.parameters(), lr=config.lr)
        criterion = build_loss()

        wandb_run.define_metric("batch_step")
        wandb_run.define_metric("batch_loss", step_metric="batch_step")
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("epoch_val_loss", step_metric="epoch")
        wandb_run.define_metric("epoch_val_acc1", step_metric="epoch")
        wandb_run.define_metric("epoch_val_acc5", step_metric="epoch")

        step = 0
        for epoch in range(1, config.epochs + 1):
            step = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                transform,
                device,
                label2idx,
                config.batch_size,
                epoch,
                config.epochs,
                wandb_run,
                step,
            )
            metrics = eval_one_epoch(
                model, val_loader, criterion, transform, device, label2idx
            )
            print(
                f"Epoch {epoch}/{config.epochs} — val_loss: {metrics['val_loss']:.4f} — acc@1: {metrics['val_acc1']:.4f} — acc@5: {metrics['val_acc5']:.4f}"
            )
            wandb_run.log({**metrics, "epoch": epoch})
