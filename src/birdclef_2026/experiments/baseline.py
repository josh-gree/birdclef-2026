import shutil
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.utils.data import DataLoader
from wm import Experiment

from birdclef_2026.data.dataset import RandomWindowDataset, StridedWindowDataset
from birdclef_2026.data.transforms import build_spectrogram_pipeline


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

    @staticmethod
    def run(config: "BirdCLEFBaseline.Config", wandb_run) -> None:
        print("Copying data to local disk...")
        shutil.copy("/data/audio.npy", "/tmp/audio.npy")
        shutil.copy("/data/index.parquet", "/tmp/index.parquet")
        audio_path, index_path = "/tmp/audio.npy", "/tmp/index.parquet"
        print("Done.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        index = pd.read_parquet(index_path)
        labels = sorted(index["primary_label"].unique())
        label2idx = {label: i for i, label in enumerate(labels)}
        n_classes = len(labels)

        rng = np.random.default_rng(42)
        val_indices, train_indices = [], []
        for _, group in index.groupby("primary_label"):
            idx = group.index.to_numpy().copy()
            rng.shuffle(idx)
            n_val = max(1, int(len(idx) * config.val_fraction))
            val_indices.extend(idx[:n_val])
            train_indices.extend(idx[n_val:])

        if config.max_samples_per_split is not None:
            train_indices = train_indices[: config.max_samples_per_split]
            val_indices = val_indices[: config.max_samples_per_split]

        train_dataset = RandomWindowDataset(
            audio_path, index_path, indices=train_indices
        )
        val_dataset = StridedWindowDataset(audio_path, index_path, indices=val_indices)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4,
        )

        transform = build_spectrogram_pipeline().to(device)

        # Frozen EfficientNet-B3 backbone with 1-channel input, linear head
        backbone = timm.create_model(
            "efficientnet_b3", pretrained=True, in_chans=1, num_classes=0
        )
        for param in backbone.parameters():
            param.requires_grad = False
        backbone_features = backbone.num_features
        head = nn.Linear(backbone_features, n_classes)
        model = nn.Sequential(backbone, head).to(device)

        optimizer = torch.optim.Adam(head.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss()

        wandb_run.define_metric("batch_step")
        wandb_run.define_metric("batch_loss", step_metric="batch_step")
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("epoch_val_loss", step_metric="epoch")
        wandb_run.define_metric("epoch_val_acc1", step_metric="epoch")
        wandb_run.define_metric("epoch_val_acc5", step_metric="epoch")

        step = 0

        for epoch in range(1, config.epochs + 1):
            # train
            model.train()
            t0 = time.perf_counter()
            n_batches = len(train_loader)
            for batch_idx, (waveforms, label_strings) in enumerate(train_loader):
                waveforms = waveforms.to(device)
                targets = torch.tensor(
                    [label2idx[label] for label in label_strings], device=device
                )

                with torch.no_grad():
                    images = transform(waveforms)

                logits = model(images)
                loss = criterion(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                wandb_run.log({"batch_loss": loss.item(), "batch_step": step})
                step += 1

                if (batch_idx + 1) % 100 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = (batch_idx + 1) * config.batch_size / elapsed
                    eta = (n_batches - (batch_idx + 1)) * config.batch_size / rate
                    print(
                        f"  epoch {epoch}/{config.epochs} batch {batch_idx + 1}/{n_batches} — loss {loss.item():.4f} — {rate:.0f} samples/s — ETA {eta:.0f}s"
                    )

            # eval
            model.eval()
            total_loss, correct1, correct5, total = 0.0, 0, 0, 0
            with torch.no_grad():
                for waveforms, label_strings in val_loader:
                    waveforms = waveforms.to(device)
                    targets = torch.tensor(
                        [label2idx[label] for label in label_strings], device=device
                    )
                    images = transform(waveforms)
                    logits = model(images)
                    loss = criterion(logits, targets)
                    total_loss += loss.item() * len(targets)
                    top5 = logits.topk(5, dim=1).indices
                    correct1 += (top5[:, 0] == targets).sum().item()
                    correct5 += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()
                    total += len(targets)
            val_loss = total_loss / total
            val_acc1 = correct1 / total
            val_acc5 = correct5 / total
            print(
                f"Epoch {epoch}/{config.epochs} — val_loss: {val_loss:.4f} — acc@1: {val_acc1:.4f} — acc@5: {val_acc5:.4f}"
            )
            wandb_run.log(
                {
                    "epoch_val_loss": val_loss,
                    "epoch_val_acc1": val_acc1,
                    "epoch_val_acc5": val_acc5,
                    "epoch": epoch,
                }
            )
