import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from birdclef_2026.experiments.baseline.metrics import topk_correct


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    transform: nn.Module,
    device: torch.device,
    label2idx: dict[str, int],
    batch_size: int,
    epoch: int,
    n_epochs: int,
    wandb_run,
    step: int,
) -> int:
    model.train()
    t0 = time.perf_counter()
    n_batches = len(loader)
    for batch_idx, (waveforms, label_strings) in enumerate(loader):
        waveforms = waveforms.to(device)
        targets = torch.tensor([label2idx[label] for label in label_strings], device=device)

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
            rate = (batch_idx + 1) * batch_size / elapsed
            eta = (n_batches - (batch_idx + 1)) * batch_size / rate
            print(
                f"  epoch {epoch}/{n_epochs} batch {batch_idx + 1}/{n_batches} — loss {loss.item():.4f} — {rate:.0f} samples/s — ETA {eta:.0f}s"
            )

    return step


def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    transform: nn.Module,
    device: torch.device,
    label2idx: dict[str, int],
) -> dict[str, float]:
    model.eval()
    total_loss, correct1, correct5, total = 0.0, 0, 0, 0
    with torch.no_grad():
        for waveforms, label_strings in loader:
            waveforms = waveforms.to(device)
            targets = torch.tensor([label2idx[label] for label in label_strings], device=device)
            images = transform(waveforms)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)
            correct1 += topk_correct(logits, targets, k=1)
            correct5 += topk_correct(logits, targets, k=5)
            total += len(targets)
    return {
        "val_loss": total_loss / total,
        "val_acc1": correct1 / total,
        "val_acc5": correct5 / total,
    }
