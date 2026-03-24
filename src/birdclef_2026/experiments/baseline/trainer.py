import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from birdclef_2026.experiments.baseline.metrics import macro_roc_auc


def train_n_steps(
    model: nn.Module,
    train_iter,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    transform: nn.Module,
    device: torch.device,
    label2idx: dict[str, int],
    batch_size: int,
    total_steps: int | None,
    total_val_rounds: int | None,
    val_every: int,
    wandb_run,
    run_dir: Path,
) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    transform.train()
    t0 = time.perf_counter()
    val_step = 0
    for step, (waveforms, label_strings) in enumerate(train_iter):
        if total_steps is not None and step >= total_steps:
            break
        if total_val_rounds is not None and val_step >= total_val_rounds:
            break

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

        if (step + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            rate = (step + 1) * batch_size / elapsed
            print(f"  step {step + 1} val_round {val_step} — loss {loss.item():.4f} — {rate:.0f} samples/s")

        if (step + 1) % val_every == 0:
            _eval(model, val_loader, criterion, transform, device, label2idx, step + 1, val_step, wandb_run)
            torch.save(
                {"val_step": val_step, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                checkpoint_dir / f"val_{val_step:04d}.pt",
            )
            val_step += 1
            model.train()
            transform.train()


def _eval(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    transform: nn.Module,
    device: torch.device,
    label2idx: dict[str, int],
    step: int,
    val_step: int,
    wandb_run,
) -> None:
    model.eval()
    transform.eval()
    total_loss, total = 0.0, 0
    all_logits, all_targets = [], []
    with torch.no_grad():
        for waveforms, label_strings in loader:
            waveforms = waveforms.to(device)
            targets = torch.tensor([label2idx[label] for label in label_strings], device=device)
            images = transform(waveforms)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)
            total += len(targets)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    roc_auc = macro_roc_auc(torch.cat(all_logits), torch.cat(all_targets), n_classes=len(label2idx))
    metrics = {
        "val_loss": total_loss / total,
        "val_roc_auc": roc_auc,
        "val_step": val_step,
    }
    print(
        f"Val {val_step} (step {step}) — val_loss: {metrics['val_loss']:.4f} — roc_auc: {roc_auc:.4f}"
    )
    wandb_run.log(metrics)
