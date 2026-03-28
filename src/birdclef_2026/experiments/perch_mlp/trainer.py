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
    device: torch.device,
    batch_size: int,
    total_steps: int | None,
    total_val_rounds: int | None,
    val_every: int,
    wandb_run,
    run_dir: Path,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    t0 = time.perf_counter()
    val_step = 0
    for step, (embeddings, targets) in enumerate(train_iter):
        if total_steps is not None and step >= total_steps:
            break
        if total_val_rounds is not None and val_step >= total_val_rounds:
            break

        embeddings = embeddings.to(device)
        targets = targets.to(device)

        logits = model(embeddings)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            pos_mask = targets.bool()
            mean_pos_logit = logits[pos_mask].mean().item()
            mean_neg_logit = logits[~pos_mask].mean().item()

        log = {
            "batch_loss": loss.item(),
            "mean_pos_logit": mean_pos_logit,
            "mean_neg_logit": mean_neg_logit,
            "batch_step": step,
        }
        if scheduler is not None:
            log["lr"] = scheduler.get_last_lr()[0]
        wandb_run.log(log)

        if (step + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            rate = (step + 1) * batch_size / elapsed
            print(f"  step {step + 1} val_round {val_step} — loss {loss.item():.4f} — {rate:.0f} samples/s")

        if (step + 1) % val_every == 0:
            _eval(model, val_loader, criterion, device, step + 1, val_step, wandb_run)
            torch.save(
                {"val_step": val_step, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                checkpoint_dir / f"val_{val_step:04d}.pt",
            )
            val_step += 1
            model.train()


def _eval(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    step: int,
    val_step: int,
    wandb_run,
) -> None:
    model.eval()
    total_loss, total = 0.0, 0
    all_logits, all_targets = [], []
    with torch.no_grad():
        for embeddings, targets in val_loader:
            embeddings = embeddings.to(device)
            targets = targets.to(device)
            logits = model(embeddings)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)
            total += len(targets)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    roc_auc = macro_roc_auc(torch.cat(all_logits), torch.cat(all_targets))
    avg_loss = total_loss / total
    print(f"  val (step {step}, round {val_step}) — loss: {avg_loss:.4f} — roc_auc: {roc_auc:.4f}")
    wandb_run.log({"val_loss": avg_loss, "val_roc_auc": roc_auc, "val_step": val_step})
