import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from birdclef_2026.experiments.baseline.metrics import macro_roc_auc
from birdclef_2026.experiments.attn_sed.model import AttnSEDModel


def train_attn_sed(
    model: AttnSEDModel,
    train_iter,
    val_loaders: dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    transform: nn.Module,
    device: torch.device,
    batch_size: int,
    total_steps: int,
    total_val_rounds: int | None,
    val_every: int,
    wandb_run,
    run_dir: Path,
    timewise_weight: float = 0.5,
    temperature: float = 0.1,
) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.GradScaler(device=device.type, enabled=(device.type == "cuda"))

    model.train()
    transform.train()
    t0 = time.perf_counter()
    val_step = 0

    for step, (waveforms, targets) in enumerate(train_iter):
        if step >= total_steps:
            break
        if total_val_rounds is not None and val_step >= total_val_rounds:
            break

        waveforms = waveforms.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            images = transform(waveforms)  # (B, 1, H, W)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits, timewise = model.forward_for_training(images)
            lse = temperature * torch.logsumexp(timewise / temperature, dim=1)
            loss = (
                (1.0 - timewise_weight) * F.binary_cross_entropy_with_logits(logits, targets)
                + timewise_weight * F.binary_cross_entropy_with_logits(lse, targets)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            pos_mask = targets.bool()
            mean_pos_logit = logits[pos_mask].mean().item() if pos_mask.any() else 0.0
            mean_neg_logit = logits[~pos_mask].mean().item() if (~pos_mask).any() else 0.0

        wandb_run.log({
            "batch_loss": loss.item(),
            "mean_pos_logit": mean_pos_logit,
            "mean_neg_logit": mean_neg_logit,
            "lr": scheduler.get_last_lr()[0],
            "batch_step": step,
        })

        if (step + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            rate = (step + 1) * batch_size / elapsed
            print(f"  step {step + 1} val_round {val_step} — loss {loss.item():.4f} — lr {scheduler.get_last_lr()[0]:.2e} — {rate:.0f} samples/s")

        if (step + 1) % val_every == 0:
            _eval_all(model, val_loaders, transform, device, step + 1, val_step, wandb_run)
            torch.save(
                {"val_step": val_step, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                checkpoint_dir / f"val_{val_step:04d}.pt",
            )
            val_step += 1
            model.train()
            transform.train()


def _eval_all(
    model: AttnSEDModel,
    val_loaders: dict[str, DataLoader],
    transform: nn.Module,
    device: torch.device,
    step: int,
    val_step: int,
    wandb_run,
) -> None:
    model.eval()
    transform.eval()
    metrics = {"val_step": val_step}
    for name, loader in val_loaders.items():
        prefix = f"val_{name}"
        total_loss, total = 0.0, 0
        all_logits, all_targets = [], []
        with torch.no_grad():
            for waveforms, targets in loader:
                waveforms = waveforms.to(device)
                targets = targets.to(device)
                images = transform(waveforms)
                logits = model(images)
                loss = F.binary_cross_entropy_with_logits(logits, targets)
                total_loss += loss.item() * len(targets)
                total += len(targets)
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())
        roc_auc = macro_roc_auc(torch.cat(all_logits), torch.cat(all_targets))
        metrics[f"{prefix}_loss"] = total_loss / total
        metrics[f"{prefix}_roc_auc"] = roc_auc
        print(f"  {prefix} (step {step}, round {val_step}) — loss: {total_loss / total:.4f} — roc_auc: {roc_auc:.4f}")
    wandb_run.log(metrics)
