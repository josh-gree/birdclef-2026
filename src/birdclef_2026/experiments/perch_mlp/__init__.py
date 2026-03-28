import itertools
from pathlib import Path

import torch
import torch.nn as nn
from pydantic import BaseModel, model_validator
from wm import Experiment

from birdclef_2026.data.loaders import build_embedding_dataloaders
from birdclef_2026.experiments.baseline.model import build_head
from birdclef_2026.experiments.perch_mlp.trainer import train_n_steps

EMBEDDING_DIM = 1280


class BirdCLEFPerchMLP(Experiment):
    name = "birdclef_perch_mlp"
    gpu = "A10G"

    class Config(BaseModel):
        lr: float = 1e-3
        batch_size: int = 256
        total_steps: int | None = None
        total_val_rounds: int | None = None
        val_every_epochs: float = 1.0
        val_fraction: float = 0.1
        hidden: int = 512
        dropout: float = 0.0
        weight_decay: float = 0.0
        use_schedule: bool = False
        warmup_fraction: float = 0.25

        @model_validator(mode="after")
        def _exactly_one_stopping_criterion(self):
            if (self.total_steps is None) == (self.total_val_rounds is None):
                raise ValueError("Exactly one of total_steps or total_val_rounds must be set")
            return self

    @staticmethod
    def run(config: "BirdCLEFPerchMLP.Config", wandb_run, run_dir: Path) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader, val_loader, label2idx = build_embedding_dataloaders(
            embeddings_path="/data/soundscape_embeddings.npy",
            index_path="/data/soundscape_index.parquet",
            taxonomy_path="/data/taxonomy.csv",
            batch_size=config.batch_size,
            val_fraction=config.val_fraction,
        )
        n_classes = len(label2idx)

        model = build_head(EMBEDDING_DIM, n_classes, dropout=config.dropout, hidden=config.hidden).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        val_every = max(1, int(len(train_loader) * config.val_every_epochs))
        total_steps = config.total_steps or (config.total_val_rounds * val_every)

        scheduler = None
        if config.use_schedule:
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
        wandb_run.define_metric("batch_loss", step_metric="batch_step")
        wandb_run.define_metric("mean_pos_logit", step_metric="batch_step")
        wandb_run.define_metric("mean_neg_logit", step_metric="batch_step")
        if config.use_schedule:
            wandb_run.define_metric("lr", step_metric="batch_step")
        wandb_run.define_metric("val_step")
        wandb_run.define_metric("val_loss", step_metric="val_step")
        wandb_run.define_metric("val_roc_auc", step_metric="val_step")

        print(f"val_every={val_every} steps ({config.val_every_epochs} epoch equivalent)")

        train_n_steps(
            model,
            itertools.cycle(train_loader),
            val_loader,
            optimizer,
            criterion,
            device,
            config.batch_size,
            config.total_steps,
            config.total_val_rounds,
            val_every,
            wandb_run,
            run_dir,
            scheduler=scheduler,
        )
