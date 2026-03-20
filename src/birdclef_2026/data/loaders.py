import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from birdclef_2026.data.dataset import RandomWindowDataset


def build_dataloaders(
    audio_path: str,
    index_path: str,
    batch_size: int,
    val_fraction: float = 0.1,
    max_samples_per_split: int | None = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """Stratified train/val split and DataLoaders.

    Returns (train_loader, val_loader, label2idx).
    """
    index = pd.read_parquet(index_path)
    labels = sorted(index["primary_label"].unique())
    label2idx = {label: i for i, label in enumerate(labels)}

    rng = np.random.default_rng(seed)
    val_indices, train_indices = [], []
    for _, group in index.groupby("primary_label"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_fraction))
        val_indices.extend(idx[:n_val])
        train_indices.extend(idx[n_val:])

    if max_samples_per_split is not None:
        train_indices = train_indices[:max_samples_per_split]
        val_indices = val_indices[:max_samples_per_split]

    train_loader = DataLoader(
        RandomWindowDataset(audio_path, index_path, indices=train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        RandomWindowDataset(audio_path, index_path, indices=val_indices, seed=0),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
    )

    return train_loader, val_loader, label2idx
