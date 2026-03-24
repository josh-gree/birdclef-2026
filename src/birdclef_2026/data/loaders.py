import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from birdclef_2026.data.dataset import RandomWindowDataset


def _label2idx_from_taxonomy(taxonomy_path: str) -> dict[str, int]:
    """Build canonical label2idx from taxonomy.csv."""
    labels = pd.read_csv(taxonomy_path)["primary_label"].astype(str).tolist()
    return {label: i for i, label in enumerate(sorted(labels))}


def build_dataloaders(
    audio_path: str,
    index_path: str,
    taxonomy_path: str,
    batch_size: int,
    val_fraction: float = 0.1,
    max_samples_per_split: int | None = None,
    seed: int = 42,
    balance_train: bool = False,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """Stratified train/val split and DataLoaders.

    Returns (train_loader, val_loader, label2idx).
    """
    index = pd.read_parquet(index_path)
    label2idx = _label2idx_from_taxonomy(taxonomy_path)

    rng = np.random.default_rng(seed)
    val_indices, train_indices_by_label = [], {}
    for label, group in index.groupby("primary_label"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = min(max(1, int(len(idx) * val_fraction)), len(idx) - 1)
        val_indices.extend(idx[:n_val])
        train_indices_by_label[label] = idx[n_val:].tolist()

    if balance_train:
        max_count = max(len(v) for v in train_indices_by_label.values())
        train_indices = []
        for idx_list in train_indices_by_label.values():
            n = len(idx_list)
            repeats, remainder = divmod(max_count, n)
            train_indices.extend(idx_list * repeats + idx_list[:remainder])
    else:
        train_indices = [i for lst in train_indices_by_label.values() for i in lst]

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
