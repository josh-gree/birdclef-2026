import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader

from birdclef_2026.data.dataset import FixedWindowDataset, OneHotLabelDataset, RandomWindowDataset


def _stratified_split_and_balance(
    index: pd.DataFrame,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Stratified train/val split with oversampling to balance train classes.

    Returns (train_indices, val_indices).
    """
    val_indices, train_indices_by_label = [], {}
    for label, group in index.groupby("primary_label"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = min(max(1, int(len(idx) * val_fraction)), len(idx) - 1)
        val_indices.extend(idx[:n_val])
        train_indices_by_label[label] = idx[n_val:].tolist()

    max_count = max(len(v) for v in train_indices_by_label.values())
    train_indices = []
    for idx_list in train_indices_by_label.values():
        n = len(idx_list)
        repeats, remainder = divmod(max_count, n)
        train_indices.extend(idx_list * repeats + idx_list[:remainder])

    return train_indices, val_indices


def _split_by_file(
    index: pd.DataFrame,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Split by file to avoid leaking recordings across splits.

    Returns (train_indices, val_indices).
    """
    files = index["filename"].unique().tolist()
    rng.shuffle(files)
    n_val_files = max(1, int(len(files) * val_fraction))
    val_files = files[:n_val_files]

    val_mask = index["filename"].isin(val_files)
    val_indices = index[val_mask].index.tolist()
    train_indices = index[~val_mask].index.tolist()

    return train_indices, val_indices


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
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """Stratified train/val split and DataLoaders.

    Returns (train_loader, val_loader, label2idx).
    """
    index = pd.read_parquet(index_path)
    label2idx = _label2idx_from_taxonomy(taxonomy_path)

    rng = np.random.default_rng(seed)
    train_indices, val_indices = _stratified_split_and_balance(index, val_fraction, rng)

    if max_samples_per_split is not None:
        train_indices = train_indices[:max_samples_per_split]
        val_indices = val_indices[:max_samples_per_split]

    train_loader = DataLoader(
        OneHotLabelDataset(RandomWindowDataset(audio_path, index_path, indices=train_indices), label2idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        OneHotLabelDataset(RandomWindowDataset(audio_path, index_path, indices=val_indices, seed=0), label2idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
    )

    return train_loader, val_loader, label2idx


def build_combined_dataloaders(
    sl_audio_path: str,
    sl_index_path: str,
    ss_audio_path: str,
    ss_index_path: str,
    taxonomy_path: str,
    batch_size: int,
    val_fraction: float = 0.1,
    soundscape_repeat: int = 5,
    max_samples_per_split: int | None = None,
    seed: int = 42,
) -> tuple[DataLoader, dict[str, DataLoader], dict[str, int]]:
    """Combined single-label + soundscape DataLoaders.

    Returns (train_loader, val_loaders, label2idx) where val_loaders is a dict
    with keys "single_label" and "soundscape".
    """
    sl_index = pd.read_parquet(sl_index_path)
    ss_index = pd.read_parquet(ss_index_path)

    label2idx = _label2idx_from_taxonomy(taxonomy_path)

    rng = np.random.default_rng(seed)

    # Single-label: stratified split + balance
    sl_train_indices, sl_val_indices = _stratified_split_and_balance(sl_index, val_fraction, rng)

    # Soundscape: split by file + repeat
    ss_train_indices, ss_val_indices = _split_by_file(ss_index, val_fraction, rng)
    ss_train_indices = ss_train_indices * soundscape_repeat

    if max_samples_per_split is not None:
        sl_train_indices = sl_train_indices[:max_samples_per_split]
        sl_val_indices = sl_val_indices[:max_samples_per_split]
        ss_train_indices = ss_train_indices[:max_samples_per_split]
        ss_val_indices = ss_val_indices[:max_samples_per_split]

    # Combined train
    sl_train_ds = OneHotLabelDataset(
        RandomWindowDataset(sl_audio_path, sl_index_path, indices=sl_train_indices), label2idx
    )
    ss_train_ds = OneHotLabelDataset(
        FixedWindowDataset(ss_audio_path, ss_index_path, indices=ss_train_indices), label2idx
    )
    train_loader = DataLoader(
        ConcatDataset([sl_train_ds, ss_train_ds]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
    )

    # Separate val loaders
    sl_val_loader = DataLoader(
        OneHotLabelDataset(
            RandomWindowDataset(sl_audio_path, sl_index_path, indices=sl_val_indices, seed=0), label2idx
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
    )
    ss_val_loader = DataLoader(
        OneHotLabelDataset(
            FixedWindowDataset(ss_audio_path, ss_index_path, indices=ss_val_indices), label2idx
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
    )

    val_loaders = {"single_label": sl_val_loader, "soundscape": ss_val_loader}
    return train_loader, val_loaders, label2idx
