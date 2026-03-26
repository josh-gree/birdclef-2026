import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SAMPLE_RATE = 32000
WINDOW_SAMPLES = 5 * SAMPLE_RATE  # 160,000 samples


class RandomWindowDataset(Dataset):
    """Dataset that returns a random 5-second window per file on each access.

    Intended for training: each call to ``__getitem__`` draws a new random
    window, so the model sees different crops of the same clip across epochs.

    Parameters
    ----------
    audio_path : str
        Path to the int16 memmap ``.npy`` file produced by the preparation
        pipeline.
    index_path : str
        Path to the index parquet file with columns ``offset_start``,
        ``offset_end``, and ``primary_label``.
    indices : list of int, optional
        Row indices into the index to use. If ``None``, all rows are used.
    """

    def __init__(
        self,
        audio_path: str,
        index_path: str,
        indices: list[int] | None = None,
        seed: int | None = None,
    ):
        self.audio = np.load(audio_path, mmap_mode="r")
        index = pd.read_parquet(index_path)
        if indices is not None:
            index = index.iloc[indices].reset_index(drop=True)
        self.index = index
        self.seed = seed

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, str]:
        row = self.index.iloc[i]
        clip_len = row.offset_end - row.offset_start
        max_start = clip_len - WINDOW_SAMPLES
        if self.seed is not None:
            rng = np.random.default_rng(self.seed + i)
            start = int(rng.integers(0, max_start + 1))
        else:
            start = np.random.randint(0, max_start + 1)
        window = self.audio[row.offset_start + start : row.offset_start + start + WINDOW_SAMPLES]
        waveform = torch.from_numpy(window.astype(np.float32) / 32767.0)
        return waveform, row.primary_label


class MixupDataset(Dataset):
    """Dataset that mixes k+1 random 5-second windows with equal weights.

    For each item i, picks a random 5s window from clip i, then samples k
    additional clips (k drawn uniformly from 1–5) and mixes all k+1 windows
    with equal weights. Returns a semicolon-joined label string of all unique
    primary labels, compatible with ``OneHotLabelDataset``.

    Parameters
    ----------
    audio_path : str
        Path to the int16 memmap ``.npy`` file.
    index_path : str
        Path to the index parquet file with columns ``offset_start``,
        ``offset_end``, and ``primary_label``.
    indices : list of int, optional
        Row indices into the index to use. If ``None``, all rows are used.
    seed : int, optional
        If set, makes window and mix sampling deterministic per item.
    """

    def __init__(
        self,
        audio_path: str,
        index_path: str,
        indices: list[int] | None = None,
        seed: int | None = None,
    ):
        self.audio = np.load(audio_path, mmap_mode="r")
        index = pd.read_parquet(index_path)
        if indices is not None:
            index = index.iloc[indices].reset_index(drop=True)
        self.index = index
        self.seed = seed

    def _get_window(self, row, rng: np.random.Generator) -> np.ndarray:
        clip_len = row.offset_end - row.offset_start
        max_start = clip_len - WINDOW_SAMPLES
        start = int(rng.integers(0, max_start + 1))
        return self.audio[row.offset_start + start : row.offset_start + start + WINDOW_SAMPLES].astype(np.float32)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, str]:
        rng = np.random.default_rng(self.seed + i if self.seed is not None else None)
        k = int(rng.integers(1, 6))
        other_indices = [j for j in range(len(self.index)) if j != i]
        chosen = rng.choice(other_indices, size=k, replace=False).tolist()
        all_indices = [i] + chosen

        mixed = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        labels = []
        for j in all_indices:
            row = self.index.iloc[j]
            mixed += self._get_window(row, rng)
            labels.append(row.primary_label)

        mixed /= len(all_indices)
        waveform = torch.from_numpy(mixed / 32767.0)
        label_str = ";".join(dict.fromkeys(labels))  # unique, order-preserving
        return waveform, label_str


class FixedWindowDataset(Dataset):
    """Dataset that returns a fixed 5-second window per row.

    Intended for data where each index row already represents an exact
    5-second segment (e.g. soundscape windows).

    Parameters
    ----------
    audio_path : str
        Path to the int16 memmap ``.npy`` file.
    index_path : str
        Path to the index parquet file with columns ``offset_start``,
        ``offset_end``, and ``primary_label``.
    indices : list of int, optional
        Row indices into the index to use. If ``None``, all rows are used.
    """

    def __init__(
        self,
        audio_path: str,
        index_path: str,
        indices: list[int] | None = None,
    ):
        self.audio = np.load(audio_path, mmap_mode="r")
        index = pd.read_parquet(index_path)
        if indices is not None:
            index = index.iloc[indices].reset_index(drop=True)
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, str]:
        row = self.index.iloc[i]
        window = self.audio[row.offset_start : row.offset_end]
        waveform = torch.from_numpy(window.astype(np.float32) / 32767.0)
        return waveform, row.primary_label


class OneHotLabelDataset(Dataset):
    """Wraps a dataset, converting string labels to one-hot float tensors.

    Handles multi-label entries (semicolon-separated, e.g. ``"species_a;species_b"``)
    by setting multiple indices to 1.0. Needed for soundscape data and
    BCEWithLogitsLoss training.

    Parameters
    ----------
    dataset : Dataset
        The base dataset to wrap.
    label2idx : dict[str, int]
        Mapping from label string to class index.
    """

    def __init__(self, dataset: Dataset, label2idx: dict[str, int]):
        self.dataset = dataset
        self.label2idx = label2idx
        self.n_classes = len(label2idx)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        waveform, label_str = self.dataset[i]
        label_vec = torch.zeros(self.n_classes, dtype=torch.float32)
        for label in label_str.split(";"):
            label_vec[self.label2idx[label]] = 1.0
        return waveform, label_vec
