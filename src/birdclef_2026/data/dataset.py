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


class StridedWindowDataset(Dataset):
    """Dataset that pre-computes all non-overlapping windows at initialisation.

    Intended for validation: the full set of ``(row_idx, window_start)`` pairs
    is fixed at ``__init__`` time, so the same index always returns the same
    audio slice regardless of when it is called.

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
    stride : int, optional
        Step size between consecutive window starts. Defaults to
        ``WINDOW_SAMPLES`` (non-overlapping). Clips shorter than
        ``WINDOW_SAMPLES`` contribute zero windows.
    """

    def __init__(
        self,
        audio_path: str,
        index_path: str,
        indices: list[int] | None = None,
        stride: int = WINDOW_SAMPLES,
    ):
        self.audio = np.load(audio_path, mmap_mode="r")
        index = pd.read_parquet(index_path)
        if indices is not None:
            index = index.iloc[indices].reset_index(drop=True)
        self.index = index

        self._windows: list[tuple[int, int]] = []
        for i, row in enumerate(index.itertuples()):
            clip_len = row.offset_end - row.offset_start
            for start in range(0, clip_len - WINDOW_SAMPLES + 1, stride):
                self._windows.append((i, start))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, str]:
        row_idx, start = self._windows[i]
        row = self.index.iloc[row_idx]
        window = self.audio[row.offset_start + start : row.offset_start + start + WINDOW_SAMPLES]
        waveform = torch.from_numpy(window.astype(np.float32) / 32767.0)
        return waveform, row.primary_label
