import numpy as np
import pandas as pd
import pytest
import torch

from birdclef_2026.data.dataset import (
    WINDOW_SAMPLES,
    RandomWindowDataset,
)


def make_memmap_and_index(tmp_path, clip_lengths, labels=None):
    if labels is None:
        labels = [f"bird{i}" for i in range(len(clip_lengths))]

    total = sum(clip_lengths)
    audio = np.zeros(total, dtype=np.int16)

    offset = 0
    offsets = [0]
    for length in clip_lengths:
        audio[offset : offset + length] = np.arange(length, dtype=np.int16) % 1000
        offset += length
        offsets.append(offset)

    audio_path = str(tmp_path / "audio.npy")
    np.save(audio_path, audio)

    rows = [
        {
            "filename": f"clip{i}.ogg",
            "primary_label": labels[i],
            "offset_start": offsets[i],
            "offset_end": offsets[i + 1],
        }
        for i in range(len(clip_lengths))
    ]
    index_path = str(tmp_path / "index.parquet")
    pd.DataFrame(rows).to_parquet(index_path, index=False)

    return audio_path, index_path, audio


@pytest.fixture()
def two_clips(tmp_path):
    """Two clips, each 2 × WINDOW_SAMPLES long."""
    return make_memmap_and_index(
        tmp_path,
        clip_lengths=[2 * WINDOW_SAMPLES, 2 * WINDOW_SAMPLES],
        labels=["sparrow", "robin"],
    )


@pytest.fixture()
def short_and_exact(tmp_path):
    """One clip shorter than WINDOW_SAMPLES, one exactly WINDOW_SAMPLES long."""
    return make_memmap_and_index(
        tmp_path,
        clip_lengths=[WINDOW_SAMPLES - 1, WINDOW_SAMPLES],
        labels=["short", "exact"],
    )


# ---------------------------------------------------------------------------
# RandomWindowDataset
# ---------------------------------------------------------------------------

def test_random_len_equals_number_of_files(two_clips):
    """__len__ returns one entry per file, not per window.

    Why: train dataloaders iterate once per file per epoch; a window-based
    length would silently inflate the epoch size.
    """
    audio_path, index_path, _ = two_clips
    ds = RandomWindowDataset(audio_path, index_path)
    assert len(ds) == 2


def test_random_getitem_shape_dtype_label(two_clips):
    """__getitem__ returns a float32 tensor of WINDOW_SAMPLES length and a string label.

    Why: downstream transforms (mel spectrogram, normalisation) assume float32
    input of a fixed length; a wrong dtype or length would cause silent
    numerical errors or shape mismatches inside the model.
    """
    audio_path, index_path, _ = two_clips
    ds = RandomWindowDataset(audio_path, index_path)
    waveform, label = ds[0]
    assert waveform.dtype == torch.float32
    assert waveform.shape == (WINDOW_SAMPLES,)
    assert isinstance(label, str)


def test_random_window_at_max_start_has_full_length(two_clips, monkeypatch):
    """A window starting at the maximum valid offset still returns WINDOW_SAMPLES samples.

    Why: an off-by-one in max_start would allow reads past the clip boundary,
    returning a truncated tensor. Forcing randint to the maximum start exercises
    that exact boundary.
    """
    audio_path, index_path, _ = two_clips
    ds = RandomWindowDataset(audio_path, index_path)
    clip_len = 2 * WINDOW_SAMPLES
    max_start = clip_len - WINDOW_SAMPLES
    monkeypatch.setattr(np.random, "randint", lambda *_: max_start)
    waveform, _ = ds[0]
    assert waveform.shape == (WINDOW_SAMPLES,)


def test_random_values_normalised(two_clips):
    """All sample values lie in [-1, 1] after int16 → float32 conversion.

    Why: the normalisation constant 32767.0 must be applied before the tensor
    is returned; values outside [-1, 1] would saturate tanh activations and
    destabilise early training.
    """
    audio_path, index_path, _ = two_clips
    ds = RandomWindowDataset(audio_path, index_path)
    for i in range(len(ds)):
        waveform, _ = ds[i]
        assert waveform.min() >= -1.0
        assert waveform.max() <= 1.0


def test_random_indices_subset(two_clips):
    """Passing indices= restricts the dataset to the specified rows only.

    Why: train/val splitting is done by passing disjoint index lists; if the
    subset is ignored, both splits would see all data and validation accuracy
    would be meaningless.
    """
    audio_path, index_path, _ = two_clips
    ds = RandomWindowDataset(audio_path, index_path, indices=[1])
    assert len(ds) == 1
    _, label = ds[0]
    assert label == "robin"

