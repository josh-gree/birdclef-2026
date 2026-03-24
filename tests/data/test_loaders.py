"""Tests for build_dataloaders: splitting, stratification, and balancing.

Each test targets one specific behaviour so failures are unambiguous.
We use offset_start as a unique clip identifier throughout.
"""

import numpy as np
import pandas as pd
import pytest

from birdclef_2026.data.dataset import WINDOW_SAMPLES
from birdclef_2026.data.loaders import build_dataloaders


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dataset(tmp_path, clips_per_class: dict[str, int]) -> tuple[str, str, str]:
    """Write a memmap + index + taxonomy with the given number of clips per class.

    Each clip is exactly WINDOW_SAMPLES long so every clip yields one valid
    window.  Returns (audio_path, index_path, taxonomy_path).
    """
    rows = []
    offset = 0
    for label, n_clips in clips_per_class.items():
        for _ in range(n_clips):
            rows.append(
                {
                    "primary_label": label,
                    "offset_start": offset,
                    "offset_end": offset + WINDOW_SAMPLES,
                }
            )
            offset += WINDOW_SAMPLES

    total = sum(n * WINDOW_SAMPLES for n in clips_per_class.values())
    audio = np.zeros(total, dtype=np.int16)
    audio_path = str(tmp_path / "audio.npy")
    np.save(audio_path, audio)

    index_path = str(tmp_path / "index.parquet")
    pd.DataFrame(rows).to_parquet(index_path, index=False)

    taxonomy_path = str(tmp_path / "taxonomy.csv")
    pd.DataFrame({"primary_label": list(clips_per_class.keys())}).to_csv(taxonomy_path, index=False)

    return audio_path, index_path, taxonomy_path


def train_index(loader) -> pd.DataFrame:
    return loader.dataset.index


def val_index(loader) -> pd.DataFrame:
    return loader.dataset.index


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def test_train_and_val_are_disjoint(tmp_path):
    """No clip appears in both train and val.

    Why: a leaky split inflates val accuracy and makes metrics meaningless.
    We identify clips by offset_start, which is unique per clip.
    """
    audio_path, index_path, taxonomy_path = make_dataset(tmp_path, {"a": 10, "b": 10})
    train_loader, val_loader, _ = build_dataloaders(
        audio_path, index_path, taxonomy_path, batch_size=4
    )

    train_offsets = set(train_index(train_loader)["offset_start"])
    val_offsets = set(val_index(val_loader)["offset_start"])
    assert train_offsets.isdisjoint(val_offsets), (
        f"Clips present in both splits: {train_offsets & val_offsets}"
    )


def test_every_class_appears_in_val(tmp_path):
    """Every class with more than one clip has at least one clip in val.

    Why: if a class is absent from val we cannot measure per-class performance.
    """
    audio_path, index_path, taxonomy_path = make_dataset(tmp_path, {"a": 10, "b": 10, "c": 5})
    _, val_loader, _ = build_dataloaders(audio_path, index_path, taxonomy_path, batch_size=4)

    val_labels = set(val_index(val_loader)["primary_label"])
    assert val_labels == {"a", "b", "c"}


def test_every_class_appears_in_train(tmp_path):
    """Every class has at least one clip in train.

    Why: the model cannot learn a class it never sees during training.
    """
    audio_path, index_path, taxonomy_path = make_dataset(tmp_path, {"a": 10, "b": 10, "c": 5})
    train_loader, _, _ = build_dataloaders(audio_path, index_path, taxonomy_path, batch_size=4)

    train_labels = set(train_index(train_loader)["primary_label"])
    assert train_labels == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Single-sample classes
# ---------------------------------------------------------------------------

def test_single_sample_class_goes_to_train(tmp_path):
    """A class with exactly one clip puts that clip in train, not val.

    Why: with only one example we can either train or eval on it, not both.
    We prioritise training so the model at least sees the class.
    """
    audio_path, index_path, taxonomy_path = make_dataset(tmp_path, {"common": 10, "rare": 1})
    train_loader, val_loader, _ = build_dataloaders(
        audio_path, index_path, taxonomy_path, batch_size=4
    )

    assert "rare" in set(train_index(train_loader)["primary_label"]), (
        "single-sample class must appear in train"
    )
    assert "rare" not in set(val_index(val_loader)["primary_label"]), (
        "single-sample class must not appear in val"
    )


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------

def test_balance_train_equalises_class_counts(tmp_path):
    """With balance_train=True every class has the same number of rows in train.

    Why: the whole point of balancing is equal class representation per step.
    We check via value_counts on the train dataset index.
    """
    # "common" has 20 clips, "rare" has 2 — a 10× imbalance
    audio_path, index_path, taxonomy_path = make_dataset(tmp_path, {"common": 20, "rare": 2})
    train_loader, _, _ = build_dataloaders(
        audio_path, index_path, taxonomy_path, batch_size=4, balance_train=True
    )

    counts = train_index(train_loader)["primary_label"].value_counts()
    assert counts.nunique() == 1, (
        f"Expected equal class counts, got: {counts.to_dict()}"
    )


def test_balance_train_target_is_max_class_count(tmp_path):
    """The balanced target count equals the largest class's train size.

    Why: we oversample minorities up to the majority — not to some arbitrary
    cap — so the majority class is never downsampled.
    """
    # "big" has 10 clips. With val_fraction=0.1, n_val=1, so 9 go to train.
    # "small" has 3 clips. n_val=1, so 2 go to train.
    # After balancing both classes should have 9 rows in train.
    audio_path, index_path, taxonomy_path = make_dataset(tmp_path, {"big": 10, "small": 3})
    train_loader, _, _ = build_dataloaders(
        audio_path, index_path, taxonomy_path, batch_size=4, balance_train=True, val_fraction=0.1
    )

    counts = train_index(train_loader)["primary_label"].value_counts()
    assert counts["big"] == counts["small"], (
        f"Expected both classes to have equal counts, got: {counts.to_dict()}"
    )
    assert counts["big"] == 9, (
        f"Expected target count of 9 (majority train size), got {counts['big']}"
    )


def test_balance_train_false_does_not_oversample(tmp_path):
    """With balance_train=False train counts reflect the raw split, not oversampled.

    Why: balancing is opt-in; the default must not silently inflate minority
    classes and change the effective epoch length.
    """
    audio_path, index_path, taxonomy_path = make_dataset(tmp_path, {"big": 20, "small": 2})
    train_loader_balanced, _, _ = build_dataloaders(
        audio_path, index_path, taxonomy_path, batch_size=4, balance_train=True
    )
    train_loader_raw, _, _ = build_dataloaders(
        audio_path, index_path, taxonomy_path, batch_size=4, balance_train=False
    )

    assert len(train_index(train_loader_balanced)) > len(train_index(train_loader_raw)), (
        "balanced train should be larger than raw train due to oversampling"
    )
