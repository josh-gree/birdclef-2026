"""Tests for build_dataloaders and build_combined_dataloaders.

Each test targets one specific behaviour so failures are unambiguous.
We use offset_start as a unique clip identifier throughout.
"""

import numpy as np
import pandas as pd
import pytest

from birdclef_2026.data.dataset import WINDOW_SAMPLES
from birdclef_2026.data.loaders import build_combined_dataloaders, build_dataloaders


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_taxonomy(tmp_path, labels: set[str]) -> str:
    """Write a taxonomy.csv with the given labels. Returns the path."""
    taxonomy_path = str(tmp_path / "taxonomy.csv")
    pd.DataFrame({"primary_label": sorted(labels)}).to_csv(taxonomy_path, index=False)
    return taxonomy_path


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

    taxonomy_path = make_taxonomy(tmp_path, set(clips_per_class.keys()))
    return audio_path, index_path, taxonomy_path


def make_soundscape_dataset(
    tmp_path, windows_per_file: dict[str, list[str]]
) -> tuple[str, str]:
    """Write a soundscape memmap + index.

    windows_per_file maps filename → list of semicolon-separated label strings,
    one per 5s window.  Returns (audio_path, index_path).
    """
    rows = []
    offset = 0
    for filename, label_list in windows_per_file.items():
        for label_str in label_list:
            rows.append({
                "filename": filename,
                "primary_label": label_str,
                "offset_start": offset,
                "offset_end": offset + WINDOW_SAMPLES,
            })
            offset += WINDOW_SAMPLES

    total_windows = sum(len(v) for v in windows_per_file.values())
    audio = np.zeros(total_windows * WINDOW_SAMPLES, dtype=np.int16)
    audio_path = str(tmp_path / "ss_audio.npy")
    np.save(audio_path, audio)

    index_path = str(tmp_path / "ss_index.parquet")
    pd.DataFrame(rows).to_parquet(index_path, index=False)
    return audio_path, index_path


def train_index(loader) -> pd.DataFrame:
    return loader.dataset.dataset.index


def val_index(loader) -> pd.DataFrame:
    return loader.dataset.dataset.index


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

def test_balancing_equalises_class_counts(tmp_path):
    """Every class has the same number of rows in train.

    Why: the whole point of balancing is equal class representation per step.
    We check via value_counts on the train dataset index.
    """
    # "common" has 20 clips, "rare" has 2 — a 10× imbalance
    audio_path, index_path, taxonomy_path = make_dataset(tmp_path, {"common": 20, "rare": 2})
    train_loader, _, _ = build_dataloaders(
        audio_path, index_path, taxonomy_path, batch_size=4
    )

    counts = train_index(train_loader)["primary_label"].value_counts()
    assert counts.nunique() == 1, (
        f"Expected equal class counts, got: {counts.to_dict()}"
    )


def test_balancing_target_is_max_class_count(tmp_path):
    """The balanced target count equals the largest class's train size.

    Why: we oversample minorities up to the majority — not to some arbitrary
    cap — so the majority class is never downsampled.
    """
    # "big" has 10 clips. With val_fraction=0.1, n_val=1, so 9 go to train.
    # "small" has 3 clips. n_val=1, so 2 go to train.
    # After balancing both classes should have 9 rows in train.
    audio_path, index_path, taxonomy_path = make_dataset(tmp_path, {"big": 10, "small": 3})
    train_loader, _, _ = build_dataloaders(
        audio_path, index_path, taxonomy_path, batch_size=4, val_fraction=0.1
    )

    counts = train_index(train_loader)["primary_label"].value_counts()
    assert counts["big"] == counts["small"], (
        f"Expected both classes to have equal counts, got: {counts.to_dict()}"
    )
    assert counts["big"] == 9, (
        f"Expected target count of 9 (majority train size), got {counts['big']}"
    )


# ---------------------------------------------------------------------------
# Combined dataloaders
# ---------------------------------------------------------------------------

def _make_combined_fixture(tmp_path):
    """Shared fixture for combined loader tests."""
    sl_audio, sl_index, _ = make_dataset(tmp_path, {"a": 10, "b": 10, "c": 5})
    ss_audio, ss_index = make_soundscape_dataset(tmp_path, {
        "file1.ogg": ["a;d", "a;d", "d", "d"],
        "file2.ogg": ["a;e", "e", "e", "e"],
        "file3.ogg": ["d;e", "d;e", "d", "d"],
    })
    taxonomy_path = make_taxonomy(tmp_path, {"a", "b", "c", "d", "e"})
    return sl_audio, sl_index, ss_audio, ss_index, taxonomy_path


def test_combined_label2idx_is_union(tmp_path):
    """label2idx contains all labels from both sources.

    Why: soundscape-only labels must have an index or OneHotLabelDataset crashes.
    """
    sl_audio, sl_index, ss_audio, ss_index, taxonomy_path = _make_combined_fixture(tmp_path)
    _, _, label2idx = build_combined_dataloaders(
        sl_audio, sl_index, ss_audio, ss_index, taxonomy_path, batch_size=4
    )

    assert {"a", "b", "c", "d", "e"} == set(label2idx.keys())


def test_combined_train_contains_both_sources(tmp_path):
    """Train loader has samples from both single-label and soundscape.

    Why: the whole point is mixing both data types in training.
    """
    sl_audio, sl_index, ss_audio, ss_index, taxonomy_path = _make_combined_fixture(tmp_path)
    train_loader, _, label2idx = build_combined_dataloaders(
        sl_audio, sl_index, ss_audio, ss_index, taxonomy_path, batch_size=4
    )

    # ConcatDataset has .datasets[0] (single-label) and .datasets[1] (soundscape)
    concat = train_loader.dataset
    assert len(concat.datasets) == 2
    assert len(concat.datasets[0]) > 0, "single-label portion is empty"
    assert len(concat.datasets[1]) > 0, "soundscape portion is empty"


def test_combined_val_loaders_are_separate(tmp_path):
    """Val loaders dict has both keys and each is non-empty.

    Why: we report metrics independently per data source.
    """
    sl_audio, sl_index, ss_audio, ss_index, taxonomy_path = _make_combined_fixture(tmp_path)
    _, val_loaders, _ = build_combined_dataloaders(
        sl_audio, sl_index, ss_audio, ss_index, taxonomy_path, batch_size=4
    )

    assert "single_label" in val_loaders
    assert "soundscape" in val_loaders
    assert len(val_loaders["single_label"].dataset) > 0
    assert len(val_loaders["soundscape"].dataset) > 0


def test_combined_soundscape_split_by_file(tmp_path):
    """No soundscape file appears in both train and val.

    Why: windows from the same recording are correlated — splitting by window
    would leak information across splits.
    """
    sl_audio, sl_index, ss_audio, ss_index, taxonomy_path = _make_combined_fixture(tmp_path)
    train_loader, val_loaders, _ = build_combined_dataloaders(
        sl_audio, sl_index, ss_audio, ss_index, taxonomy_path, batch_size=4
    )

    # Soundscape train is datasets[1] in the ConcatDataset
    ss_train_files = set(train_loader.dataset.datasets[1].dataset.index["filename"])
    ss_val_files = set(val_loaders["soundscape"].dataset.dataset.index["filename"])
    assert ss_train_files.isdisjoint(ss_val_files), (
        f"Soundscape files in both splits: {ss_train_files & ss_val_files}"
    )


def test_combined_soundscape_repeat(tmp_path):
    """Soundscape train size equals base windows × repeat factor.

    Why: repeat controls how much soundscape data the model sees per epoch.
    """
    sl_audio, sl_index, ss_audio, ss_index, taxonomy_path = _make_combined_fixture(tmp_path)

    repeat = 3
    train_loader, val_loaders, _ = build_combined_dataloaders(
        sl_audio, sl_index, ss_audio, ss_index, taxonomy_path, batch_size=4, soundscape_repeat=repeat
    )

    ss_train_len = len(train_loader.dataset.datasets[1])
    ss_val_len = len(val_loaders["soundscape"].dataset)
    # Total soundscape windows = 12, some go to val (1 file), rest to train
    ss_base_train = 12 - ss_val_len
    assert ss_train_len == ss_base_train * repeat


def test_combined_single_label_still_balanced(tmp_path):
    """Single-label portion of combined train is still class-balanced.

    Why: combining with soundscape data should not break single-label balancing.
    """
    sl_audio, sl_index, ss_audio, ss_index, taxonomy_path = _make_combined_fixture(tmp_path)
    train_loader, _, _ = build_combined_dataloaders(
        sl_audio, sl_index, ss_audio, ss_index, taxonomy_path, batch_size=4
    )

    sl_train_index = train_loader.dataset.datasets[0].dataset.index
    counts = sl_train_index["primary_label"].value_counts()
    assert counts.nunique() == 1, (
        f"Expected equal class counts, got: {counts.to_dict()}"
    )
