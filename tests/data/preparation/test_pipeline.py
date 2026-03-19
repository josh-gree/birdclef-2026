import io
import zipfile

import numpy as np
import pandas as pd
import soundfile as sf

from birdclef_2026.data.preparation.pipeline import (
    build_index,
    decode_to_int16,
    filter_by_min_duration,
    get_num_frames,
    run_pipeline,
    write_audio_memmap,
)
from tests.conftest import make_ogg_bytes


def test_get_num_frames_returns_correct_count():
    """get_num_frames reads frame count from the file header without decoding.

    We allow a small tolerance because OGG/Vorbis encoders may pad the stream
    by a few samples, so the reported frame count can differ slightly from the
    number of samples originally passed to the encoder.
    """
    n_samples = 32000
    ogg = make_ogg_bytes(n_samples=n_samples)
    frames = get_num_frames(ogg)
    assert abs(frames - n_samples) <= 100


def test_decode_to_int16_returns_int16_array():
    """decode_to_int16 produces a 1-D int16 array of the expected length.

    The dtype and shape are checked because downstream code allocates the
    memmap as int16 and indexes into it by sample count — a wrong dtype or
    unexpected length would silently corrupt the dataset.
    """
    n_samples = 32000
    ogg = make_ogg_bytes(n_samples=n_samples)
    arr = decode_to_int16(ogg)
    assert arr.dtype == np.int16
    assert abs(len(arr) - n_samples) <= 100


def test_decode_to_int16_averages_stereo_to_mono():
    """decode_to_int16 collapses multi-channel audio to a single mono channel.

    The competition audio is mono, but the function must handle stereo files
    robustly. A 2-channel input should produce a 1-D output array.
    """
    n_samples = 16000
    sample_rate = 32000
    audio = np.random.uniform(-0.5, 0.5, (n_samples, 2)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="OGG", subtype="VORBIS")
    ogg = buf.getvalue()

    arr = decode_to_int16(ogg)
    assert arr.ndim == 1
    assert abs(len(arr) - n_samples) <= 100


def test_filter_by_min_duration_drops_short_files():
    """filter_by_min_duration removes files below the frame threshold.

    Short clips are unsuitable for training 5-second windows, so they must be
    excluded before memmap allocation. Both the filename and frame count lists
    must stay in sync after filtering.
    """
    filenames = ["a.ogg", "b.ogg", "c.ogg"]
    frame_counts = [1000, 5000, 3000]
    fns, fcs = filter_by_min_duration(filenames, frame_counts, min_frames=2000)
    assert fns == ["b.ogg", "c.ogg"]
    assert fcs == [5000, 3000]


def test_filter_by_min_duration_keeps_all_above_threshold():
    """filter_by_min_duration does not drop files that meet the threshold.

    Ensures the function is not overly aggressive — files at or above the
    minimum must pass through unchanged.
    """
    filenames = ["a.ogg", "b.ogg"]
    frame_counts = [5000, 6000]
    fns, fcs = filter_by_min_duration(filenames, frame_counts, min_frames=1000)
    assert fns == ["a.ogg", "b.ogg"]
    assert fcs == [5000, 6000]


def test_filter_by_min_duration_empty_result():
    """filter_by_min_duration returns empty lists when no files pass the threshold.

    The caller must handle this gracefully rather than receiving a zip/unzip
    error from an empty sequence.
    """
    filenames = ["a.ogg"]
    frame_counts = [100]
    fns, fcs = filter_by_min_duration(filenames, frame_counts, min_frames=9999)
    assert fns == []
    assert fcs == []


def test_write_audio_memmap_writes_contiguously_and_returns_offsets():
    """write_audio_memmap packs arrays back-to-back and returns correct offsets.

    The memmap must be filled without gaps so that slicing by [offset_start:
    offset_end] later recovers exactly the right samples for each file.
    """
    arrays = [np.array([1, 2, 3], dtype=np.int16), np.array([4, 5], dtype=np.int16)]
    total = sum(len(a) for a in arrays)
    mm = np.zeros(total, dtype=np.int16)

    offsets = write_audio_memmap(mm, iter(arrays))

    assert offsets == [0, 3, 5]
    np.testing.assert_array_equal(mm, [1, 2, 3, 4, 5])


def test_write_audio_memmap_empty_iter():
    """write_audio_memmap handles an empty iterator without error.

    A dataset that has been fully filtered out should not cause a crash during
    the write pass — the function should return the sentinel [0] offset list.
    """
    mm = np.zeros(0, dtype=np.int16)
    offsets = write_audio_memmap(mm, iter([]))
    assert offsets == [0]


def test_build_index_merges_metadata():
    """build_index attaches per-file metadata to the offset DataFrame.

    The index is the primary lookup table used by the dataset loader. Offsets
    and metadata must be aligned correctly so that slicing audio by
    offset_start/offset_end retrieves the file described by the same row.
    """
    filenames = ["a.ogg", "b.ogg"]
    offsets = [0, 100, 250]
    meta = pd.DataFrame({"filename": ["a.ogg", "b.ogg"], "label": ["cat", "dog"]})

    index = build_index(filenames, offsets, meta)

    assert list(index["filename"]) == ["a.ogg", "b.ogg"]
    assert list(index["offset_start"]) == [0, 100]
    assert list(index["offset_end"]) == [100, 250]
    assert list(index["label"]) == ["cat", "dog"]
    assert index["offset_end"].iloc[-1] == 250


def test_run_pipeline_end_to_end(tmp_path):
    """run_pipeline produces a valid memmap and index from a synthetic zip.

    This is the closest we can get to testing the full Modal job locally.
    We verify that both output files are written, the index has the right
    shape and columns, and that slicing the memmap by the index offsets
    recovers exactly the original decoded audio.
    """
    ogg1 = make_ogg_bytes(n_samples=32000)
    ogg2 = make_ogg_bytes(n_samples=64000)

    zip_path = str(tmp_path / "dataset.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        meta = "filename,primary_label\na.ogg,cat\nb.ogg,dog\n"
        zf.writestr("train.csv", meta)
        zf.writestr("train_audio/a.ogg", ogg1)
        zf.writestr("train_audio/b.ogg", ogg2)

    audio_out = str(tmp_path / "audio.npy")
    index_out = str(tmp_path / "index.parquet")

    run_pipeline(zip_path, audio_out, index_out, min_duration_s=0.0)

    audio = np.load(audio_out, mmap_mode="r")
    index = pd.read_parquet(index_out)

    assert len(index) == 2
    assert list(index["filename"]) == ["a.ogg", "b.ogg"]
    assert index["offset_end"].iloc[-1] == len(audio[: index["offset_end"].iloc[-1]])
    assert set(index.columns) >= {"filename", "offset_start", "offset_end", "primary_label"}

    arr1 = decode_to_int16(ogg1)
    arr2 = decode_to_int16(ogg2)
    np.testing.assert_array_equal(audio[index["offset_start"].iloc[0] : index["offset_end"].iloc[0]], arr1)
    np.testing.assert_array_equal(audio[index["offset_start"].iloc[1] : index["offset_end"].iloc[1]], arr2)


def test_run_pipeline_filters_short_files(tmp_path):
    """run_pipeline excludes files below min_duration_s from the outputs.

    Verifies that the filtering step is wired up correctly inside the full
    pipeline, not just in the isolated filter_by_min_duration unit test.
    """
    ogg_short = make_ogg_bytes(n_samples=16000)   # 0.5s
    ogg_long = make_ogg_bytes(n_samples=160000)   # 5s

    zip_path = str(tmp_path / "dataset.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        meta = "filename,primary_label\nshort.ogg,cat\nlong.ogg,dog\n"
        zf.writestr("train.csv", meta)
        zf.writestr("train_audio/short.ogg", ogg_short)
        zf.writestr("train_audio/long.ogg", ogg_long)

    run_pipeline(
        zip_path,
        str(tmp_path / "audio.npy"),
        str(tmp_path / "index.parquet"),
        min_duration_s=1.0,
    )

    index = pd.read_parquet(tmp_path / "index.parquet")
    assert len(index) == 1
    assert index["filename"].iloc[0] == "long.ogg"


def test_round_trip_decode_write_slice():
    """Decoding OGGs and writing to a memmap preserves sample values exactly.

    Confirms that the encode → decode → write → slice chain is lossless at
    the int16 level, so that audio retrieved from the memmap during training
    is bit-for-bit identical to what would be decoded directly from the file.
    """
    ogg1 = make_ogg_bytes(n_samples=32000)
    ogg2 = make_ogg_bytes(n_samples=16000)

    arr1 = decode_to_int16(ogg1)
    arr2 = decode_to_int16(ogg2)

    total = len(arr1) + len(arr2)
    mm = np.zeros(total, dtype=np.int16)
    offsets = write_audio_memmap(mm, iter([arr1, arr2]))

    assert offsets[-1] == total

    slice1 = mm[offsets[0] : offsets[1]]
    slice2 = mm[offsets[1] : offsets[2]]

    np.testing.assert_array_equal(slice1, arr1)
    np.testing.assert_array_equal(slice2, arr2)
