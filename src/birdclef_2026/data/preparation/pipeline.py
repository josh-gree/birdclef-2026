import io
from typing import Iterable

import numpy as np
import pandas as pd
import soundfile as sf


def get_num_frames(audio_bytes: bytes) -> int:
    """Return the number of samples in an audio file without decoding it.

    Parameters
    ----------
    audio_bytes : bytes
        Raw bytes of an audio file (e.g. OGG/Vorbis).

    Returns
    -------
    int
        Number of frames (samples) reported by the file header.
    """
    return sf.info(io.BytesIO(audio_bytes)).frames


def decode_to_int16(audio_bytes: bytes) -> np.ndarray:
    """Decode an audio file to a mono int16 numpy array.

    Reads as float32 internally and scales to [-32767, 32767].
    Multi-channel audio is averaged down to mono before conversion.

    Parameters
    ----------
    audio_bytes : bytes
        Raw bytes of an audio file (e.g. OGG/Vorbis).

    Returns
    -------
    np.ndarray
        1-D array of dtype int16.
    """
    audio, _ = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return (audio * 32767).astype(np.int16)


def filter_by_min_duration(
    filenames: list[str], frame_counts: list[int], min_frames: int
) -> tuple[list[str], list[int]]:
    """Drop files whose frame count is below a minimum threshold.

    Parameters
    ----------
    filenames : list[str]
        File paths corresponding to each entry in ``frame_counts``.
    frame_counts : list[int]
        Number of frames for each file.
    min_frames : int
        Files with fewer than this many frames are dropped.

    Returns
    -------
    tuple[list[str], list[int]]
        Filtered ``(filenames, frame_counts)`` containing only files that
        meet or exceed ``min_frames``.
    """
    kept = [(fn, fc) for fn, fc in zip(filenames, frame_counts) if fc >= min_frames]
    if not kept:
        return [], []
    fns, fcs = zip(*kept)
    return list(fns), list(fcs)


def write_audio_memmap(
    mm: np.ndarray,
    audio_iter: Iterable[np.ndarray],
) -> list[int]:
    """Write decoded int16 arrays contiguously into a pre-allocated memmap.

    Parameters
    ----------
    mm : np.ndarray
        Pre-allocated output array (typically a memmap of dtype int16).
    audio_iter : Iterable[np.ndarray]
        Sequence of 1-D int16 arrays to write, one per file.

    Returns
    -------
    list[int]
        Offsets list of length ``n + 1`` where ``n`` is the number of arrays
        written. ``offsets[i]`` is the start sample of file ``i`` and
        ``offsets[-1]`` is the total number of samples written.
    """
    actual_offsets = [0]
    pos = 0
    for arr in audio_iter:
        mm[pos : pos + len(arr)] = arr
        pos += len(arr)
        actual_offsets.append(pos)
    return actual_offsets


def build_index(
    filenames: list[str],
    actual_offsets: list[int],
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Construct an index DataFrame from filenames, offsets, and metadata.

    Parameters
    ----------
    filenames : list[str]
        Ordered list of filenames matching the offsets.
    actual_offsets : list[int]
        Offsets list of length ``len(filenames) + 1`` as returned by
        :func:`write_audio_memmap`.
    metadata_df : pd.DataFrame
        Source metadata containing at least a ``filename`` column to join on.

    Returns
    -------
    pd.DataFrame
        One row per file with columns ``filename``, ``offset_start``,
        ``offset_end``, plus all columns from ``metadata_df``.
    """
    offsets = np.array(actual_offsets, dtype=np.int64)
    index = pd.DataFrame({
        "filename": filenames,
        "offset_start": offsets[:-1],
        "offset_end": offsets[1:],
    })
    return index.merge(metadata_df, on="filename", how="left")


def run_pipeline(
    zip_path: str,
    audio_out: str,
    index_out: str,
    min_duration_s: float = 5.0,
    sample_rate: int = 32000,
) -> None:
    """Run the full dataset preparation pipeline from a zip file to memmap + index.

    Performs two passes over the zip:

    1. Count frames cheaply (no decoding) to allocate the memmap.
    2. Decode each file and write directly into the memmap.

    Short files (below ``min_duration_s``) are dropped before allocation.

    Parameters
    ----------
    zip_path : str
        Path to the competition zip file containing ``train.csv`` and
        ``train_audio/<filename>`` entries.
    audio_out : str
        Output path for the ``audio.npy`` memmap file.
    index_out : str
        Output path for the ``index.parquet`` file.
    min_duration_s : float, optional
        Minimum clip duration in seconds. Files shorter than this are dropped.
        Defaults to 5.0.
    sample_rate : int, optional
        Sample rate used to convert ``min_duration_s`` to frames.
        Defaults to 32000.
    """
    import time
    import zipfile

    min_frames = int(min_duration_s * sample_rate)

    with zipfile.ZipFile(zip_path) as zf:
        df = pd.read_csv(io.BytesIO(zf.read("train.csv")))
        filenames = df["filename"].tolist()
        n_files = len(filenames)

        print("Pass 1: counting frames...")
        t0 = time.perf_counter()
        frame_counts = [get_num_frames(zf.read(f"train_audio/{fn}")) for fn in filenames]
        t_pass1 = time.perf_counter() - t0
        print(f"  done in {t_pass1:.1f}s ({t_pass1 / n_files * 1000:.1f}ms/file)")

        filenames, frame_counts = filter_by_min_duration(filenames, frame_counts, min_frames)
        n_kept = len(filenames)
        print(f"Dropped {n_files - n_kept} files under {min_duration_s}s, keeping {n_kept}")

        estimated_total = sum(frame_counts)
        alloc_total = estimated_total + n_kept * 10
        print(f"Estimated samples: {estimated_total} ({estimated_total * 2 / 1e9:.2f} GB)")

        mm = np.lib.format.open_memmap(audio_out, mode="w+", dtype=np.int16, shape=(alloc_total,))

        print("Pass 2: decoding and writing...")
        t0 = time.perf_counter()

        def audio_iter():
            for i, fn in enumerate(filenames):
                arr = decode_to_int16(zf.read(f"train_audio/{fn}"))
                if (i + 1) % 100 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = (i + 1) / elapsed
                    pct = (i + 1) / n_kept * 100
                    eta = (n_kept - (i + 1)) / rate
                    print(f"  {i + 1}/{n_kept} ({pct:.1f}%) — {rate:.1f} files/s — ETA {eta:.0f}s")
                yield arr

        actual_offsets = write_audio_memmap(mm, audio_iter())
        t_pass2 = time.perf_counter() - t0
        print(f"  done in {t_pass2:.1f}s ({t_pass2 / n_kept * 1000:.1f}ms/file)")
        mm.flush()

    index = build_index(filenames, actual_offsets, df)
    index.to_parquet(index_out, index=False)
    print("Done.")
