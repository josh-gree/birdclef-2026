import io
import time
import zipfile

import numpy as np
import pandas as pd

from birdclef_2026.data.preparation.int16_memmap.utils import (
    decode_to_int16,
    get_num_frames,
    write_audio_memmap,
)

SAMPLE_RATE = 32000


def _parse_time(t: str) -> int:
    """Convert HH:MM:SS to seconds."""
    parts = t.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def run_soundscape_pipeline(
    zip_path: str,
    audio_out: str,
    index_out: str,
) -> None:
    """Build memmap + index for soundscape data.

    Writes each full soundscape ogg contiguously into the memmap, then
    creates an index row per 5-second window using offsets derived from
    the start/end times in train_soundscapes_labels.csv.
    """
    with zipfile.ZipFile(zip_path) as zf:
        labels_df = pd.read_csv(io.BytesIO(zf.read("train_soundscapes_labels.csv")))
        labels_df = labels_df.drop_duplicates(subset=["filename", "start", "end"])
        filenames = labels_df["filename"].unique().tolist()
        n_files = len(filenames)

        print(f"Soundscapes: {n_files} files, {len(labels_df)} windows")

        # Pass 1: count frames
        print("Pass 1: counting frames...")
        t0 = time.perf_counter()
        frame_counts = [
            get_num_frames(zf.read(f"train_soundscapes/{fn}")) for fn in filenames
        ]
        t_pass1 = time.perf_counter() - t0
        print(f"  done in {t_pass1:.1f}s ({t_pass1 / n_files * 1000:.1f}ms/file)")

        estimated_total = sum(frame_counts)
        alloc_total = estimated_total + n_files * 10
        print(f"Estimated samples: {estimated_total} ({estimated_total * 2 / 1e9:.2f} GB)")

        mm = np.lib.format.open_memmap(
            audio_out, mode="w+", dtype=np.int16, shape=(alloc_total,)
        )

        # Pass 2: decode and write full files
        print("Pass 2: decoding and writing...")
        t0 = time.perf_counter()

        def audio_iter():
            for i, fn in enumerate(filenames):
                arr = decode_to_int16(zf.read(f"train_soundscapes/{fn}"))
                if (i + 1) % 10 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = (i + 1) / elapsed
                    pct = (i + 1) / n_files * 100
                    eta = (n_files - (i + 1)) / rate
                    print(f"  {i + 1}/{n_files} ({pct:.1f}%) — {rate:.1f} files/s — ETA {eta:.0f}s")
                yield arr

        actual_offsets = write_audio_memmap(mm, audio_iter())
        t_pass2 = time.perf_counter() - t0
        print(f"  done in {t_pass2:.1f}s ({t_pass2 / n_files * 1000:.1f}ms/file)")
        mm.flush()

    # Build per-file offset lookup
    file_offsets = {fn: actual_offsets[i] for i, fn in enumerate(filenames)}

    # Build index: one row per 5-second window
    rows = []
    for _, row in labels_df.iterrows():
        file_start = file_offsets[row["filename"]]
        window_start = _parse_time(row["start"]) * SAMPLE_RATE
        window_end = _parse_time(row["end"]) * SAMPLE_RATE
        rows.append({
            "filename": row["filename"],
            "offset_start": file_start + window_start,
            "offset_end": file_start + window_end,
            "primary_label": row["primary_label"],
        })

    index = pd.DataFrame(rows)
    index.to_parquet(index_out, index=False)
    print(f"Wrote {len(index)} window rows to {index_out}")
    print("Done.")
