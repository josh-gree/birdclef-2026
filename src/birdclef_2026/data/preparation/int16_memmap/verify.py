import zipfile

import modal
import numpy as np
import pandas as pd

from birdclef_2026.data.preparation.int16_memmap.utils import decode_to_int16

app = modal.App("birdclef-2026-verify")

raw_volume = modal.Volume.from_name("birdclef-2026-raw")
processed_volume = modal.Volume.from_name("birdclef-2026-processed")

image = (
    modal.Image.debian_slim()
    .apt_install("libsndfile1")
    .pip_install("numpy", "pandas", "pyarrow", "soundfile")
    .add_local_python_source("birdclef_2026")
)


@app.function(
    image=image,
    volumes={"/raw": raw_volume, "/processed": processed_volume},
)
def verify():
    audio = np.load("/processed/audio.npy", mmap_mode="r")
    index = pd.read_parquet("/processed/index.parquet")

    durations_s = (index["offset_end"] - index["offset_start"]) / 32000

    assert audio.dtype == np.int16, f"expected int16, got {audio.dtype}"
    assert {"filename", "offset_start", "offset_end"}.issubset(index.columns), "missing index columns"
    assert (durations_s >= 5.0).all(), f"clips shorter than 5s present (min={durations_s.min():.2f}s)"
    assert index["offset_end"].iloc[-1] == len(audio), "last offset does not match audio length"
    assert index["offset_start"].iloc[0] == 0, "first offset is not 0"
    assert (index["offset_end"].values > index["offset_start"].values).all(), "zero-length clips present"

    print(f"audio.npy: shape={audio.shape}, dtype={audio.dtype}")
    print(f"index.parquet: {len(index)} rows, columns={list(index.columns)}")
    print(f"duration — min={durations_s.min():.1f}s  median={durations_s.median():.1f}s  max={durations_s.max():.1f}s")

    # Spot-check: decode a few OGGs from the raw zip and compare against memmap slices
    rng = np.random.default_rng(42)
    check_indices = [0, len(index) // 2, len(index) - 1] + list(rng.integers(0, len(index), size=3))

    print("Spot-checking memmap against raw OGG files...")
    with zipfile.ZipFile("/raw/birdclef-2026.zip") as zf:
        for i in check_indices:
            row = index.iloc[i]
            reference = decode_to_int16(zf.read(f"train_audio/{row.filename}"))
            memmap_clip = np.array(audio[row.offset_start : row.offset_end])
            assert np.array_equal(memmap_clip, reference), (
                f"[{i}] {row.filename}: memmap mismatch "
                f"(memmap len={len(memmap_clip)}, ref len={len(reference)})"
            )
            print(f"  [{i}] {row.filename}: OK")

    print("All checks passed.")


@app.local_entrypoint()
def main():
    verify.remote()
