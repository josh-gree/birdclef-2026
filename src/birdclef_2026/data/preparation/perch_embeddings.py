import modal

app = modal.App("birdclef-2026-perch-embeddings")

processed_volume = modal.Volume.from_name("birdclef-2026-processed")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "pyarrow", "tensorflow", "kagglehub")
)

BATCH_SIZE = 64
EMBEDDING_DIM = 1280
MODEL_HANDLE = "google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier"


@app.function(
    image=image,
    volumes={"/processed": processed_volume},
    secrets=[modal.Secret.from_name("kaggle-secret")],
    timeout=3600,
    gpu="any",
)
def extract_perch_embeddings():
    import time

    import kagglehub
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    print("Downloading Perch model...")
    model_path = kagglehub.model_download(MODEL_HANDLE)
    model = tf.saved_model.load(model_path)
    print(f"Model loaded from {model_path}")

    audio = np.load("/processed/soundscape_audio.npy", mmap_mode="r")
    index = pd.read_parquet("/processed/soundscape_index.parquet")
    n_clips = len(index)
    print(f"Extracting embeddings for {n_clips} clips...")

    embeddings = np.zeros((n_clips, EMBEDDING_DIM), dtype=np.float32)

    t0 = time.perf_counter()
    for batch_start in range(0, n_clips, BATCH_SIZE):
        batch_rows = index.iloc[batch_start : batch_start + BATCH_SIZE]
        waveforms = np.stack([
            audio[row.offset_start : row.offset_end].astype(np.float32) / 32767.0
            for _, row in batch_rows.iterrows()
        ])
        outputs = model.infer_tf(tf.constant(waveforms))
        embeddings[batch_start : batch_start + len(batch_rows)] = outputs["embedding"].numpy()

        if (batch_start // BATCH_SIZE + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            done = batch_start + len(batch_rows)
            rate = done / elapsed
            eta = (n_clips - done) / rate
            print(f"  {done}/{n_clips} ({100*done/n_clips:.1f}%) — {rate:.1f} clips/s — ETA {eta:.0f}s")

    np.save("/processed/soundscape_embeddings.npy", embeddings)
    print(f"Saved embeddings shape {embeddings.shape} to /processed/soundscape_embeddings.npy")
    processed_volume.commit()


@app.function(
    image=image,
    volumes={"/processed": processed_volume},
)
def verify_perch_embeddings():
    import numpy as np
    import pandas as pd

    embeddings = np.load("/processed/soundscape_embeddings.npy", mmap_mode="r")
    index = pd.read_parquet("/processed/soundscape_index.parquet")

    n_clips = len(index)

    assert embeddings.dtype == np.float32, f"expected float32, got {embeddings.dtype}"
    assert embeddings.ndim == 2, f"expected 2D array, got shape {embeddings.shape}"
    assert embeddings.shape[0] == n_clips, (
        f"row count mismatch: {embeddings.shape[0]} embeddings vs {n_clips} index rows"
    )
    assert embeddings.shape[1] == EMBEDDING_DIM, (
        f"expected embedding dim {EMBEDDING_DIM}, got {embeddings.shape[1]}"
    )
    assert not np.isnan(embeddings).any(), "NaN values found in embeddings"
    assert not (embeddings == 0).all(axis=1).any(), "found all-zero embedding rows"

    # Check embeddings vary (not all identical)
    row_norms = np.linalg.norm(embeddings, axis=1)
    assert row_norms.std() > 0, "all embeddings are identical — something went wrong"

    print(f"soundscape_embeddings.npy: shape={embeddings.shape}, dtype={embeddings.dtype}")
    print(f"row norms — min={row_norms.min():.3f}  mean={row_norms.mean():.3f}  max={row_norms.max():.3f}")
    print(f"index rows: {n_clips}")
    print("All checks passed.")


@app.local_entrypoint()
def main():
    extract_perch_embeddings.remote()
