import modal

from birdclef_2026.data.preparation.pipeline import run_pipeline

app = modal.App("birdclef-2026-memmap")

raw_volume = modal.Volume.from_name("birdclef-2026-raw")
processed_volume = modal.Volume.from_name("birdclef-2026-processed", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .apt_install("libsndfile1")
    .pip_install("soundfile", "numpy", "pandas", "pyarrow")
    .add_local_python_source("birdclef_2026")
)


@app.function(
    image=image,
    volumes={"/raw": raw_volume, "/processed": processed_volume},
    timeout=3600,
)
def build_dataset():
    run_pipeline("/raw/birdclef-2026.zip", "/processed/audio.npy", "/processed/index.parquet")
    processed_volume.commit()


@app.local_entrypoint()
def main():
    build_dataset.remote()
