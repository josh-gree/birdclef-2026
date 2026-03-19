import modal

app = modal.App("birdclef-2026-raw")

volume = modal.Volume.from_name("birdclef-2026-raw", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("kaggle")
)


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("kaggle-secret")],
    timeout=3600,
)
def download_raw_data():
    import subprocess

    subprocess.run(
        ["kaggle", "competitions", "download", "-c", "birdclef-2026", "-p", "/data"],
        check=True,
    )
    volume.commit()


@app.local_entrypoint()
def main():
    download_raw_data.remote()
