# birdclef-2026

## Data preparation

Data preparation runs on Modal and produces two files on the `birdclef-2026-processed` volume:

- `audio.npy` — all training audio concatenated as a flat int16 memmap
- `index.parquet` — per-file offsets and metadata for slicing the memmap

### Prerequisites

- A Kaggle API token stored as a Modal secret named `kaggle-secret`
- Modal installed and authenticated (`modal setup`)

### Steps

**1. Download the raw data**

Runs a Kaggle CLI download on Modal and saves the competition zip to the `birdclef-2026-raw` volume.

```bash
modal run src/birdclef_2026/data/preparation/raw.py
```

**2. Build the processed dataset**

Reads the zip, decodes all audio to int16, writes the memmap and index, then commits the `birdclef-2026-processed` volume.

```bash
modal run src/birdclef_2026/data/preparation/int16_memmap/job.py
```

**3. Verify the processed dataset**

Runs structural checks and spot-checks a sample of audio slices against the original OGG files.

```bash
modal run src/birdclef_2026/data/preparation/int16_memmap/verify.py
```
