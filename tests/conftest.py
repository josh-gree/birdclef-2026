import io

import numpy as np
import soundfile as sf


def make_ogg_bytes(n_samples=32000, sample_rate=32000) -> bytes:
    audio = np.random.uniform(-0.5, 0.5, n_samples).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="OGG", subtype="VORBIS")
    return buf.getvalue()
