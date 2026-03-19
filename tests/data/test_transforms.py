import torch
import pytest

from birdclef_2026.data.transforms import AmplitudeToDB, PerSampleMinMaxNorm


# ---------------------------------------------------------------------------
# AmplitudeToDB
# ---------------------------------------------------------------------------

def test_amplitude_to_db_output_in_range():
    """Output is always within [-top_db, 0] dB relative to the loudest frame.

    Why: the top_db clamp is the entire point of the transform — values below
    the floor corrupt the dynamic range contract that PerSampleMinMaxNorm
    depends on.
    """
    x = torch.rand(4, 128, 313).clamp(min=1e-9)
    out = AmplitudeToDB(top_db=80.0)(x)
    max_per_sample = out.amax(dim=(-2, -1))
    min_per_sample = out.amin(dim=(-2, -1))
    assert (max_per_sample <= 0.0).all()
    assert (min_per_sample >= -80.0).all()


def test_amplitude_to_db_louder_input_gives_higher_db():
    """A louder signal produces a higher dB value than a quieter one.

    Why: confirms the log10 scaling is in the right direction — inverted
    scaling would make louder audio appear quieter.
    """
    quiet = torch.full((1, 128, 313), 1e-4)
    loud = torch.full((1, 128, 313), 1.0)
    assert AmplitudeToDB()(loud).mean() > AmplitudeToDB()(quiet).mean()


# ---------------------------------------------------------------------------
# PerSampleMinMaxNorm
# ---------------------------------------------------------------------------

def test_per_sample_min_max_norm_output_in_unit_range():
    """Output values are always in [0, 1] for every sample in the batch.

    Why: values outside [0, 1] break the input contract for backbone weights
    loaded with in_chans=1 and would silently degrade training.
    """
    x = torch.randn(4, 128, 313) * 50 - 20  # arbitrary scale/offset
    out = PerSampleMinMaxNorm()(x)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_per_sample_min_max_norm_each_sample_uses_full_range():
    """Each sample individually spans the full [0, 1] range.

    Why: if normalisation were applied globally across the batch, quiet samples
    would be compressed to near-zero, losing all frequency detail.
    """
    x = torch.randn(4, 128, 313) * 50 - 20
    out = PerSampleMinMaxNorm()(x)
    for i in range(out.shape[0]):
        assert out[i].max().item() == pytest.approx(1.0, abs=1e-5)
        assert out[i].min().item() == pytest.approx(0.0, abs=1e-5)


def test_per_sample_min_max_norm_silent_clip_does_not_crash():
    """A flat (silent) clip returns a finite tensor without dividing by zero.

    Why: the eps term exists specifically for this case — without it a silent
    clip would produce NaN, which propagates silently through the model.
    """
    x = torch.zeros(1, 128, 313)
    out = PerSampleMinMaxNorm()(x)
    assert torch.isfinite(out).all()
