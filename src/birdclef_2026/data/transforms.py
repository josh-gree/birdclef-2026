import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio.features import MelSpectrogram as _MelSpectrogram


class MelSpectrogram(nn.Module):
    """Compute a mel spectrogram from a batch of waveforms using nnAudio.

    Weights are frozen by default. Set ``trainable=True`` to allow the
    filterbank and STFT to be learned during training.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the input waveforms in Hz.
    n_fft : int
        FFT size.
    hop_length : int
        Hop length between STFT frames.
    n_mels : int
        Number of mel filterbank bins.
    fmin : float
        Lowest frequency of the mel filterbank in Hz.
    fmax : float
        Highest frequency of the mel filterbank in Hz.
    trainable : bool
        If ``True``, filterbank and STFT weights are learnable parameters.
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 20.0,
        fmax: float = 16000.0,
        trainable: bool = False,
    ):
        super().__init__()
        self.mel = _MelSpectrogram(
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            trainable_mel=trainable,
            trainable_STFT=trainable,
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveforms : torch.Tensor
            Shape ``(batch, samples)``, float32 in ``[-1, 1]``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_mels, time)``, float32 power spectrogram.
        """
        return self.mel(waveforms)


class AmplitudeToDB(nn.Module):
    """Convert a power spectrogram to decibel scale and clamp the dynamic range.

    Applies ``10 * log10(x)``, then floors values at ``max - top_db`` so the
    output is always in ``[-top_db, 0]`` dB relative to the loudest frame.

    Parameters
    ----------
    top_db : float
        Dynamic range to retain in dB. Values below ``max - top_db`` are
        clamped to ``max - top_db``.
    """

    def __init__(self, top_db: float = 80.0):
        super().__init__()
        self.top_db = top_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_mels, time)``, float32 power spectrogram.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_mels, time)``, float32 in ``[-top_db, 0]`` dB.
        """
        x_db = 10.0 * torch.log10(x.clamp(min=1e-9))
        max_db = x_db.amax(dim=(-2, -1), keepdim=True)
        return x_db.clamp(min=max_db - self.top_db)


class PerSampleMinMaxNorm(nn.Module):
    """Normalise each spectrogram independently to ``[0, 1]``.

    Applies per-sample min-max scaling across the frequency and time axes so
    that every spectrogram uses the full ``[0, 1]`` range regardless of
    recording volume. This is required to produce valid input for pretrained
    backbones loaded with ``in_chans=1``.

    Parameters
    ----------
    eps : float
        Small constant added to the denominator to avoid division by zero for
        silent clips.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_mels, time)``, float32.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_mels, time)``, float32 in ``[0, 1]``.
        """
        min_ = x.amin(dim=(-2, -1), keepdim=True)
        max_ = x.amax(dim=(-2, -1), keepdim=True)
        return (x - min_) / (max_ - min_ + self.eps)


class Resize(nn.Module):
    """Resize a ``(batch, n_mels, time)`` spectrogram to a fixed spatial size.

    Adds a channel dimension before interpolating and retains it, producing
    ``(batch, 1, height, width)`` output suitable for backbones with
    ``in_chans=1``.

    Parameters
    ----------
    height : int
        Target height in pixels.
    width : int
        Target width in pixels.
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        self.size = (height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_mels, time)``, float32.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1, height, width)``, float32.
        """
        x = x.unsqueeze(1)  # (batch, 1, n_mels, time)
        return F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)


class FrequencyMask(nn.Module):
    """Mask one or more contiguous frequency bands in a spectrogram.

    Applied only during training (no-op at eval time). Each mask stripe is
    applied independently with probability ``p``.

    Parameters
    ----------
    max_mask_size : int
        Maximum number of frequency bins to mask per mask stripe.
    num_masks : int
        Number of independent frequency masks to attempt.
    p : float
        Probability of applying each individual mask stripe.
    fill_value : float
        Value to fill masked regions with.
    """

    def __init__(
        self,
        max_mask_size: int,
        num_masks: int = 1,
        p: float = 0.5,
        fill_value: float = 0.0,
    ):
        super().__init__()
        self.max_mask_size = max_mask_size
        self.num_masks = num_masks
        self.p = p
        self.fill_value = fill_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(..., freq, time)``, float32.

        Returns
        -------
        torch.Tensor
            Same shape as input with frequency bands zeroed out.
        """
        if not self.training:
            return x
        x = x.clone()
        n_freq = x.shape[-2]
        for _ in range(self.num_masks):
            if torch.rand(()).item() > self.p:
                continue
            mask_size = torch.randint(1, self.max_mask_size + 1, ()).item()
            start = torch.randint(0, max(1, n_freq - mask_size + 1), ()).item()
            x[..., start : start + mask_size, :] = self.fill_value
        return x


class TimeMask(nn.Module):
    """Mask one or more contiguous time bands in a spectrogram.

    Applied only during training (no-op at eval time). Each mask stripe is
    applied independently with probability ``p``.

    Parameters
    ----------
    max_mask_size : int
        Maximum number of time frames to mask per mask stripe.
    num_masks : int
        Number of independent time masks to attempt.
    p : float
        Probability of applying each individual mask stripe.
    fill_value : float
        Value to fill masked regions with.
    """

    def __init__(
        self,
        max_mask_size: int,
        num_masks: int = 1,
        p: float = 0.5,
        fill_value: float = 0.0,
    ):
        super().__init__()
        self.max_mask_size = max_mask_size
        self.num_masks = num_masks
        self.p = p
        self.fill_value = fill_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(..., freq, time)``, float32.

        Returns
        -------
        torch.Tensor
            Same shape as input with time bands zeroed out.
        """
        if not self.training:
            return x
        x = x.clone()
        n_time = x.shape[-1]
        for _ in range(self.num_masks):
            if torch.rand(()).item() > self.p:
                continue
            mask_size = torch.randint(1, self.max_mask_size + 1, ()).item()
            start = torch.randint(0, max(1, n_time - mask_size + 1), ()).item()
            x[..., start : start + mask_size] = self.fill_value
        return x


class RectangleMask(nn.Module):
    """Apply one or more random rectangular masks to a spectrogram.

    Each mask is independently sized, positioned, and gated by probability
    ``p``. Applied only during training (no-op at eval time).

    Parameters
    ----------
    max_freq_size : int
        Maximum height (frequency bins) of each rectangle.
    max_time_size : int
        Maximum width (time frames) of each rectangle.
    num_masks : int
        Number of independent rectangles to attempt.
    p : float
        Probability of applying each individual rectangle.
    fill_value : float
        Value to fill masked regions with.
    """

    def __init__(
        self,
        max_freq_size: int,
        max_time_size: int,
        num_masks: int = 1,
        p: float = 0.5,
        fill_value: float = 0.0,
    ):
        super().__init__()
        self.max_freq_size = max_freq_size
        self.max_time_size = max_time_size
        self.num_masks = num_masks
        self.p = p
        self.fill_value = fill_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(..., freq, time)``, float32.

        Returns
        -------
        torch.Tensor
            Same shape as input with rectangular regions zeroed out.
        """
        if not self.training:
            return x
        x = x.clone()
        n_freq, n_time = x.shape[-2], x.shape[-1]
        for _ in range(self.num_masks):
            if torch.rand(()).item() > self.p:
                continue
            fh = torch.randint(1, self.max_freq_size + 1, ()).item()
            fw = torch.randint(1, self.max_time_size + 1, ()).item()
            f0 = torch.randint(0, max(1, n_freq - fh + 1), ()).item()
            t0 = torch.randint(0, max(1, n_time - fw + 1), ()).item()
            x[..., f0 : f0 + fh, t0 : t0 + fw] = self.fill_value
        return x


class GaussianNoise(nn.Module):
    """Add random Gaussian noise to a spectrogram.

    Applied only during training (no-op at eval time).

    Parameters
    ----------
    min_std : float
        Minimum standard deviation of the noise.
    max_std : float
        Maximum standard deviation of the noise. The actual std is sampled
        uniformly from ``[min_std, max_std]`` each forward pass.
    p : float
        Probability of applying the noise.
    """

    def __init__(self, min_std: float = 0.0, max_std: float = 0.05, p: float = 0.5):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(..., freq, time)``, float32.

        Returns
        -------
        torch.Tensor
            Same shape as input with Gaussian noise added.
        """
        if not self.training or torch.rand(()).item() > self.p:
            return x
        std = self.min_std + torch.rand(()).item() * (self.max_std - self.min_std)
        return x + torch.randn_like(x) * std


def build_spectrogram_pipeline(
    sample_rate: int = 32000,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 20.0,
    fmax: float = 16000.0,
    top_db: float = 80.0,
    height: int = 256,
    width: int = 256,
) -> nn.Sequential:
    """Build the standard waveform-to-image pipeline for training.

    Converts a batch of waveforms to normalised single-channel spectrograms
    ready for a pretrained backbone loaded with ``in_chans=1``. The pipeline
    follows the approach used by top BirdCLEF solutions: dB-scale mel
    spectrogram followed by per-sample min-max normalisation to ``[0, 1]``.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the input waveforms in Hz.
    n_fft : int
        FFT size.
    hop_length : int
        Hop length between STFT frames.
    n_mels : int
        Number of mel filterbank bins.
    fmin : float
        Lowest frequency of the mel filterbank in Hz.
    fmax : float
        Highest frequency of the mel filterbank in Hz.
    top_db : float
        Dynamic range retained by ``AmplitudeToDB``.
    height : int
        Target spectrogram image height.
    width : int
        Target spectrogram image width.

    Returns
    -------
    nn.Sequential
        ``waveforms (batch, samples) -> images (batch, 1, height, width)``
    """
    return nn.Sequential(
        MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        ),
        AmplitudeToDB(top_db=top_db),
        PerSampleMinMaxNorm(),
        Resize(height, width),
    )
