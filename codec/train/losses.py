from __future__ import annotations

from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def _as_float_signal(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(x, dtype=jnp.float32)


def _ensure_batch_time(x: jnp.ndarray) -> jnp.ndarray:
    x = _as_float_signal(x)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim == 3 and x.shape[-1] == 1:
        return x[..., 0]
    if x.ndim == 3 and x.shape[1] == 1:
        return x[:, 0, :]
    if x.ndim != 2:
        raise ValueError(f"Expected a (B,T) or compatible signal tensor, got {x.shape}")
    return x


def _hann_window(length: int, *, dtype: jnp.dtype) -> jnp.ndarray:
    if length <= 1:
        return jnp.ones((max(1, int(length)),), dtype=dtype)
    n = jnp.arange(length, dtype=dtype)
    return 0.5 - 0.5 * jnp.cos((2.0 * jnp.pi * n) / (length - 1))


def _frame_signal(x: jnp.ndarray, *, frame_length: int, hop_length: int) -> jnp.ndarray:
    total_length = int(x.shape[-1])
    if total_length < frame_length:
        x = jnp.pad(x, ((0, 0), (0, frame_length - total_length)))
        total_length = frame_length
    num_frames = 1 + max(0, (total_length - frame_length) // hop_length)
    frames = [
        jax.lax.dynamic_slice_in_dim(x, i * hop_length, frame_length, axis=-1)
        for i in range(num_frames)
    ]
    return jnp.stack(frames, axis=1)


def _frame_mask(valid_mask: jnp.ndarray | None, *, frame_length: int, hop_length: int) -> jnp.ndarray | None:
    if valid_mask is None:
        return None
    mask = _ensure_batch_time(valid_mask).astype(jnp.float32)
    frames = _frame_signal(mask, frame_length=frame_length, hop_length=hop_length)
    return jnp.mean(frames, axis=-1)


def _masked_mean_like(x: jnp.ndarray, weights: jnp.ndarray | None = None, *, eps: float = 1e-8) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=jnp.float32)
    if weights is None:
        return jnp.mean(x)
    w = jnp.asarray(weights, dtype=jnp.float32)
    while w.ndim < x.ndim:
        w = w[..., None]
    denom = jnp.maximum(jnp.sum(w) * (x.size / max(1, w.size)), eps)
    return jnp.sum(x * w) / denom


def _stft_complex(x: jnp.ndarray, *, n_fft: int, hop_length: int, win_length: int) -> jnp.ndarray:
    signal = _ensure_batch_time(x)
    fft_size = max(1, int(n_fft))
    frame_length = max(1, min(int(win_length), fft_size))
    hop = max(1, int(hop_length))
    window = _hann_window(frame_length, dtype=signal.dtype)
    frames = _frame_signal(signal, frame_length=frame_length, hop_length=hop)
    windowed = frames * window[None, None, :]
    return jnp.fft.rfft(windowed, n=fft_size, axis=-1)


def _stft_logmag(x: jnp.ndarray, *, n_fft: int, hop_length: int, win_length: int) -> jnp.ndarray:
    spec = _stft_complex(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return jnp.log1p(jnp.abs(spec))


def l1_time_loss(y: jnp.ndarray, y_hat: jnp.ndarray, *, valid_mask: jnp.ndarray | None = None) -> jnp.ndarray:
    y = _ensure_batch_time(y)
    y_hat = _ensure_batch_time(y_hat)
    return _masked_mean_like(jnp.abs(y - y_hat), valid_mask)


# Frequency bands (Hz) of the band losses: (low, high], the first band includes DC and the top
# band ends at Nyquist. Together they cover the whole spectrum once.
LOSS_BANDS = {
    "0_500hz": (0.0, 500.0),
    "500_1000hz": (500.0, 1000.0),
    "1000_2000hz": (1000.0, 2000.0),
    "2000_2500hz": (2000.0, 2500.0),
}


def _band(freqs: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    return ((freqs > low_hz) | (low_hz <= 0.0)) & (freqs <= high_hz)


def band_l1_loss(
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    *,
    low_hz: float,
    high_hz: float,
    sample_rate_hz: float,
    pa_half_range: jnp.ndarray,
    pa_scale: float = 50.0,
    valid_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Time-domain L1 of the error's components in the band (phase-sensitive: the waveform must
    line up), in pA / pa_scale: each chunk's error is scaled back by its half range."""
    y = _ensure_batch_time(y)
    y_hat = _ensure_batch_time(y_hat)
    n = int(y.shape[-1])
    keep = jnp.asarray(_band(np.fft.rfftfreq(n, d=1.0 / float(sample_rate_hz)), low_hz, high_hz), dtype=jnp.float32)
    diff = jnp.fft.irfft((jnp.fft.rfft(y, axis=-1) - jnp.fft.rfft(y_hat, axis=-1)) * keep[None, :], n=n, axis=-1)
    half_range = jnp.asarray(pa_half_range, dtype=jnp.float32).reshape((-1, 1))
    return _masked_mean_like(jnp.abs(diff) * half_range / float(pa_scale), valid_mask)


def band_stft_logmag_l1_loss(
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    *,
    low_hz: float,
    high_hz: float,
    sample_rate_hz: float,
    scales: Sequence[tuple[int, int, int]],
    valid_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """L1 of log(1 + |STFT|) over the bins in the band, averaged over the STFT scales
    (n_fft, win_length, hop_length). It matches how much energy each frequency has, not the phase."""
    losses = []
    for n_fft, win_length, hop_length in scales:
        freqs = np.fft.rfftfreq(int(n_fft), d=1.0 / float(sample_rate_hz))
        bins = np.nonzero(_band(freqs, low_hz, high_hz))[0]  # static, so the slice is fixed under jit
        diff = jnp.abs(
            _stft_logmag(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            - _stft_logmag(y_hat, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        )[..., bins]
        frame_weights = _frame_mask(valid_mask, frame_length=min(int(win_length), int(n_fft)), hop_length=int(hop_length))
        losses.append(_masked_mean_like(diff, frame_weights))
    return jnp.mean(jnp.stack(losses))


def compute_reconstruction_losses(
    *,
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    weights: Dict[str, float],
    stft_loss_scales: Sequence[tuple[int, int, int]],
    pa_half_range: jnp.ndarray,
    valid_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Weighted sum of the band losses and a log entry (raw and weighted) for each. Terms with
    zero weight are not computed; the normalized time-domain L1 is always logged as a metric."""
    sample_rate = float(weights.get("sample_rate", 5000.0))
    total = jnp.asarray(0.0, dtype=jnp.float32)
    logs = {"norm_l1_metric": l1_time_loss(y, y_hat, valid_mask=valid_mask)}
    for name, (low, high) in LOSS_BANDS.items():
        terms = {
            f"band_l1_{name}": lambda: band_l1_loss(
                y, y_hat, low_hz=low, high_hz=high, sample_rate_hz=sample_rate, pa_half_range=pa_half_range,
                pa_scale=weights.get("pa_l1_scale", 50.0), valid_mask=valid_mask,
            ),
            f"stft_logmag_{name}": lambda: band_stft_logmag_l1_loss(
                y, y_hat, low_hz=low, high_hz=high, sample_rate_hz=sample_rate, scales=stft_loss_scales,
                valid_mask=valid_mask,
            ),
        }
        for key, loss in terms.items():
            weight = float(weights.get(key, 0.0))
            if weight == 0.0:
                continue
            value = loss()
            total = total + weight * value
            logs[f"{key}_loss_raw"] = value
            logs[f"{key}_loss"] = weight * value
    logs["reconstruct_loss"] = total
    return total, logs
