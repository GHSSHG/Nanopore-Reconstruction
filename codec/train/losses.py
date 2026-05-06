from __future__ import annotations

from typing import Any, Dict, Sequence

import jax
import jax.numpy as jnp


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


def _finite_difference(x: jnp.ndarray, *, order: int) -> jnp.ndarray:
    if order < 0:
        raise ValueError(f"Difference order must be >= 0, got {order}")
    diff = x
    for _ in range(order):
        if int(diff.shape[-1]) <= 1:
            return jnp.zeros(diff.shape[:-1] + (1,), dtype=diff.dtype)
        diff = diff[..., 1:] - diff[..., :-1]
    return diff


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


def _stft_complex(
    x: jnp.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    norm: str | None = None,
) -> jnp.ndarray:
    signal = _ensure_batch_time(x)
    fft_size = max(1, int(n_fft))
    frame_length = max(1, min(int(win_length), fft_size))
    hop = max(1, int(hop_length))
    window = _hann_window(frame_length, dtype=signal.dtype)
    frames = _frame_signal(signal, frame_length=frame_length, hop_length=hop)
    windowed = frames * window[None, None, :]
    spec = jnp.fft.rfft(windowed, n=fft_size, axis=-1, norm=norm)
    return spec


def _stft_logmag(x: jnp.ndarray, *, n_fft: int, hop_length: int, win_length: int) -> jnp.ndarray:
    spec = _stft_complex(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return jnp.log1p(jnp.abs(spec))


def l1_time_loss(y: jnp.ndarray, y_hat: jnp.ndarray, *, valid_mask: jnp.ndarray | None = None) -> jnp.ndarray:
    y = _ensure_batch_time(y)
    y_hat = _ensure_batch_time(y_hat)
    return _masked_mean_like(jnp.abs(y - y_hat), valid_mask)


def l1_diff_loss(
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    *,
    order: int,
    valid_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    y = _ensure_batch_time(y)
    y_hat = _ensure_batch_time(y_hat)
    y_diff = _finite_difference(y, order=order)
    y_hat_diff = _finite_difference(y_hat, order=order)
    diff_mask = None
    if valid_mask is not None:
        mask = _ensure_batch_time(valid_mask).astype(jnp.float32)
        if order == 1:
            diff_mask = mask[..., 1:] * mask[..., :-1]
        elif order == 2:
            diff_mask = mask[..., 2:] * mask[..., 1:-1] * mask[..., :-2]
        elif order > 0:
            diff_mask = mask
            for _ in range(order):
                diff_mask = diff_mask[..., 1:] * diff_mask[..., :-1]
    return _masked_mean_like(jnp.abs(y_diff - y_hat_diff), diff_mask)


def stft_logmag_l1_loss(
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    *,
    n_fft: int = 256,
    hop_length: int = 64,
    win_length: int = 256,
    valid_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    y_logmag = _stft_logmag(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    y_hat_logmag = _stft_logmag(y_hat, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    frame_weights = _frame_mask(
        valid_mask,
        frame_length=max(1, min(int(win_length), int(n_fft))),
        hop_length=max(1, int(hop_length)),
    )
    return _masked_mean_like(jnp.abs(y_logmag - y_hat_logmag), frame_weights)


def complex_stft_l1_loss(
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    *,
    n_fft: int = 128,
    hop_length: int = 24,
    win_length: int = 128,
    valid_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    y_spec = _stft_complex(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, norm="ortho")
    y_hat_spec = _stft_complex(y_hat, n_fft=n_fft, hop_length=hop_length, win_length=win_length, norm="ortho")
    diff = jnp.abs(jnp.real(y_spec) - jnp.real(y_hat_spec)) + jnp.abs(
        jnp.imag(y_spec) - jnp.imag(y_hat_spec)
    )
    frame_weights = _frame_mask(
        valid_mask,
        frame_length=max(1, min(int(win_length), int(n_fft))),
        hop_length=max(1, int(hop_length)),
    )
    return _masked_mean_like(diff, frame_weights)


def lowpass_l1_loss(
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    *,
    cutoff_hz: float | jnp.ndarray,
    sample_rate_hz: float | jnp.ndarray = 5000.0,
    valid_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    y = _ensure_batch_time(y)
    y_hat = _ensure_batch_time(y_hat)
    n = int(y.shape[-1])
    sample_rate = jnp.asarray(sample_rate_hz, dtype=jnp.float32)
    cutoff = jnp.asarray(cutoff_hz, dtype=jnp.float32)
    freqs = jnp.arange(n // 2 + 1, dtype=jnp.float32) * (sample_rate / float(n))
    keep = (freqs <= cutoff).astype(jnp.float32)
    y_lp = jnp.fft.irfft(jnp.fft.rfft(y, axis=-1) * keep[None, :], n=n, axis=-1)
    y_hat_lp = jnp.fft.irfft(jnp.fft.rfft(y_hat, axis=-1) * keep[None, :], n=n, axis=-1)
    return _masked_mean_like(jnp.abs(y_lp - y_hat_lp), valid_mask)


def pa_l1_loss(
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    *,
    pa_half_range: jnp.ndarray,
    scale: float | jnp.ndarray = 50.0,
    valid_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    y = _ensure_batch_time(y)
    y_hat = _ensure_batch_time(y_hat)
    half_range = jnp.asarray(pa_half_range, dtype=jnp.float32).reshape((-1, 1))
    scale_arr = jnp.maximum(jnp.asarray(scale, dtype=jnp.float32), 1e-6)
    err = jnp.abs(y - y_hat) * half_range / scale_arr
    return _masked_mean_like(err, valid_mask)


def ms_stft_logmag_l1_loss(
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    *,
    scales: Sequence[tuple[int, int, int]],
) -> jnp.ndarray:
    losses = [
        stft_logmag_l1_loss(
            y,
            y_hat,
            n_fft=int(n_fft),
            win_length=int(win_length),
            hop_length=int(hop_length),
        )
        for n_fft, win_length, hop_length in scales
    ]
    if not losses:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.mean(jnp.stack(losses))


def _stft_scale_labels(num_scales: int) -> tuple[str, ...]:
    if num_scales == 3:
        return ("small", "medium", "large")
    if num_scales == 1:
        return ("stft",)
    return tuple(f"scale_{idx}" for idx in range(num_scales))


def compute_reconstruction_losses(
    *,
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    weights: Dict[str, float | jnp.ndarray],
    stft_loss_scales: Sequence[tuple[int, int, int]] = ((256, 256, 64),),
    pa_half_range: jnp.ndarray | None = None,
    valid_mask: jnp.ndarray | None = None,
    decoder_aux: Dict[str, Any] | None = None,
) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    l_time = l1_time_loss(y, y_hat, valid_mask=valid_mask)
    l_diff1 = l1_diff_loss(y, y_hat, order=1, valid_mask=valid_mask)
    l_diff2 = l1_diff_loss(y, y_hat, order=2, valid_mask=valid_mask)
    dtype = l_time.dtype

    def _weight(name: str, default: float | jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(weights.get(name, default), dtype=dtype)

    w_time = _weight("time_l1", 1.0)
    w_pa = _weight("pa_l1", 0.0)
    w_diff1 = _weight("diff1_l1", 0.0)
    w_diff2 = _weight("diff2_l1", 0.0)
    legacy_stft_weight = _weight("stft_logmag_l1", 0.0)

    time_term = w_time * l_time
    if pa_half_range is not None:
        l_pa = pa_l1_loss(
            y,
            y_hat,
            pa_half_range=pa_half_range,
            scale=_weight("pa_l1_scale", 50.0),
            valid_mask=valid_mask,
        )
    else:
        l_pa = jnp.asarray(0.0, dtype=dtype)
    pa_term = w_pa * l_pa
    diff1_term = w_diff1 * l_diff1
    diff2_term = w_diff2 * l_diff2
    reconstruct = time_term + pa_term + diff1_term + diff2_term
    total = reconstruct
    logs = {
        "reconstruct_loss": reconstruct,
        "norm_l1_metric": l_time,
        "time_l1_loss_raw": l_time,
        "time_l1_loss": time_term,
        "pa_l1_loss_raw": l_pa,
        "pa_l1_loss": pa_term,
        "diff1_loss_raw": l_diff1,
        "diff1_loss": diff1_term,
        "diff2_loss_raw": l_diff2,
        "diff2_loss": diff2_term,
    }

    scale_labels = _stft_scale_labels(len(stft_loss_scales))
    if len(scale_labels) != len(stft_loss_scales):
        raise ValueError("STFT scale labels must match the number of STFT scales.")
    for label, (n_fft, win_length, hop_length) in zip(scale_labels, stft_loss_scales):
        raw_loss = stft_logmag_l1_loss(
            y,
            y_hat,
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            win_length=int(win_length),
            valid_mask=valid_mask,
        )
        weight = _weight(f"{label}_stft_logmag_l1", legacy_stft_weight)
        weighted_loss = weight * raw_loss
        reconstruct = reconstruct + weighted_loss
        total = total + weighted_loss
        logs[f"{label}_stft_logmag_loss_raw"] = raw_loss
        logs[f"{label}_stft_logmag_loss"] = weighted_loss

    complex_weight = _weight("complex_stft_l1", 0.0)
    if stft_loss_scales:
        n_fft, win_length, hop_length = stft_loss_scales[0]
    else:
        n_fft, win_length, hop_length = (128, 128, 24)
    complex_raw = complex_stft_l1_loss(
        y,
        y_hat,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
        valid_mask=valid_mask,
    )
    complex_term = complex_weight * complex_raw
    reconstruct = reconstruct + complex_term
    total = total + complex_term
    logs["complex_stft_l1_loss_raw"] = complex_raw
    logs["complex_stft_l1_loss"] = complex_term

    sample_rate_hz = _weight("sample_rate", 5000.0)
    for cutoff in (200, 500, 1000):
        weight = _weight(f"lowpass_l1_{cutoff}hz", 0.0)
        raw = lowpass_l1_loss(
            y,
            y_hat,
            cutoff_hz=float(cutoff),
            sample_rate_hz=sample_rate_hz,
            valid_mask=valid_mask,
        )
        term = weight * raw
        reconstruct = reconstruct + term
        total = total + term
        logs[f"lowpass_l1_{cutoff}hz_loss_raw"] = raw
        logs[f"lowpass_l1_{cutoff}hz_loss"] = term

    decoder_aux = dict(decoder_aux or {})
    residual_raw = jnp.asarray(decoder_aux.get("residual_energy_l2_raw", 0.0), dtype=dtype)
    residual_weight = _weight("residual_energy_l2", 0.0)
    residual_term = residual_weight * residual_raw
    reconstruct = reconstruct + residual_term
    total = total + residual_term
    logs["residual_energy_l2_raw"] = residual_raw
    logs["residual_energy_l2"] = residual_term
    for key, value in decoder_aux.items():
        if key in logs:
            continue
        arr = jnp.asarray(value)
        if arr.ndim == 0:
            logs[f"decoder_{key}"] = arr.astype(dtype)

    logs["reconstruct_loss"] = reconstruct

    return total, logs
