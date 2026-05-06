from __future__ import annotations

import numpy as np


NORMALIZATION_MINMAX_PM1 = "minmax_pm1"
NORMALIZATION_GLOBAL_ZSCORE = "global_zscore"


def normalize_mode(mode: str | None) -> str:
    """Return the canonical normalization mode name."""
    text = str(mode or NORMALIZATION_MINMAX_PM1).strip().lower().replace("-", "_")
    aliases = {
        "legacy": NORMALIZATION_MINMAX_PM1,
        "minmax": NORMALIZATION_MINMAX_PM1,
        "minmax_pm1": NORMALIZATION_MINMAX_PM1,
        "pm1": NORMALIZATION_MINMAX_PM1,
        "per_chunk_minmax": NORMALIZATION_MINMAX_PM1,
        "per_read_minmax": NORMALIZATION_MINMAX_PM1,
        "global_zscore": NORMALIZATION_GLOBAL_ZSCORE,
        "zscore": NORMALIZATION_GLOBAL_ZSCORE,
        "sup": NORMALIZATION_GLOBAL_ZSCORE,
        "sup_zscore": NORMALIZATION_GLOBAL_ZSCORE,
        "dorado_zscore": NORMALIZATION_GLOBAL_ZSCORE,
    }
    if text not in aliases:
        raise ValueError(
            f"Unsupported normalization mode {mode!r}; choose from "
            f"['{NORMALIZATION_MINMAX_PM1}', '{NORMALIZATION_GLOBAL_ZSCORE}']."
        )
    return aliases[text]


def _as_1d_float(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return np.asarray(arr, dtype=np.float32)


def normalize_to_pm1_with_stats(signal: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, float, float]:
    """Normalize a 1D signal to [-1, 1] using per-signal center/half-range."""
    x = _as_1d_float(signal)
    if x.size == 0:
        return x, 0.0, 0.0
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if not np.isfinite(x_min):
        x_min = 0.0
    if not np.isfinite(x_max):
        x_max = x_min
    center = 0.5 * (x_min + x_max)
    half_range = 0.5 * (x_max - x_min)
    if not np.isfinite(center):
        center = 0.0
    if not np.isfinite(half_range):
        half_range = 0.0
    if half_range < eps:
        normalized = np.zeros_like(x, dtype=np.float32)
    else:
        normalized = np.asarray((x - center) / half_range, dtype=np.float32)
        normalized = np.clip(normalized, -1.0, 1.0)
    return normalized, center, half_range


def normalize_to_pm1(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convenience wrapper for per-signal [-1, 1] normalization."""
    normalized, _, _ = normalize_to_pm1_with_stats(signal, eps)
    return normalized


def normalize_global_zscore_with_stats(
    signal: np.ndarray,
    *,
    mean: float,
    std: float,
    eps: float = 1e-6,
) -> tuple[np.ndarray, float, float]:
    """Normalize a 1D signal with fixed global pA mean/std, without clipping."""
    x = _as_1d_float(signal)
    center = float(mean)
    scale = float(std)
    if not np.isfinite(center):
        raise ValueError(f"global_zscore mean must be finite, got {mean!r}")
    if not np.isfinite(scale) or scale < eps:
        raise ValueError(f"global_zscore std must be finite and >= {eps}, got {std!r}")
    if x.size == 0:
        return x, center, scale
    normalized = np.asarray((x - center) / scale, dtype=np.float32)
    return normalized, center, scale


def normalize_signal_with_stats(
    signal: np.ndarray,
    *,
    mode: str | None = NORMALIZATION_MINMAX_PM1,
    mean: float | None = None,
    std: float | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, float, float]:
    """Normalize a signal and return values plus reversible center/scale stats."""
    resolved = normalize_mode(mode)
    if resolved == NORMALIZATION_MINMAX_PM1:
        return normalize_to_pm1_with_stats(signal, eps=eps)
    if mean is None or std is None:
        raise ValueError("global_zscore normalization requires both mean and std.")
    return normalize_global_zscore_with_stats(signal, mean=float(mean), std=float(std), eps=eps)
