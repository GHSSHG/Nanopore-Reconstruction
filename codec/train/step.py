from __future__ import annotations

from functools import partial
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from .losses import compute_reconstruction_losses

_ENCODER_LOG_KEYS = ("vq_codebook_loss", "vq_commitment_loss", "q_z_dist", "z_norm", "code_norm")


def make_train_step(loss_weights: Dict[str, float], stft_loss_scales: tuple[tuple[int, int, int], ...], codebook_size: int):
    """Data-parallel train step over the local devices: (replicated state, sharded batch,
    per-device code counts) -> (new state, logs averaged over devices, code counts). The model is
    deterministic, so no random keys. Each device counts the codes it chose (no communication);
    the host sums the devices' counts once per logging window for the codebook statistics."""
    w_codebook = float(loss_weights.get("vq_codebook", 0.0))
    w_commitment = float(loss_weights.get("vq_commitment", 0.0))

    @partial(jax.pmap, axis_name="data", donate_argnums=(2,))
    def train_step(state, batch: Dict[str, Any], counts):
        def loss_fn(params):
            out = state.apply_fn({"params": params, "vq": state.vq_vars}, batch["signal"])
            recon, logs = compute_reconstruction_losses(
                y=batch["signal"],
                y_hat=out["wave_hat"],
                weights=loss_weights,
                stft_loss_scales=stft_loss_scales,
                pa_half_range=batch["pa_half_range"],
                valid_mask=batch["valid_mask"],
                decoder_aux=out["dec"],
            )
            enc = out["enc"]
            total = recon + w_codebook * enc["vq_codebook_loss"] + w_commitment * enc["vq_commitment_loss"]
            logs = dict(logs, total_loss=total, **{k: enc[k] for k in _ENCODER_LOG_KEYS})
            return total, (logs, enc["indices"])

        (_, (logs, indices)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, "data")
        counts = counts + jnp.bincount(indices.reshape(-1), length=codebook_size)
        return state.apply_gradients(grads=grads), jax.lax.pmean(logs, "data"), counts

    return train_step


def codebook_stats(counts: np.ndarray) -> Dict[str, float]:
    """Perplexity and utilization of the codebook from accumulated code counts."""
    probs = counts.astype(np.float64) / max(int(counts.sum()), 1)
    nz = probs[probs > 0]
    return {
        "codebook_perplexity": float(np.exp(-np.sum(nz * np.log(nz)))),
        "codebook_util": float(np.mean(counts > 0)),
    }
