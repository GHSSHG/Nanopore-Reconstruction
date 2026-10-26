from __future__ import annotations

from functools import partial
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from .losses import compute_reconstruction_losses

_ENCODER_LOG_KEYS = ("vq_codebook_loss", "vq_commitment_loss", "q_z_dist", "z_norm", "code_norm")


def make_train_step(
    loss_weights: Dict[str, float],
    stft_loss_scales: tuple[tuple[int, int, int], ...],
    codebook_size: int,
    ema_decay: float,
):
    """Data-parallel train step over the local devices: (replicated state, sharded batch,
    per-device code counts) -> (new state, logs averaged over devices, code counts). The model is
    deterministic, so no random keys. Each device counts the codes it chose (no communication);
    the host sums the devices' counts once per logging window for the codebook statistics.
    state.ema_params follows the params with decay min(ema_decay, (1 + step) / (10 + step)), so
    the average is not dominated by the random initialization early on."""
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
            )
            enc = out["enc"]
            total = recon + w_codebook * enc["vq_codebook_loss"] + w_commitment * enc["vq_commitment_loss"]
            logs = dict(logs, total_loss=total, **{k: enc[k] for k in _ENCODER_LOG_KEYS})
            return total, (logs, enc["indices"])

        (_, (logs, indices)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, "data")
        counts = counts + jnp.bincount(indices.reshape(-1), length=codebook_size)
        state = state.apply_gradients(grads=grads)
        step = state.step.astype(jnp.float32)
        decay = jnp.minimum(ema_decay, (1.0 + step) / (10.0 + step))
        ema = jax.tree_util.tree_map(lambda e, p: decay * e + (1.0 - decay) * p, state.ema_params, state.params)
        return state.replace(ema_params=ema), jax.lax.pmean(logs, "data"), counts

    return train_step


def codebook_stats(counts: np.ndarray) -> Dict[str, float]:
    """Perplexity and utilization of the codebook from accumulated code counts."""
    probs = counts.astype(np.float64) / max(int(counts.sum()), 1)
    nz = probs[probs > 0]
    return {
        "codebook_perplexity": float(np.exp(-np.sum(nz * np.log(nz)))),
        "codebook_util": float(np.mean(counts > 0)),
    }
