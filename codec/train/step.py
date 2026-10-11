from __future__ import annotations

from functools import partial
from typing import Any, Dict

import jax
import jax.numpy as jnp

from .losses import compute_reconstruction_losses
from ..models.model import SimVQAudioModel

_ENCODER_LOG_KEYS = ("q_z_dist", "log_q_z_dist")


def make_train_step(loss_weights: Dict[str, float], stft_loss_scales: tuple[tuple[int, int, int], ...]):
    """Data-parallel train step over the local devices: (replicated state, sharded batch) ->
    (new state, logs averaged over devices). The model is deterministic, so no random keys."""

    @partial(jax.pmap, axis_name="data")
    def train_step(state, batch: Dict[str, Any]):
        def loss_fn(params):
            out = state.apply_fn({"params": params, "vq": state.vq_vars}, batch["signal"])
            total, logs = compute_reconstruction_losses(
                y=batch["signal"],
                y_hat=out["wave_hat"],
                weights=loss_weights,
                stft_loss_scales=stft_loss_scales,
                pa_half_range=batch["pa_half_range"],
                valid_mask=batch["valid_mask"],
                decoder_aux=out["dec"],
            )
            logs = dict(logs, total_loss=total, **{k: out["enc"][k] for k in _ENCODER_LOG_KEYS})
            return total, logs

        (_, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, "data")
        return state.apply_gradients(grads=grads), jax.lax.pmean(logs, "data")

    return train_step


def make_codebook_stats(codebook_size: int):
    """Codebook perplexity and utilization over the whole (all-device) batch; encoder forward only.
    Run every few thousand steps: the statistics change slowly and only their trend matters."""

    @partial(jax.pmap, axis_name="data")
    def codebook_stats(state, batch: Dict[str, Any]):
        _, _, info = state.apply_fn(
            {"params": state.params, "vq": state.vq_vars}, batch["signal"], method=SimVQAudioModel.encode
        )
        counts = jax.lax.psum(jnp.bincount(info["indices"].reshape(-1), length=codebook_size), "data")
        probs = counts.astype(jnp.float32) / jnp.sum(counts)
        return {
            "codebook_perplexity": jnp.exp(-jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0.0))),
            "codebook_util": jnp.mean((counts > 0).astype(jnp.float32)),
        }

    return codebook_stats
