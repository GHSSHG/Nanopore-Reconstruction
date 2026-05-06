from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .losses import compute_reconstruction_losses


def _extract_signal(batch):
    if isinstance(batch, dict):
        return batch["signal"]
    return batch


def _extract_pa_half_range(batch):
    if not isinstance(batch, dict):
        return None
    return batch.get("pa_half_range")


def _extract_valid_mask(batch):
    if not isinstance(batch, dict):
        return None
    return batch.get("valid_mask")


@partial(
    jax.jit,
    static_argnames=("collect_codebook_stats", "stft_loss_scales"),
)
def compute_grads(
    gen_state,
    batch,
    rng,
    loss_weights,
    *,
    stft_loss_scales: tuple[tuple[int, int, int], ...] = ((256, 256, 64),),
    collect_codebook_stats: bool = True,
):
    def gen_loss_fn(params):
        signal = _extract_signal(batch)
        pa_half_range = _extract_pa_half_range(batch)
        valid_mask = _extract_valid_mask(batch)
        vq_in = gen_state.vq_vars if gen_state.vq_vars is not None else {}
        outs = gen_state.apply_fn(
            {"params": params, "vq": vq_in},
            signal,
            train=True,
            offset=0,
            rng=rng,
            collect_codebook_stats=collect_codebook_stats,
        )
        vq_vars = gen_state.vq_vars
        wave_hat = outs["wave_hat"]
        dec_aux = outs.get("dec", {})
        total_loss, logs = compute_reconstruction_losses(
            y=signal,
            y_hat=wave_hat,
            weights=loss_weights,
            stft_loss_scales=stft_loss_scales,
            pa_half_range=pa_half_range,
            valid_mask=valid_mask,
            decoder_aux=dec_aux,
        )
        logs = dict(logs)
        logs["total_loss"] = total_loss
        logs["q_z_dist"] = outs["enc"].get("q_z_dist", jnp.asarray(0.0, dtype=jnp.float32))
        logs["log_q_z_dist"] = outs["enc"].get("log_q_z_dist", jnp.asarray(0.0, dtype=jnp.float32))
        if collect_codebook_stats:
            logs["perplexity"] = outs["enc"].get("perplexity", jnp.array(0.0))
            usage_ratio = outs["enc"].get("usage_ratio")
            if usage_ratio is not None:
                logs["code_usage"] = usage_ratio
        loss_dtype = signal.dtype
        total_loss = jnp.asarray(total_loss, dtype=loss_dtype)
        logs = {k: jnp.asarray(v, dtype=loss_dtype) for k, v in logs.items()}
        aux = {"logs": logs, "vq_vars": vq_vars}
        return total_loss, aux

    (_g_loss, aux), g_grads = jax.value_and_grad(gen_loss_fn, has_aux=True)(gen_state.params)
    return g_grads, aux["logs"], aux["vq_vars"]


@partial(jax.jit, static_argnames=("train",))
def compute_codebook_stats(
    gen_state,
    batch,
    rng,
    *,
    train: bool = False,
):
    signal = _extract_signal(batch)
    vq_in = gen_state.vq_vars if gen_state.vq_vars is not None else {}
    outs = gen_state.apply_fn(
        {"params": gen_state.params, "vq": vq_in},
        signal,
        train=train,
        offset=0,
        rng=rng,
        collect_codebook_stats=True,
    )
    enc = outs["enc"]
    loss_dtype = signal.dtype
    return {
        "perplexity": jnp.asarray(enc.get("perplexity", jnp.asarray(0.0, dtype=jnp.float32)), dtype=loss_dtype),
        "code_usage": jnp.asarray(enc.get("usage_ratio", jnp.asarray(0.0, dtype=jnp.float32)), dtype=loss_dtype),
    }
