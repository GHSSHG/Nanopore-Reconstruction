from __future__ import annotations

from typing import Any, Dict, Sequence

import jax
import jax.numpy as jnp

from ..dorado import DoradoEncoderState, dorado_crf_nll, extract_dorado_crf_logits
from .losses import compute_reconstruction_losses


def dorado_logit_lengths(signal_lengths: jnp.ndarray) -> jnp.ndarray:
    lengths = jnp.maximum(jnp.asarray(signal_lengths, dtype=jnp.int32), 1)

    def _ceil_div(x: jnp.ndarray, divisor: int) -> jnp.ndarray:
        return (x + int(divisor) - 1) // int(divisor)

    lengths = _ceil_div(lengths, 1)
    lengths = _ceil_div(lengths, 1)
    lengths = _ceil_div(lengths, 3)
    lengths = _ceil_div(lengths, 2)
    lengths = _ceil_div(lengths, 2)
    return jnp.maximum(lengths * 2, 1)


def stitch_chunks_to_pa(
    chunks_norm: jnp.ndarray,
    chunk_center: jnp.ndarray,
    chunk_half_range: jnp.ndarray,
    chunk_valid_mask: jnp.ndarray,
    chunk_start: jnp.ndarray,
    *,
    output_length: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    chunks_norm = jnp.asarray(chunks_norm, dtype=jnp.float32)
    center = jnp.asarray(chunk_center, dtype=jnp.float32)[..., None]
    half_range = jnp.asarray(chunk_half_range, dtype=jnp.float32)[..., None]
    valid_mask = jnp.asarray(chunk_valid_mask, dtype=jnp.float32)
    starts = jnp.asarray(chunk_start, dtype=jnp.int32)
    read_count, chunk_count, chunk_size = chunks_norm.shape
    output_length = int(output_length)
    chunk_pa = chunks_norm * half_range + center
    offsets = starts[..., None] + jnp.arange(chunk_size, dtype=jnp.int32)[None, None, :]
    valid = (starts[..., None] >= 0) & (offsets >= 0) & (offsets < output_length) & (valid_mask > 0.0)
    offsets = jnp.clip(offsets, 0, max(0, output_length - 1))
    read_idx = jnp.broadcast_to(
        jnp.arange(read_count, dtype=jnp.int32)[:, None, None],
        (read_count, chunk_count, chunk_size),
    )
    weighted = jnp.where(valid, chunk_pa, 0.0)
    weights = jnp.where(valid, valid_mask, 0.0)
    out = jnp.zeros((read_count, output_length), dtype=jnp.float32)
    acc_w = jnp.zeros((read_count, output_length), dtype=jnp.float32)
    out = out.at[read_idx, offsets].add(weighted)
    acc_w = acc_w.at[read_idx, offsets].add(weights)
    stitched = jnp.where(acc_w > 0.0, out / jnp.maximum(acc_w, 1.0e-6), 0.0)
    return stitched, acc_w


def _extract_vq_vars(gen_state: Any) -> Any:
    return gen_state.vq_vars if gen_state.vq_vars is not None else {}


def compute_posttrain_grads(
    gen_state: Any,
    batch: dict[str, jnp.ndarray],
    rng: jax.Array,
    loss_weights: Dict[str, float],
    *,
    dorado_state: DoradoEncoderState,
    stft_loss_scales: Sequence[tuple[int, int, int]],
    seq_loss_weight: float,
    output_length: int,
    seq_state_len: int = 5,
    seq_n_base: int = 4,
) -> tuple[Any, dict[str, jnp.ndarray], Any]:
    def loss_fn(params: Any) -> tuple[jnp.ndarray, dict[str, Any]]:
        chunks = jnp.asarray(batch["chunks_norm"], dtype=jnp.float32)
        read_count, chunk_count, chunk_size = chunks.shape
        flat_chunks = chunks.reshape((read_count * chunk_count, chunk_size))
        flat_mask = jnp.asarray(batch["chunk_valid_mask"], dtype=jnp.float32).reshape(
            (read_count * chunk_count, chunk_size)
        )
        flat_half_range = jnp.asarray(batch["chunk_half_range"], dtype=jnp.float32).reshape(
            (read_count * chunk_count,)
        )
        outs = gen_state.apply_fn(
            {"params": params, "vq": _extract_vq_vars(gen_state)},
            flat_chunks,
            train=True,
            offset=0,
            rng=rng,
            collect_codebook_stats=False,
        )
        wave_hat = jnp.asarray(outs["wave_hat"], dtype=jnp.float32)
        recon_loss, logs = compute_reconstruction_losses(
            y=flat_chunks,
            y_hat=wave_hat,
            weights=loss_weights,
            stft_loss_scales=stft_loss_scales,
            pa_half_range=flat_half_range,
            valid_mask=flat_mask,
            decoder_aux=outs.get("dec", {}),
        )
        chunks_hat = wave_hat.reshape((read_count, chunk_count, chunk_size))
        read_pa, read_weight = stitch_chunks_to_pa(
            chunks_hat,
            batch["chunk_center"],
            batch["chunk_half_range"],
            batch["chunk_valid_mask"],
            batch["chunk_start"],
            output_length=output_length,
        )
        dorado_input = (read_pa - jnp.asarray(dorado_state.pa_mean, dtype=jnp.float32)) / jnp.asarray(
            dorado_state.pa_std, dtype=jnp.float32
        )
        scores = extract_dorado_crf_logits(dorado_input, dorado_state)
        logit_lengths = dorado_logit_lengths(batch["valid_length"])
        seq_nll_per_read = dorado_crf_nll(
            scores,
            batch["truth_tokens"],
            batch["truth_length"],
            logit_lengths,
            state_len=seq_state_len,
            n_base=seq_n_base,
            normalise_scores=True,
        )
        read_mask = jnp.asarray(batch.get("read_mask", jnp.ones((read_count,), dtype=jnp.float32)), dtype=jnp.float32)
        denom = jnp.maximum(jnp.sum(read_mask), 1.0)
        seq_nll = jnp.sum(seq_nll_per_read * read_mask) / denom
        seq_term = jnp.asarray(float(seq_loss_weight), dtype=jnp.float32) * seq_nll
        total = recon_loss + seq_term
        logs = dict(logs)
        logs["seq_nll"] = seq_nll
        logs["seq_loss"] = seq_term
        logs["total_loss"] = total
        logs["mean_logit_length"] = jnp.sum(logit_lengths.astype(jnp.float32) * read_mask) / denom
        logs["mean_truth_length"] = jnp.sum(batch["truth_length"].astype(jnp.float32) * read_mask) / denom
        logs["mean_valid_length"] = jnp.sum(batch["valid_length"].astype(jnp.float32) * read_mask) / denom
        logs["stitch_weight_min"] = jnp.min(jnp.where(read_weight > 0.0, read_weight, 1.0))
        logs["q_z_dist"] = outs["enc"].get("q_z_dist", jnp.asarray(0.0, dtype=jnp.float32))
        logs["log_q_z_dist"] = outs["enc"].get("log_q_z_dist", jnp.asarray(0.0, dtype=jnp.float32))
        return total, {"logs": logs, "vq_vars": gen_state.vq_vars}

    (_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(gen_state.params)
    return grads, aux["logs"], aux["vq_vars"]


__all__ = [
    "compute_posttrain_grads",
    "dorado_logit_lengths",
    "stitch_chunks_to_pa",
]
