from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def _prev_state_indices(*, state_len: int, n_base: int) -> jnp.ndarray:
    n_states = int(n_base) ** int(state_len)
    repeated = jnp.repeat(jnp.arange(n_states, dtype=jnp.int32), int(n_base))
    move_sources = repeated.reshape((int(n_base), n_states)).T
    blank_sources = jnp.arange(n_states, dtype=jnp.int32)[:, None]
    return jnp.concatenate((blank_sources, move_sources), axis=1)


@partial(jax.jit, static_argnames=("state_len", "n_base"))
def crf_log_partition(
    scores: jnp.ndarray,
    logit_lengths: jnp.ndarray | None = None,
    *,
    state_len: int = 5,
    n_base: int = 4,
) -> jnp.ndarray:
    scores = jnp.asarray(scores, dtype=jnp.float32)
    if scores.ndim != 3:
        raise ValueError(f"Expected CRF scores with shape (T,B,C), got {scores.shape}")
    n_states = int(n_base) ** int(state_len)
    n_actions = int(n_base) + 1
    if int(scores.shape[-1]) != n_states * n_actions:
        raise ValueError(f"Expected {n_states * n_actions} CRF scores, got {scores.shape[-1]}")
    transitions = scores.reshape((scores.shape[0], scores.shape[1], n_states, n_actions))
    prev_idx = _prev_state_indices(state_len=state_len, n_base=n_base)
    alpha0 = jnp.zeros((scores.shape[1], n_states), dtype=jnp.float32)
    if logit_lengths is None:
        logit_lengths = jnp.full((scores.shape[1],), int(scores.shape[0]), dtype=jnp.int32)
    else:
        logit_lengths = jnp.asarray(logit_lengths, dtype=jnp.int32)
    time_idx = jnp.arange(int(scores.shape[0]), dtype=jnp.int32)

    def _step(alpha: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, None]:
        edge_scores, t = inputs
        gathered_alpha = alpha[:, prev_idx]
        next_alpha = jax.nn.logsumexp(gathered_alpha + edge_scores, axis=-1)
        active = t < logit_lengths
        next_alpha = jnp.where(active[:, None], next_alpha, alpha)
        return next_alpha, None

    alpha_t, _ = jax.lax.scan(_step, alpha0, (transitions, time_idx))
    return jax.nn.logsumexp(alpha_t, axis=-1)


@partial(jax.jit, static_argnames=("state_len", "n_base"))
def prepare_ctc_scores(
    scores: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    state_len: int = 5,
    n_base: int = 4,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    scores = jnp.asarray(scores, dtype=jnp.float32)
    targets = jnp.asarray(targets, dtype=jnp.int32)
    if scores.ndim != 3:
        raise ValueError(f"Expected CRF scores with shape (T,B,C), got {scores.shape}")
    if targets.ndim != 2:
        raise ValueError(f"Expected target tokens with shape (B,L), got {targets.shape}")
    target_len = int(targets.shape[1])
    n_pos = target_len - int(state_len) + 1
    if n_pos <= 0:
        raise ValueError(f"Target token axis must be at least state_len={state_len}, got {target_len}")
    zero_based = jnp.clip(targets - 1, 0, int(n_base) - 1)
    powers = (int(n_base) ** jnp.arange(int(state_len) - 1, -1, -1, dtype=jnp.int32)).astype(jnp.int32)
    state_ids = jnp.zeros((targets.shape[0], n_pos), dtype=jnp.int32)
    for idx in range(int(state_len)):
        state_ids = state_ids + zero_based[:, idx : n_pos + idx] * powers[idx]
    stay_indices = state_ids * (int(n_base) + 1)
    move_indices = stay_indices[:, 1:] + zero_based[:, : n_pos - 1] + 1
    stay_scores = jnp.take_along_axis(
        scores,
        jnp.broadcast_to(stay_indices[None, :, :], (scores.shape[0], targets.shape[0], n_pos)),
        axis=2,
    )
    move_scores = jnp.take_along_axis(
        scores,
        jnp.broadcast_to(move_indices[None, :, :], (scores.shape[0], targets.shape[0], max(0, n_pos - 1))),
        axis=2,
    )
    return stay_scores, move_scores


@partial(jax.jit, static_argnames=("state_len", "n_base", "max_positions"))
def restricted_ctc_logz_from_scores(
    scores: jnp.ndarray,
    targets: jnp.ndarray,
    target_lengths: jnp.ndarray,
    logit_lengths: jnp.ndarray | None = None,
    *,
    state_len: int = 5,
    n_base: int = 4,
    max_positions: int,
) -> jnp.ndarray:
    scores = jnp.asarray(scores, dtype=jnp.float32)
    targets = jnp.asarray(targets, dtype=jnp.int32)
    target_lengths = jnp.asarray(target_lengths, dtype=jnp.int32)
    if scores.ndim != 3:
        raise ValueError(f"Expected CRF scores with shape (T,B,C), got {scores.shape}")
    if targets.ndim != 2:
        raise ValueError(f"Expected target tokens with shape (B,L), got {targets.shape}")
    if logit_lengths is None:
        logit_lengths = jnp.full((scores.shape[1],), int(scores.shape[0]), dtype=jnp.int32)
    else:
        logit_lengths = jnp.asarray(logit_lengths, dtype=jnp.int32)
    n_pos = int(max_positions)
    n_actions = int(n_base) + 1
    neg_inf = jnp.asarray(-1.0e30, dtype=jnp.float32)
    zero_based = jnp.clip(targets - 1, 0, int(n_base) - 1)
    powers = (int(n_base) ** jnp.arange(int(state_len) - 1, -1, -1, dtype=jnp.int32)).astype(jnp.int32)
    state_ids = jnp.zeros((targets.shape[0], n_pos), dtype=jnp.int32)
    for idx in range(int(state_len)):
        state_ids = state_ids + zero_based[:, idx : n_pos + idx] * powers[idx]
    stay_indices = state_ids * n_actions
    move_indices = stay_indices[:, 1:] + zero_based[:, : n_pos - 1] + 1
    alpha0 = jnp.full((scores.shape[1], n_pos), neg_inf, dtype=jnp.float32)
    alpha0 = alpha0.at[:, 0].set(0.0)
    pos = jnp.arange(n_pos, dtype=jnp.int32)[None, :]
    valid_pos = pos < target_lengths[:, None]
    time_idx = jnp.arange(int(scores.shape[0]), dtype=jnp.int32)

    def _step(alpha: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, None]:
        scores_t, t = inputs
        stay_t = jnp.take_along_axis(scores_t, stay_indices, axis=1)
        move_t = jnp.take_along_axis(scores_t, move_indices, axis=1)
        stay = alpha + stay_t
        move_from_prev = alpha[:, :-1] + move_t
        move = jnp.concatenate((jnp.full((scores.shape[1], 1), neg_inf, dtype=jnp.float32), move_from_prev), axis=1)
        next_alpha = jnp.logaddexp(stay, move)
        next_alpha = jnp.where(valid_pos, next_alpha, neg_inf)
        active = t < logit_lengths
        next_alpha = jnp.where(active[:, None], next_alpha, alpha)
        return next_alpha, None

    alpha_t, _ = jax.lax.scan(_step, alpha0, (scores, time_idx))
    end_idx = jnp.maximum(target_lengths - 1, 0)
    return jnp.take_along_axis(alpha_t, end_idx[:, None], axis=1)[:, 0]


@partial(jax.jit, static_argnames=("max_positions",))
def restricted_ctc_logz(
    stay_scores: jnp.ndarray,
    move_scores: jnp.ndarray,
    target_lengths: jnp.ndarray,
    *,
    max_positions: int,
) -> jnp.ndarray:
    stay_scores = jnp.asarray(stay_scores, dtype=jnp.float32)
    move_scores = jnp.asarray(move_scores, dtype=jnp.float32)
    target_lengths = jnp.asarray(target_lengths, dtype=jnp.int32)
    batch = int(stay_scores.shape[1])
    n_pos = int(max_positions)
    neg_inf = jnp.asarray(-1.0e30, dtype=jnp.float32)
    alpha0 = jnp.full((batch, n_pos), neg_inf, dtype=jnp.float32)
    alpha0 = alpha0.at[:, 0].set(0.0)
    pos = jnp.arange(n_pos, dtype=jnp.int32)[None, :]
    valid_pos = pos < target_lengths[:, None]

    def _step(alpha: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, None]:
        stay_t, move_t = inputs
        stay = alpha + stay_t
        move_from_prev = alpha[:, :-1] + move_t
        move = jnp.concatenate((jnp.full((batch, 1), neg_inf, dtype=jnp.float32), move_from_prev), axis=1)
        next_alpha = jnp.logaddexp(stay, move)
        next_alpha = jnp.where(valid_pos, next_alpha, neg_inf)
        return next_alpha, None

    alpha_t, _ = jax.lax.scan(_step, alpha0, (stay_scores, move_scores))
    end_idx = jnp.maximum(target_lengths - 1, 0)
    return jnp.take_along_axis(alpha_t, end_idx[:, None], axis=1)[:, 0]


@partial(jax.jit, static_argnames=("state_len", "n_base", "normalise_scores"))
def dorado_crf_nll(
    scores: jnp.ndarray,
    targets: jnp.ndarray,
    target_lengths: jnp.ndarray,
    logit_lengths: jnp.ndarray | None = None,
    *,
    state_len: int = 5,
    n_base: int = 4,
    normalise_scores: bool = True,
) -> jnp.ndarray:
    scores = jnp.asarray(scores, dtype=jnp.float32)
    targets = jnp.asarray(targets, dtype=jnp.int32)
    target_lengths = jnp.asarray(target_lengths, dtype=jnp.int32)
    seq_lengths = target_lengths + 1 - int(state_len)
    valid = seq_lengths > 0
    safe_seq_lengths = jnp.maximum(seq_lengths, 1)
    safe_target_lengths = jnp.maximum(target_lengths, 1)
    if logit_lengths is None:
        logit_lengths = jnp.full((scores.shape[1],), int(scores.shape[0]), dtype=jnp.int32)
    else:
        logit_lengths = jnp.asarray(logit_lengths, dtype=jnp.int32)
    if normalise_scores:
        log_partition = crf_log_partition(scores, logit_lengths, state_len=state_len, n_base=n_base)
        safe_logit_lengths = jnp.maximum(logit_lengths, 1).astype(jnp.float32)
        scores = scores - log_partition[None, :, None] / safe_logit_lengths[None, :, None]
    target_logz = restricted_ctc_logz_from_scores(
        scores,
        targets,
        safe_seq_lengths,
        logit_lengths,
        state_len=state_len,
        n_base=n_base,
        max_positions=int(targets.shape[1]) - int(state_len) + 1,
    )
    loss = -(target_logz / safe_target_lengths.astype(jnp.float32))
    return jnp.where(valid, loss, 0.0)


__all__ = [
    "crf_log_partition",
    "prepare_ctc_scores",
    "restricted_ctc_logz_from_scores",
    "restricted_ctc_logz",
    "dorado_crf_nll",
]
