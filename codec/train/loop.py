from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import jax
import numpy as np
import optax
from flax import jax_utils
from flax.training import checkpoints as flax_ckpt

from .states import create_generator_state
from .step import codebook_stats, make_train_step
from ..data.pod5_dataset import Pod5ChunkDataset


def train_model_from_pod5(
    ds: Pod5ChunkDataset,
    *,
    model_cfg: Dict[str, Any],
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    warmup_steps: int,
    grad_clip: float,
    lr_multipliers: Dict[str, float],
    loss_weights: Dict[str, float],
    stft_loss_scales: tuple[tuple[int, int, int], ...],
    seed: int,
    ckpt_dir: Path,
    resume_from: Path | None = None,
    log_every_steps: int = 500,
    checkpoint_every_steps: int = 10000,
    prefetch_batches: int = 8,
    max_steps_total: int | None = None,
    max_steps_per_epoch: int | None = None,
    wandb_logger: Any | None = None,
):
    from ..models import build_audio_model

    ckpt_dir = Path(ckpt_dir)
    periodic_dir = ckpt_dir / "periodic"
    periodic_dir.mkdir(parents=True, exist_ok=True)
    log_fp = open(ckpt_dir / "train.log", "a", encoding="utf-8")

    def _log(msg: str) -> None:
        print(msg, flush=True)
        log_fp.write(msg + "\n")
        log_fp.flush()

    ndev = jax.local_device_count()
    if batch_size % ndev != 0:
        raise ValueError(f"batch_size={batch_size} must be divisible by the {ndev} local devices.")
    _log(f"[setup] {ndev} devices, global batch {batch_size} ({batch_size // ndev} per device)")

    # Initialized on the first GPU (cuDNN attention has no CPU version), then kept on the host
    # until replication.
    model = build_audio_model(model_cfg)
    state, _ = create_generator_state(
        jax.random.PRNGKey(seed),
        model,
        (1, ds.chunk_samples),
        # Linear warmup from 0, then constant.
        optax.linear_schedule(0.0, float(learning_rate), int(warmup_steps)) if warmup_steps > 0 else float(learning_rate),
        grad_clip=grad_clip,
        group_lrs=lr_multipliers,
    )
    state = jax.device_get(state)
    if resume_from is not None:
        state = flax_ckpt.restore_checkpoint(ckpt_dir=str(resume_from), target={"gen": state})["gen"]
        _log(f"[resume] restored {resume_from} (step={int(state.step)})")
    state = jax_utils.replicate(state)
    train_step = make_train_step(loss_weights, stft_loss_scales, model.codebook_size)

    def _zero_counts():  # fresh buffers: the train step donates (reuses) the counts it is given
        return jax_utils.replicate(np.zeros(model.codebook_size, np.int32))

    counts = _zero_counts()

    def _save(step: int) -> None:
        path = flax_ckpt.save_checkpoint(
            str(periodic_dir.resolve()),
            target={"gen": jax.device_get(jax_utils.unreplicate(state))},
            step=step,
            overwrite=False,
            keep=1_000_000,
        )
        _log(f"[ckpt] step={step} written to {path}")

    def _shard(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {k: v.reshape((ndev, v.shape[0] // ndev) + v.shape[1:]) for k, v in batch.items()}

    global_step = int(jax.device_get(state.step[0]))
    window: list[Dict[str, Any]] = []
    window_start, data_wait = time.perf_counter(), 0.0
    saved_step = global_step
    try:
        for epoch in range(1, int(num_epochs) + 1):
            batches = ds.batches(batch_size, epoch, prefetch=prefetch_batches)
            device_batches = jax_utils.prefetch_to_device((_shard(b) for b in batches), 2)
            steps_this_epoch = 0
            while not (max_steps_per_epoch and steps_this_epoch >= max_steps_per_epoch):
                if max_steps_total and global_step >= max_steps_total:
                    break
                t0 = time.perf_counter()
                batch = next(device_batches, None)
                if batch is None:
                    break
                data_wait += time.perf_counter() - t0
                state, logs, counts = train_step(state, batch, counts)
                window.append(logs)
                steps_this_epoch += 1
                global_step += 1

                if log_every_steps > 0 and global_step % log_every_steps == 0:
                    host = jax.device_get(window)
                    means = {k: float(np.mean([w[k][0] for w in host])) for k in host[0]}
                    # Codebook statistics over every code chosen in the window, all devices.
                    means.update(codebook_stats(jax.device_get(counts).sum(axis=0)))
                    elapsed = time.perf_counter() - window_start
                    perf = {"step_ms": 1000.0 * elapsed / len(window), "data_wait_ms": 1000.0 * data_wait / len(window)}
                    _log(f"[step {global_step}] " + ", ".join(f"{k}={v:.4f}" for k, v in sorted(means.items())))
                    _log(f"[perf step {global_step}] step_ms={perf['step_ms']:.1f}, data_wait_ms={perf['data_wait_ms']:.1f}")
                    if wandb_logger is not None:
                        wandb_logger.log({**means, **{f"perf/{k}": v for k, v in perf.items()}}, step=global_step)
                    window, counts = [], _zero_counts()
                    window_start, data_wait = time.perf_counter(), 0.0

                if checkpoint_every_steps > 0 and global_step % checkpoint_every_steps == 0:
                    _save(global_step)
                    saved_step = global_step
            batches.close()
            if steps_this_epoch == 0 and not (max_steps_total and global_step >= max_steps_total):
                raise ValueError(f"Epoch {epoch} yielded no training batches; check the data settings.")
            _log(f"[epoch {epoch}/{num_epochs}] completed {steps_this_epoch} steps (global={global_step}).")
            if max_steps_total and global_step >= max_steps_total:
                break
        if global_step != saved_step:
            _save(global_step)
        return jax_utils.unreplicate(state)
    finally:
        log_fp.close()
