from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Any, Callable, Dict

import jax
import numpy as np
import optax
import pynvml
from flax import jax_utils
from flax.training import checkpoints as flax_ckpt

from .states import create_generator_state
from .step import codebook_stats, make_train_step
from ..data.pod5_dataset import Pod5ChunkDataset


def _host_broadcast(tree: Any, ndev: int) -> Any:
    """Host copies of every leaf stacked to (ndev, ...); pmap then sends each device its copy from
    host memory. Not jax_utils.replicate: it fills the other GPUs by GPU-to-GPU copies, which
    arrive corrupted on the lab server (GPU1-3 then train on garbage weights)."""
    return jax.tree_util.tree_map(lambda x: np.broadcast_to(np.asarray(x), (ndev,) + np.shape(x)).copy(), jax.device_get(tree))


@functools.cache
def _nvml_handles() -> list[Any]:
    pynvml.nvmlInit()
    return [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]


def _gpu_temps() -> list[int]:
    """Temperature (degC) of every GPU in the machine (NVML)."""
    return [pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU) for h in _nvml_handles()]


def _temperature_guard(pause: float, resume: float, log: Callable[[str], None], wait: float = 10.0) -> float:
    """If any GPU is at or above `pause` degC, sleep until every GPU is at or below `resume`.
    Returns the seconds spent here."""
    temps = _gpu_temps()
    if max(temps) < pause:
        return 0.0
    t0 = time.perf_counter()
    log(f"[temp] {max(temps)} degC, pausing until every GPU is <= {resume:.0f}")
    while max(temps) > resume:
        time.sleep(wait)
        temps = _gpu_temps()
    log(f"[temp] resumed after {time.perf_counter() - t0:.0f} s")
    return time.perf_counter() - t0


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
    on_checkpoint: Callable[[int, str], None] | None = None,
    temp_every_steps: int = 20,
    temp_pause: float = 80.0,
    temp_resume: float = 72.0,
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
    state = _host_broadcast(state, ndev)
    train_step = make_train_step(loss_weights, stft_loss_scales, model.codebook_size)

    def _zero_counts():  # fresh buffers: the train step donates (reuses) the counts it is given
        return np.zeros((ndev, model.codebook_size), np.int32)

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
        if on_checkpoint is not None:
            on_checkpoint(step, path)

    def _shard(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {k: v.reshape((ndev, v.shape[0] // ndev) + v.shape[1:]) for k, v in batch.items()}

    global_step = int(jax.device_get(state.step[0]))
    window: list[Dict[str, Any]] = []
    window_start, data_wait, paused = time.perf_counter(), 0.0, 0.0
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
                if temp_every_steps > 0 and global_step % temp_every_steps == 0:
                    paused += _temperature_guard(temp_pause, temp_resume, _log)

                if log_every_steps > 0 and global_step % log_every_steps == 0:
                    host = jax.device_get(window)
                    means = {k: float(np.mean([w[k][0] for w in host])) for k in host[0]}
                    # Codebook statistics over every code chosen in the window, all devices.
                    means.update(codebook_stats(jax.device_get(counts).sum(axis=0)))
                    elapsed = time.perf_counter() - window_start - paused
                    perf = {"step_ms": 1000.0 * elapsed / len(window), "data_wait_ms": 1000.0 * data_wait / len(window),
                            "paused_s": paused}
                    temps = {f"gpu{i}/temp": t for i, t in enumerate(_gpu_temps())}
                    _log(f"[step {global_step}] " + ", ".join(f"{k}={v:.4f}" for k, v in sorted(means.items())))
                    _log(f"[perf step {global_step}] step_ms={perf['step_ms']:.1f}, data_wait_ms={perf['data_wait_ms']:.1f}, "
                         f"paused_s={paused:.0f}, temps={list(temps.values())}")
                    if wandb_logger is not None:
                        wandb_logger.log({**means, **{f"perf/{k}": v for k, v in perf.items()}, **temps}, step=global_step)
                    window, counts = [], _zero_counts()
                    window_start, data_wait, paused = time.perf_counter(), 0.0, 0.0

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
