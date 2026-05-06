#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import json
import os
import queue
import shutil
import sys
import threading
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from codec.runtime import configure_runtime_env, enable_jax_compilation_cache

configure_runtime_env()

try:
    from jax import config as _jax_config  # type: ignore

    _jax_config.update("jax_default_matmul_precision", "high")
except Exception:
    pass

enable_jax_compilation_cache()

import jax
import numpy as np
from flax import jax_utils as flax_jax_utils
from flax.core import FrozenDict, freeze
from flax.training import checkpoints as flax_ckpt

from codec.data import PosttrainShardDataset
from codec.dorado import load_dorado_encoder_state
from codec.models import build_audio_model
from codec.train import compute_posttrain_grads, create_generator_state
from codec.utils import init_wandb


NUMERIC_BATCH_KEYS = frozenset(
    {
        "chunks_norm",
        "chunk_center",
        "chunk_half_range",
        "chunk_valid_mask",
        "chunk_start",
        "chunk_valid_length",
        "valid_length",
        "truth_tokens",
        "truth_qscores",
        "truth_length",
        "read_mask",
        "read_length_bases",
        "identity",
        "accuracy",
        "mean_quality",
        "raw_signal_length_samples",
        "chunk_count",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_path(value: str | os.PathLike[str] | None, *, base: Path) -> str | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def _sanitize_subdir(name: str | None) -> str | None:
    text = str(name or "").strip()
    if not text:
        return None
    for token in (os.sep, "/", "\\"):
        text = text.replace(token, "_")
    return text


def _resolve_checkpoint_dir(root_dir: str | os.PathLike[str], run_name: str | None) -> Path:
    root = Path(root_dir).expanduser()
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    else:
        root = root.resolve()
    subdir = _sanitize_subdir(run_name)
    if subdir and root.name != subdir:
        root = root / subdir
    return root


def _reset_checkpoint_dir(path: Path) -> None:
    protected = {Path("/"), Path.home().resolve(), REPO_ROOT.resolve(), REPO_ROOT.parent.resolve()}
    if path.resolve() in protected:
        raise ValueError(f"Refusing to delete protected path: {path}")
    if path.exists():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def _normalize_stft_loss_scales(stft_loss_cfg: dict[str, Any] | None) -> tuple[tuple[int, int, int], ...]:
    cfg = dict(stft_loss_cfg or {})
    scales_raw = cfg.get("scales")
    if scales_raw is None:
        n_fft = max(1, int(cfg.get("n_fft", 256)))
        win_length = max(1, min(int(cfg.get("win_length", n_fft)), n_fft))
        hop_length = max(1, int(cfg.get("hop_length", max(1, win_length // 4))))
        return ((n_fft, win_length, hop_length),)
    scales: list[tuple[int, int, int]] = []
    for item in scales_raw:
        if isinstance(item, dict):
            n_fft = max(1, int(item.get("n_fft", 256)))
            win_length = max(1, min(int(item.get("win_length", n_fft)), n_fft))
            hop_length = max(1, int(item.get("hop_length", max(1, win_length // 4))))
        else:
            n_fft, win_length, hop_length = (int(v) for v in item)
            n_fft = max(1, n_fft)
            win_length = max(1, min(win_length, n_fft))
            hop_length = max(1, hop_length)
        scales.append((n_fft, win_length, hop_length))
    if not scales:
        raise ValueError("train.stft_loss.scales must not be empty.")
    return tuple(scales)


def _loss_weights(train_cfg: dict[str, Any], data_cfg: dict[str, Any]) -> dict[str, float]:
    lw = dict(train_cfg.get("loss_weights") or {})
    loss_cfg = dict(train_cfg.get("loss") or {})
    return {
        "time_l1": float(lw.get("time_l1", lw.get("recon", 0.0))),
        "pa_l1": float(lw.get("pa_l1", 0.0)),
        "pa_l1_scale": float(loss_cfg.get("pa_l1_scale", lw.get("pa_l1_scale", 50.0))),
        "diff1_l1": float(lw.get("diff1_l1", 0.1)),
        "diff2_l1": float(lw.get("diff2_l1", 0.05)),
        "small_stft_logmag_l1": float(lw.get("small_stft_logmag_l1", lw.get("stft_logmag_l1", 0.1))),
        "medium_stft_logmag_l1": float(lw.get("medium_stft_logmag_l1", lw.get("stft_logmag_l1", 0.1))),
        "large_stft_logmag_l1": float(lw.get("large_stft_logmag_l1", lw.get("stft_logmag_l1", 0.1))),
        "complex_stft_l1": float(lw.get("complex_stft_l1", 0.0)),
        "lowpass_l1_200hz": float(lw.get("lowpass_l1_200hz", 0.0)),
        "lowpass_l1_500hz": float(lw.get("lowpass_l1_500hz", 0.0)),
        "lowpass_l1_1000hz": float(lw.get("lowpass_l1_1000hz", 0.0)),
        "residual_energy_l2": float(lw.get("residual_energy_l2", 0.0)),
        "sample_rate": float(data_cfg.get("sample_rate", loss_cfg.get("sample_rate", 5000.0))),
    }


def _numeric_batch(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in batch.items():
        if key not in NUMERIC_BATCH_KEYS:
            continue
        arr = np.asarray(value)
        if arr.dtype.kind in {"U", "S", "O"}:
            continue
        if key in {"truth_tokens", "truth_length", "valid_length", "chunk_start", "chunk_valid_length", "read_length_bases", "raw_signal_length_samples", "chunk_count"}:
            out[key] = arr.astype(np.int32, copy=False)
        else:
            out[key] = arr.astype(np.float32, copy=False)
    return out


def _shard_batch(batch: dict[str, np.ndarray], *, device_count: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in batch.items():
        arr = np.asarray(value)
        if arr.shape[0] % int(device_count) != 0:
            raise ValueError(f"Batch key {key} has leading dim {arr.shape[0]} not divisible by {device_count}.")
        per_device = arr.shape[0] // int(device_count)
        out[key] = arr.reshape((int(device_count), per_device) + arr.shape[1:])
    return out


def _host_broadcast_tree_for_pmap(tree: Any, replica_count: int) -> Any:
    host_tree = jax.device_get(tree)

    def _broadcast_leaf(x: Any) -> np.ndarray:
        arr = np.asarray(x)
        return np.broadcast_to(arr, (replica_count,) + arr.shape).copy()

    return jax.tree_util.tree_map(_broadcast_leaf, host_tree)


def _state_step_as_int(state: Any) -> int:
    step = getattr(state, "step", 0)
    if getattr(step, "ndim", 0):
        step = step[0]
    return int(jax.device_get(step))


def _with_vq_default(state: Any, base_vq_vars: Any) -> Any:
    vq_existing = getattr(state, "vq_vars", None)
    try:
        has_vq = vq_existing is not None and len(vq_existing) > 0
    except TypeError:
        has_vq = vq_existing is not None
    if has_vq:
        return state
    if base_vq_vars is not None:
        return state.replace(vq_vars=base_vq_vars)
    return state


def _value_to_float(value: Any) -> float:
    try:
        return float(jax.device_get(value))
    except Exception:
        return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-train SimVQGAN with Dorado SUP sequence NLL.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "SUP" / "sup-offline.json")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--bucket", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-reset", action="store_true", help="Do not delete an existing output checkpoint directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = _read_json(config_path)
    cfg_dir = config_path.parent
    train_cfg = dict(cfg.get("train") or {})
    data_cfg = dict(cfg.get("data") or {})
    ckpt_cfg = dict(cfg.get("checkpoint") or {})
    logging_cfg = dict(cfg.get("logging") or {})
    wandb_cfg = dict(logging_cfg.get("wandb") or {})
    model_cfg = dict(cfg.get("model") or {})
    dorado_cfg = dict(cfg.get("dorado") or {})

    dataset_dir = Path(args.dataset or _resolve_path(data_cfg.get("dataset"), base=cfg_dir) or ".").expanduser().resolve()
    bucket_chunks = int(args.bucket or train_cfg.get("bucket_chunks", 8))
    batch_size = int(train_cfg.get("batch_size", 8))
    device_count = int(train_cfg.get("device_count", 4))
    local_devices = jax.local_devices()
    if len(local_devices) < device_count:
        raise RuntimeError(f"Requested {device_count} devices, but only {len(local_devices)} local JAX devices are visible.")
    devices = local_devices[:device_count]
    if batch_size % device_count != 0:
        raise ValueError(f"batch_size={batch_size} must be divisible by device_count={device_count}.")
    per_device_reads = batch_size // device_count
    if per_device_reads * bucket_chunks > int(train_cfg.get("per_device_chunk_budget", 16)):
        raise ValueError(
            f"per_device_reads={per_device_reads} * bucket_chunks={bucket_chunks} exceeds "
            f"per_device_chunk_budget={train_cfg.get('per_device_chunk_budget', 16)}."
        )

    run_name = str(wandb_cfg.get("run_name") or train_cfg.get("run_name") or f"sup-posttrain-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    ckpt_dir = _resolve_checkpoint_dir(ckpt_cfg.get("root_dir", "checkpoints/SUP"), run_name)
    if not args.no_reset:
        _reset_checkpoint_dir(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    periodic_ckpt_dir = ckpt_dir / "periodic"
    periodic_ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train.log"

    def _log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    wandb_logger = None
    if bool(wandb_cfg.get("enabled", False)):
        wandb_logger = init_wandb(
            str(wandb_cfg.get("project") or "simvq-posttrain"),
            run_name,
            cfg,
            api_key=wandb_cfg.get("api_key"),
            entity=wandb_cfg.get("entity"),
        )
        if wandb_logger is not None:
            _log(f"[setup] wandb enabled project={wandb_cfg.get('project') or 'simvq-posttrain'} run={run_name}")

    _WANDB_STOP = object()
    wandb_queue: queue.Queue[tuple[Dict[str, float], int] | object] | None = None
    wandb_thread: threading.Thread | None = None
    wandb_drop_count = 0
    wandb_finished = False
    if wandb_logger is not None:
        wandb_queue = queue.Queue(maxsize=1024)

        def _wandb_worker() -> None:
            while True:
                item = wandb_queue.get()
                if item is _WANDB_STOP:
                    return
                metrics, step = item  # type: ignore[misc]
                wandb_logger.log(metrics, step=step)

        wandb_thread = threading.Thread(target=_wandb_worker, daemon=True)
        wandb_thread.start()

    def _log_wandb(metrics: Dict[str, float], step: int) -> None:
        nonlocal wandb_drop_count
        if wandb_logger is None or not metrics:
            return
        if wandb_queue is None:
            wandb_logger.log(metrics, step=step)
            return
        try:
            wandb_queue.put_nowait((metrics, step))
        except queue.Full:
            wandb_drop_count += 1
            if wandb_drop_count in (1, 10, 100):
                _log(f"[warn] wandb queue full, dropping metrics (dropped={wandb_drop_count}).")

    def _finish_wandb() -> None:
        nonlocal wandb_finished
        if wandb_finished:
            return
        wandb_finished = True
        if wandb_queue is not None:
            try:
                wandb_queue.put(_WANDB_STOP, timeout=5.0)
            except Exception:
                pass
        if wandb_thread is not None and wandb_thread.is_alive():
            wandb_thread.join(timeout=10.0)
        if wandb_logger is not None:
            wandb_logger.finish()

    if wandb_logger is not None:
        atexit.register(_finish_wandb)

    _write_json(ckpt_dir / "resolved_config.json", cfg)
    _log(f"[setup] config={config_path}")
    _log(f"[setup] dataset={dataset_dir} bucket={bucket_chunks} batch_size={batch_size} devices={device_count}")

    ds = PosttrainShardDataset.from_dir(dataset_dir)
    if bucket_chunks not in ds.buckets:
        raise ValueError(f"Bucket {bucket_chunks} not found in dataset; available={ds.buckets}")

    generator = build_audio_model(model_cfg)
    rng = jax.random.PRNGKey(int(cfg.get("seed", train_cfg.get("seed", 42))))
    rng, init_rng, step_rng = jax.random.split(rng, 3)
    chunk_size = int(data_cfg.get("chunk_size_samples", 6144))
    gen_state, _ = create_generator_state(
        init_rng,
        generator,
        (per_device_reads * bucket_chunks, chunk_size),
        float(train_cfg.get("learning_rate", 1e-6)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        group_lrs={str(k): float(v) for k, v in dict(cfg.get("optim", {}).get("lr_multipliers") or {}).items()},
    )
    base_vq_vars = gen_state.vq_vars
    if not isinstance(gen_state.params, FrozenDict):
        gen_state = gen_state.replace(params=freeze(gen_state.params))

    resume_from = str(args.resume_from) if args.resume_from is not None else ckpt_cfg.get("resume_from")
    resume_from = _resolve_path(resume_from, base=REPO_ROOT) if resume_from else None
    if not resume_from:
        raise ValueError("checkpoint.resume_from or --resume-from is required for SUP post-training.")
    if not Path(resume_from).exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
    restore_target = {"gen": gen_state}
    restored = flax_ckpt.restore_checkpoint(ckpt_dir=resume_from, target=restore_target)
    restored_state = restored.get("gen") if isinstance(restored, dict) and "gen" in restored else restored
    gen_state = _with_vq_default(restored_state, base_vq_vars)
    gen_state = _host_broadcast_tree_for_pmap(gen_state, device_count)
    _log(f"[resume] restored {resume_from} step={_state_step_as_int(gen_state)}")

    dorado_model = _resolve_path(dorado_cfg.get("model_path"), base=cfg_dir)
    if not dorado_model:
        raise ValueError("dorado.model_path is required.")
    _log(f"[setup] loading Dorado model {dorado_model}")
    dorado_state = load_dorado_encoder_state(model_path=dorado_model, layers=("crf_logits",))

    loss_weights = _loss_weights(train_cfg, data_cfg)
    stft_loss_scales = _normalize_stft_loss_scales(train_cfg.get("stft_loss"))
    seq_loss_weight = float(train_cfg.get("seq_loss_weight", 0.01))
    output_length = int(bucket_chunks * chunk_size)
    epochs = max(1, int(train_cfg.get("epochs", 1)))
    log_every = max(1, int(train_cfg.get("log_every_steps", 10)))
    checkpoint_every = max(0, int(ckpt_cfg.get("every_steps", 100)))
    max_steps = args.max_steps if args.max_steps is not None else train_cfg.get("max_steps")
    max_steps = None if max_steps in (None, "", 0) else int(max_steps)

    @partial(jax.pmap, axis_name="data", devices=devices, in_axes=(0, 0, 0), out_axes=(0, 0))
    def _p_train_step(state, batch, apply_rngs):
        grads, logs, _new_vq = compute_posttrain_grads(
            state,
            batch,
            apply_rngs,
            loss_weights,
            dorado_state=dorado_state,
            stft_loss_scales=stft_loss_scales,
            seq_loss_weight=seq_loss_weight,
            output_length=output_length,
        )
        grads = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, "data"), grads)
        logs = {key: jax.lax.pmean(value, "data") for key, value in logs.items()}
        state = state.apply_gradients(grads=grads, vq_vars=state.vq_vars)
        return state, logs

    def _save_checkpoint(step: int, *, keep: int = 3) -> str:
        save_state = flax_jax_utils.unreplicate(gen_state)
        path = flax_ckpt.save_checkpoint(
            str(periodic_ckpt_dir),
            target={"gen": save_state},
            step=step,
            overwrite=False,
            keep=keep,
        )
        return str(path)

    _log("[warmup] first train step will compile SimVQGAN + Dorado SUP + CRF NLL")
    global_step = _state_step_as_int(gen_state)
    last_logs: dict[str, Any] | None = None
    started_at = time.time()
    for epoch in range(epochs):
        iterator = ds.batches(
            bucket_chunks=bucket_chunks,
            batch_size=batch_size,
            shuffle_shards=bool(train_cfg.get("shuffle_shards", True)),
            shuffle_rows=bool(train_cfg.get("shuffle_rows", True)),
            seed=int(train_cfg.get("seed", 42)) + epoch,
            drop_last=True,
        )
        epoch_steps = 0
        for host_batch_raw in iterator:
            host_batch = _numeric_batch(host_batch_raw)
            sharded_batch = _shard_batch(host_batch, device_count=device_count)
            step_rng, use_rng = jax.random.split(step_rng)
            apply_rngs = jax.random.split(use_rng, device_count)
            step_start = time.time()
            gen_state, logs = _p_train_step(gen_state, sharded_batch, apply_rngs)
            # Sync only the reduced logs; this keeps progress visible and catches NaNs early.
            logs_host = jax.tree_util.tree_map(lambda x: x[0], logs)
            last_logs = logs_host
            global_step = _state_step_as_int(gen_state)
            epoch_steps += 1
            if global_step % log_every == 0 or epoch_steps == 1:
                floats = {key: _value_to_float(value) for key, value in logs_host.items()}
                step_s = time.time() - step_start
                msg = "[step {} epoch {}] ".format(global_step, epoch + 1) + ", ".join(
                    f"{key}={value:.5f}" for key, value in sorted(floats.items())
                    if key in {"total_loss", "reconstruct_loss", "seq_nll", "seq_loss", "mean_logit_length", "mean_truth_length"}
                )
                msg += f", step_s={step_s:.2f}"
                _log(msg)
                if not all(np.isfinite(value) for value in floats.values()):
                    raise FloatingPointError(f"Non-finite training logs at step {global_step}: {floats}")
                wandb_metrics = dict(floats)
                wandb_metrics.update(
                    {
                        "epoch": float(epoch + 1),
                        "epoch_step": float(epoch_steps),
                        "step_s": float(step_s),
                    }
                )
                _log_wandb(wandb_metrics, global_step)
            if checkpoint_every > 0 and global_step % checkpoint_every == 0:
                saved = _save_checkpoint(global_step)
                _log(f"[ckpt] saved {saved}")
            if max_steps is not None and global_step >= max_steps:
                break
        _log(f"[epoch {epoch + 1}] steps={epoch_steps}")
        if max_steps is not None and global_step >= max_steps:
            break
    saved = _save_checkpoint(global_step, keep=5)
    elapsed = time.time() - started_at
    _log(f"[done] step={global_step} elapsed_s={elapsed:.1f} saved={saved}")
    if last_logs is not None:
        _write_json(
            ckpt_dir / "last_logs.json",
            {key: _value_to_float(value) for key, value in last_logs.items()},
        )
    _finish_wandb()


if __name__ == "__main__":
    main()
