#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from codec.runtime import configure_runtime_env, enable_jax_compilation_cache

configure_runtime_env()

from jax import config as _jax_config

_jax_config.update("jax_default_matmul_precision", "high")

enable_jax_compilation_cache()

from codec.data import Pod5ChunkDataset
from codec.train import train_model_from_pod5
from codec.utils import discover_pod5_files, init_wandb

LOSS_WEIGHT_KEYS = frozenset(
    {
        "time_l1",
        "pa_l1",
        "diff1_l1",
        "diff2_l1",
        "small_stft_logmag_l1",
        "medium_stft_logmag_l1",
        "large_stft_logmag_l1",
        "complex_stft_l1",
        "lowpass_l1_200hz",
        "lowpass_l1_500hz",
        "lowpass_l1_1000hz",
        "residual_energy_l2",
        "vq_codebook",
        "vq_commitment",
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the NanoRecon codec (JAX/Flax) from a JSON config")
    p.add_argument("--config", type=Path, default=_ROOT / "configs" / "offline-high.json")
    p.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size (global, all GPUs)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="Override train.learning_rate")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--ckpt-dir", type=Path, default=None, help="Checkpoint directory (default: checkpoint.root_dir/run name)")
    p.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from (overrides checkpoint.resume_from)")
    p.add_argument("--log-every-steps", type=int, default=None)
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--max-steps", type=int, default=None, help="Cap on global training steps (for quick tests)")
    p.add_argument("--max-steps-per-epoch", type=int, default=None, help="Cap on steps per epoch (for quick tests)")
    return p.parse_args()


def _resolve_path(value: str, base: Path) -> Path:
    expanded = os.path.expandvars(str(value))
    if "${" in expanded:
        raise ValueError(
            f"Path {value!r} contains an unset environment variable. Set DATA_PATH to the directory "
            "that contains hereditary_cancer_2025.09, or edit data.root in the config."
        )
    path = Path(expanded).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def main() -> None:
    args = parse_args()
    cfg_path = args.config if args.config.is_absolute() else (_ROOT / args.config).resolve()
    cfg: Dict[str, Any] = json.loads(cfg_path.read_text())
    train_cfg, data_cfg, ckpt_cfg = cfg["train"], cfg["data"], cfg.get("checkpoint", {})
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))

    weights = dict(train_cfg.get("loss_weights", {}))
    unknown = sorted(set(weights) - LOSS_WEIGHT_KEYS)
    if unknown:
        raise ValueError(f"Unsupported train.loss_weights keys: {unknown}")
    loss_weights = {k: float(weights.get(k, 0.0)) for k in LOSS_WEIGHT_KEYS}
    loss_weights["pa_l1_scale"] = float(train_cfg.get("loss", {}).get("pa_l1_scale", 50.0))
    loss_weights["sample_rate"] = float(data_cfg["sample_rate"])
    stft_loss_scales = tuple(
        (int(s["n_fft"]), int(s["win_length"]), int(s["hop_length"])) for s in train_cfg["stft_loss"]["scales"]
    )

    files = discover_pod5_files(_resolve_path(data_cfg["root"], cfg_path.parent), data_cfg.get("subdirs"))
    ds = Pod5ChunkDataset(
        files,
        chunk_samples=int(data_cfg["segment_samples"]),
        hop_samples=int(data_cfg["segment_hop_samples"]),
        sample_rate=float(data_cfg["sample_rate"]),
        pad_short_reads=bool(data_cfg.get("pad_short_reads", True)),
        min_read_length_for_pad=int(data_cfg.get("min_read_length_for_pad", 6144)),
        shuffle_buffer=int(data_cfg.get("shuffle_buffer", 32768)),
        files_in_flight=int(data_cfg.get("files_in_flight", 8)),
        seed=seed,
    )

    wandb_cfg = cfg.get("logging", {}).get("wandb", {})
    run_name = wandb_cfg.get("run_name") or cfg_path.stem
    ckpt_dir = args.ckpt_dir or _resolve_path(ckpt_cfg.get("root_dir", "checkpoints"), _ROOT) / run_name
    ckpt_dir = ckpt_dir if ckpt_dir.is_absolute() else (_ROOT / ckpt_dir).resolve()
    resume_from = args.resume or (ckpt_cfg.get("resume_from") and _resolve_path(ckpt_cfg["resume_from"], _ROOT))
    if resume_from is None and ckpt_dir.exists() and any(ckpt_dir.iterdir()):
        raise FileExistsError(f"{ckpt_dir} is not empty; pick another run name or --ckpt-dir, or --resume.")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    wandb_logger = None
    if wandb_cfg.get("enabled", False) or args.wandb:
        wandb_logger = init_wandb(
            wandb_cfg.get("project", "Nanopore Reconstruction"),
            run_name,
            cfg,
            api_key=wandb_cfg.get("api_key"),
            entity=wandb_cfg.get("entity"),
        )
    print(f"[setup] {len(files)} POD5 files, checkpoints in {ckpt_dir}", flush=True)
    try:
        train_model_from_pod5(
            ds,
            model_cfg=cfg["model"],
            batch_size=int(args.batch_size or train_cfg["batch_size"]),
            num_epochs=int(args.epochs or train_cfg.get("epochs", 1)),
            learning_rate=float(args.lr if args.lr is not None else train_cfg["learning_rate"]),
            warmup_steps=int(train_cfg.get("warmup_steps", 0)),
            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
            lr_multipliers={str(k): float(v) for k, v in cfg.get("optim", {}).get("lr_multipliers", {}).items()},
            loss_weights=loss_weights,
            stft_loss_scales=stft_loss_scales,
            seed=seed,
            ckpt_dir=ckpt_dir,
            resume_from=resume_from,
            log_every_steps=int(args.log_every_steps or train_cfg.get("log_every_steps", 500)),
            checkpoint_every_steps=int(ckpt_cfg.get("every_steps", 10000)),
            prefetch_batches=int(train_cfg.get("prefetch_batches", 8)),
            max_steps_total=args.max_steps,
            max_steps_per_epoch=args.max_steps_per_epoch,
            wandb_logger=wandb_logger,
        )
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()


if __name__ == "__main__":
    main()
