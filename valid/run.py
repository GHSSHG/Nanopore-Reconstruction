#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from codec.runtime import configure_runtime_env, enable_jax_compilation_cache

configure_runtime_env()
enable_jax_compilation_cache()

from valid.common import (  # noqa: E402
    TRUTH_MODE_ANALYSIS_HAC_PROXY,
    default_tmp_path,
    load_json,
    read_fastq,
    resolve_dorado_bin,
    resolve_repo_path,
    run_dorado,
    sanitize_tag,
    write_json,
    write_jsonl,
)
from valid.manifests import build_true_valid_manifest, load_manifest_reads, load_or_create_regular_manifest  # noqa: E402
from valid.metrics import (  # noqa: E402
    compare_fastq_to_truth,
    compute_group_summaries,
    compute_regular_metrics,
    compute_summary_delta,
)
from valid.pod5_reconstruct import (  # noqa: E402
    DIVEQ_VALID_MODEL_APPLY_MODE,
    DIVEQ_VALID_RNG_SEED,
    RECON_MODE_OVERLAP,
    SUPPORTED_RECON_MODES,
    CheckpointReconstructor,
    finalize_reconstruction,
    load_prepared_reads,
    load_prepared_reads_pad_last,
    load_prepared_reads_shift_last,
    reconstruction_per_read_metrics,
    resolve_recon_hop,
    resolve_segment_hop_samples,
    resolve_segment_samples,
    resolve_split_files,
    write_generated_pod5,
    write_selected_real_pod5,
    write_selected_real_pod5_shift_last,
    VALID_RECON_DEVICE_COUNT,
)


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return None
    return int(raw)


def _env_flag(name: str) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return False
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _positive_int(value: int | None, default: int) -> int:
    if value is None:
        return int(default)
    return max(1, int(value))


def _default_manifest_path(mode: str, *, split: str, num_reads: int) -> Path:
    return default_tmp_path("valid_runs", "manifests", f"{mode}_{sanitize_tag(split)}_{int(num_reads)}reads.json")


def _default_output_dir(mode: str, *, label: str | None, checkpoint: Path | None, num_reads: int) -> Path:
    if label:
        tag = sanitize_tag(label)
    elif checkpoint is not None:
        parent = checkpoint.parent.parent.name if checkpoint.parent.name == "periodic" else checkpoint.parent.name
        tag = sanitize_tag(f"{parent}_{checkpoint.name}")
    else:
        tag = "run"
    return default_tmp_path("valid_runs", f"{tag}_{mode}_{int(num_reads)}reads")


def _ensure_tmp_path(path: Path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    tmp_root = default_tmp_path().resolve()
    if resolved != tmp_root and tmp_root not in resolved.parents:
        raise SystemExit(f"{description} must be under {tmp_root}")
    return resolved


def _resolve_input_files(config: dict[str, Any], split: str, source_pod5: Path | None) -> list[Path]:
    if source_pod5 is not None:
        resolved = source_pod5.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Source POD5 not found: {resolved}")
        return [resolved]
    return resolve_split_files(config, split)


def _resolve_regular_split(config: dict[str, Any], requested_split: str) -> str:
    data_cfg = dict(config.get("data") or {})
    requested_split = str(requested_split or "").strip().lower()
    if requested_split in data_cfg:
        return requested_split
    for fallback in ("valid", "valid_fc01", "train"):
        if fallback in data_cfg:
            return fallback
    raise RuntimeError(
        f"Unable to resolve validation split {requested_split!r}; looked for it plus 'valid', 'valid_fc01', and 'train'."
    )


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    chunk_batch_default = _env_int("CHUNK_BATCH_SIZE")
    if chunk_batch_default is None:
        chunk_batch_default = _env_int("MICROBATCH") or 128
    source_pod5_default = os.environ.get("SOURCE_POD5")
    manifest_path_default = os.environ.get("MANIFEST_PATH")

    parser.add_argument("--config", type=Path, default=os.environ.get("CONFIG_PATH"), required=False)
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("CHECKPOINT_PATH"), required=False)
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("OUTPUT_DIR"), required=False)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--num-reads", type=int, default=int(os.environ.get("NUM_READS", "2000")))
    parser.add_argument("--min-read-length", type=int, default=int(os.environ.get("MIN_READ_LENGTH", "6144")))
    parser.add_argument("--max-read-length", type=int, default=_env_int("MAX_READ_LENGTH"))
    parser.add_argument("--microbatch", type=int, default=None, help="Legacy alias for --chunk-batch-size.")
    parser.add_argument("--chunk-batch-size", type=int, default=chunk_batch_default)
    parser.add_argument("--data-split", type=str, default=os.environ.get("DATA_SPLIT", "valid"))
    parser.add_argument("--source-pod5", type=Path, default=(Path(source_pod5_default) if source_pod5_default else None))
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=(Path(manifest_path_default) if manifest_path_default else None),
    )
    parser.add_argument("--recon-mode", type=str, default=os.environ.get("RECON_MODE", RECON_MODE_OVERLAP))
    parser.add_argument("--hop-samples", type=int, default=_env_int("HOP_SAMPLES"))
    parser.add_argument("--dorado-bin", type=str, default=os.environ.get("DORADO_BIN", "dorado"))
    parser.add_argument("--dorado-model", type=str, default=os.environ.get("DORADO_MODEL"))
    parser.add_argument("--dorado-device", type=str, default=os.environ.get("DORADO_DEVICE", "cuda:all"))
    parser.add_argument("--trim-mode", type=str, default=os.environ.get("TRIM_MODE", "drop"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SimVQGAN validation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    regular = subparsers.add_parser("regular", help="Run ordinary valid: original vs reconstructed.")
    _add_common_run_args(regular)
    regular.add_argument("--prepare-manifest-only", action="store_true", default=_env_flag("PREPARE_MANIFEST_ONLY"))
    regular.add_argument("--skip-dorado", action="store_true", default=_env_flag("SKIP_DORADO"))

    true = subparsers.add_parser("true", help="Run true valid against a truth manifest.")
    _add_common_run_args(true)
    true.add_argument("--tail-chunk-mode", type=str, default="shift_last")

    manifest = subparsers.add_parser("build-true-manifest", help="Build a true-valid manifest and truth FASTQ.")
    manifest.add_argument("--analysis-root", type=Path, default=Path("/data/nanopore/hereditary_cancer_2025.09/analysis"))
    manifest.add_argument("--raw-root", type=Path, default=Path("/data/nanopore/hereditary_cancer_2025.09/raw"))
    manifest.add_argument("--output-dir", type=Path, default=default_tmp_path("valid_runs", "manifests", "true_valid"))
    manifest.add_argument("--fc", type=str, default="FC01")
    manifest.add_argument("--barcodes", type=str, default="barcode01,barcode02,barcode03")
    manifest.add_argument("--selection-mode", type=str, default="global_random")
    manifest.add_argument("--quality-filter-mode", type=str, default="len_only")
    manifest.add_argument("--target-total-reads", type=int, default=2000)
    manifest.add_argument("--target-per-barcode", type=int, default=200)
    manifest.add_argument("--random-seed", type=int, default=42)
    manifest.add_argument("--min-read-length", type=int, default=6144)
    manifest.add_argument("--min-acc", type=float, default=99.0)
    manifest.add_argument("--min-coverage", type=float, default=98.0)
    manifest.add_argument("--min-mean-quality", type=float, default=16.0)
    manifest.add_argument("--max-qstart", type=int, default=100)
    manifest.add_argument("--max-tail", type=int, default=100)
    manifest.add_argument("--truth-mode", type=str, default=TRUTH_MODE_ANALYSIS_HAC_PROXY)
    manifest.add_argument("--dry-run", action="store_true")

    return parser


def _normalize_common_args(args: argparse.Namespace) -> None:
    if args.config is None:
        raise SystemExit("--config/CONFIG_PATH is required")
    if args.checkpoint is None and not getattr(args, "prepare_manifest_only", False):
        raise SystemExit("--checkpoint/CHECKPOINT_PATH is required")
    if args.microbatch is not None:
        args.chunk_batch_size = args.microbatch
    args.chunk_batch_size = _positive_int(args.chunk_batch_size, 128)
    args.recon_mode = str(args.recon_mode).strip().lower() or RECON_MODE_OVERLAP
    if args.recon_mode not in SUPPORTED_RECON_MODES:
        raise SystemExit(f"Unsupported --recon-mode={args.recon_mode!r}; choose from {sorted(SUPPORTED_RECON_MODES)}")
    args.trim_mode = str(args.trim_mode).strip().lower() or "drop"
    if args.trim_mode not in {"drop", "pad"}:
        raise SystemExit("Unsupported --trim-mode; choose from ['drop', 'pad'].")


def _prepare_reconstruction(
    *,
    config: dict[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    manifest_reads: list[Any],
    split: str,
    chunk_batch_size: int,
    recon_mode: str,
    hop_override: int | None,
    trim_mode: str,
    tail_chunk_mode: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path, dict[str, int]]:
    segment_samples = resolve_segment_samples(config, split)
    configured_hop_samples = resolve_segment_hop_samples(config, split)
    hop_samples = resolve_recon_hop(recon_mode, segment_samples, configured_hop_samples, hop_override)

    pod5_dir = output_dir / "pod5"
    metrics_dir = output_dir / "metrics"
    pod5_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    original_pod5_path = pod5_dir / "original_selected.pod5"
    if tail_chunk_mode in {"shift_last", "pad_last"}:
        original_written, trimmed_lengths = write_selected_real_pod5_shift_last(
            manifest_reads=manifest_reads,
            output_path=original_pod5_path,
        )
    else:
        original_written, trimmed_lengths = write_selected_real_pod5(
            manifest_reads=manifest_reads,
            output_path=original_pod5_path,
            segment_samples=segment_samples,
            hop_samples=hop_samples,
            trim_mode=trim_mode,
        )
    if original_written <= 0:
        raise RuntimeError("Selected validation POD5 is empty.")

    if tail_chunk_mode == "shift_last":
        prepared_reads, chunk_specs, chunk_inputs = load_prepared_reads_shift_last(
            source_pod5=original_pod5_path,
            segment_samples=segment_samples,
            hop_samples=hop_samples,
            recon_mode=recon_mode,
        )
    elif tail_chunk_mode == "pad_last":
        prepared_reads, chunk_specs, chunk_inputs = load_prepared_reads_pad_last(
            source_pod5=original_pod5_path,
            segment_samples=segment_samples,
            hop_samples=hop_samples,
            recon_mode=recon_mode,
        )
    else:
        prepared_reads, chunk_specs, chunk_inputs = load_prepared_reads(
            source_pod5=original_pod5_path,
            segment_samples=segment_samples,
            hop_samples=hop_samples,
            recon_mode=recon_mode,
        )
    reconstructor = CheckpointReconstructor(
        model_cfg=dict(config.get("model") or {}),
        checkpoint_path=checkpoint_path,
        chunk_batch_size=chunk_batch_size,
    )
    chunk_outputs = reconstructor.reconstruct_chunks(chunk_inputs)
    reconstruction_summary = finalize_reconstruction(
        reads=prepared_reads,
        chunk_specs=chunk_specs,
        chunk_inputs=chunk_inputs,
        chunk_outputs=chunk_outputs,
        recon_mode=recon_mode,
    )

    generated_pod5_path = pod5_dir / "generated_selected.pod5"
    generated_written = write_generated_pod5(output_path=generated_pod5_path, reads=prepared_reads)
    if generated_written <= 0:
        raise RuntimeError("Generated validation POD5 is empty.")

    recon_metrics_path = metrics_dir / "reconstruction_per_read_metrics.jsonl"
    write_jsonl(recon_metrics_path, reconstruction_per_read_metrics(prepared_reads))

    run_meta = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "data_split": split,
        "selected_trimmed_pod5_read_count": int(original_written),
        "generated_pod5_read_count": int(generated_written),
        "segment_samples": int(segment_samples),
        "recon_mode": recon_mode,
        "recon_chunk_size": int(segment_samples),
        "recon_hop_size": int(hop_samples),
        "recon_overlap_size": int(segment_samples - hop_samples),
        "chunk_batch_size": int(chunk_batch_size),
        "effective_chunk_batch_size": int(reconstructor.chunk_batch_size),
        "per_device_chunk_batch_size": int(reconstructor.per_device_chunk_batch_size),
        "recon_device_count": int(reconstructor.device_count),
        "recon_parallelism": reconstructor.parallelism,
        "required_recon_device_count": int(VALID_RECON_DEVICE_COUNT),
        "trim_mode": trim_mode,
        "tail_chunk_mode": tail_chunk_mode,
        "model_apply_mode": DIVEQ_VALID_MODEL_APPLY_MODE,
        "model_apply_train": True,
        "model_apply_rng_seed": int(DIVEQ_VALID_RNG_SEED),
    }
    paths = {
        "original_pod5": str(original_pod5_path),
        "generated_pod5": str(generated_pod5_path),
        "reconstruction_per_read_metrics": str(recon_metrics_path),
    }
    return run_meta, reconstruction_summary, original_pod5_path, generated_pod5_path, recon_metrics_path, trimmed_lengths


def _resolve_dorado_model(config: dict[str, Any], config_path: Path, override: str | None) -> Path:
    validation_cfg = dict(config.get("validation") or {})
    validation_dorado_cfg = dict(validation_cfg.get("dorado") or {})
    top_level_dorado_cfg = dict(config.get("dorado") or {})
    dorado_model_path = (
        override
        or validation_cfg.get("dorado_model_path")
        or validation_dorado_cfg.get("model_path")
        or top_level_dorado_cfg.get("model_path")
    )
    dorado_model = resolve_repo_path(dorado_model_path, config_path.parent)
    if dorado_model is None:
        raise RuntimeError("No Dorado model configured. Set --dorado-model, DORADO_MODEL, or validation.dorado_model_path.")
    return dorado_model


def run_regular(args: argparse.Namespace) -> None:
    _normalize_common_args(args)
    started_at = time.time()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = None if args.checkpoint is None else Path(args.checkpoint).expanduser().resolve()
    source_pod5 = None if args.source_pod5 is None else Path(args.source_pod5).expanduser().resolve()
    split = str(args.data_split).strip().lower() or "valid"
    num_reads = _positive_int(args.num_reads, 2000)
    min_read_length = _positive_int(args.min_read_length, 6144)
    max_read_length = None if args.max_read_length is None else int(args.max_read_length)

    config = load_json(config_path)
    config["_config_dir"] = str(config_path.parent.resolve())
    recon_split = _resolve_regular_split(config, split)
    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path is not None
        else _default_manifest_path("regular", split=split, num_reads=num_reads).resolve()
    )
    if not manifest_path.exists():
        manifest_path = _ensure_tmp_path(manifest_path, "--manifest-path for a new regular manifest")
    if manifest_path.exists() and source_pod5 is None:
        manifest_payload = load_json(manifest_path)
        split_files = [Path(path).expanduser().resolve() for path in manifest_payload.get("source_files", [])]
        if not split_files:
            split_files = sorted(
                {
                    Path(item["source_file"]).expanduser().resolve()
                    for item in manifest_payload.get("selected_reads", [])
                }
            )
        if not split_files:
            raise RuntimeError(f"Existing manifest does not list source files: {manifest_path}")
    else:
        split_files = _resolve_input_files(config, recon_split, source_pod5)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir("regular", label=args.label, checkpoint=checkpoint_path, num_reads=num_reads).resolve()
    )
    output_dir = _ensure_tmp_path(output_dir, "--output-dir")

    manifest_reads, manifest_path, manifest_warnings = load_or_create_regular_manifest(
        manifest_path=manifest_path,
        split=split,
        files=split_files,
        num_reads=num_reads,
        min_read_length=min_read_length,
        max_read_length=max_read_length,
    )

    if args.prepare_manifest_only:
        payload = {
            "status": "manifest_ready",
            "manifest_path": str(manifest_path),
            "requested_read_count": int(num_reads),
            "selected_read_count": int(len(manifest_reads)),
            "min_read_length": int(min_read_length),
            "max_read_length": None if max_read_length is None else int(max_read_length),
            "source_file_count": len(split_files),
            "source_files": [str(path) for path in split_files],
            "warnings": manifest_warnings,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        return

    if checkpoint_path is None:
        raise RuntimeError("Checkpoint path is required for reconstruction")
    output_dir.mkdir(parents=True, exist_ok=True)
    fastq_dir = output_dir / "fastq"
    metrics_dir = output_dir / "metrics"
    fastq_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    run_meta, reconstruction_summary, real_pod5_path, generated_pod5_path, recon_metrics_path, trimmed_lengths = (
        _prepare_reconstruction(
            config=config,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            manifest_reads=manifest_reads,
            split=recon_split,
            chunk_batch_size=args.chunk_batch_size,
            recon_mode=args.recon_mode,
            hop_override=args.hop_samples,
            trim_mode=args.trim_mode,
            tail_chunk_mode="drop",
        )
    )

    metric_summary: dict[str, Any] = {}
    fastq_per_read: list[dict[str, Any]] = []
    real_fastq_path = fastq_dir / "original.fastq"
    generated_fastq_path = fastq_dir / "generated.fastq"
    dorado_bin_text = None
    dorado_model_text = None
    if not args.skip_dorado:
        dorado_model = _resolve_dorado_model(config, config_path, args.dorado_model)
        dorado_bin = resolve_dorado_bin(args.dorado_bin, cfg_dir=config_path.parent, dorado_model=dorado_model)
        run_dorado(
            dorado_bin=str(dorado_bin),
            dorado_model=str(dorado_model),
            pod5_path=real_pod5_path,
            out_fastq=real_fastq_path,
            device=str(args.dorado_device),
        )
        run_dorado(
            dorado_bin=str(dorado_bin),
            dorado_model=str(dorado_model),
            pod5_path=generated_pod5_path,
            out_fastq=generated_fastq_path,
            device=str(args.dorado_device),
        )
        metric_summary, fastq_per_read = compute_regular_metrics(real_fastq_path, generated_fastq_path)
        write_jsonl(metrics_dir / "per_read_metrics.jsonl", fastq_per_read)
        dorado_bin_text = str(dorado_bin)
        dorado_model_text = str(dorado_model)

    finished_at = time.time()
    summary = {
        "status": "ok",
        "mode": "regular",
        **run_meta,
        "source_file_count": len(split_files),
        "source_files": [str(path) for path in split_files],
        "source_pod5": (None if source_pod5 is None else str(source_pod5)),
        "requested_read_count": int(num_reads),
        "selected_read_count": int(len(manifest_reads)),
        "min_read_length": int(min_read_length),
        "max_read_length": None if max_read_length is None else int(max_read_length),
        "skip_dorado": bool(args.skip_dorado),
        "dorado_bin": dorado_bin_text,
        "dorado_model": dorado_model_text,
        "dorado_device": None if args.skip_dorado else str(args.dorado_device),
        "manifest_path": str(manifest_path),
        "started_at_unix": float(started_at),
        "finished_at_unix": float(finished_at),
        "elapsed_seconds": float(finished_at - started_at),
        "paths": {
            "original_pod5": str(real_pod5_path),
            "generated_pod5": str(generated_pod5_path),
            "reconstruction_per_read_metrics": str(recon_metrics_path),
            "original_fastq": (None if args.skip_dorado else str(real_fastq_path)),
            "generated_fastq": (None if args.skip_dorado else str(generated_fastq_path)),
            "per_read_metrics": (None if args.skip_dorado else str(metrics_dir / "per_read_metrics.jsonl")),
        },
        "selected_trimmed_lengths": trimmed_lengths,
        "warnings": manifest_warnings,
    }
    summary.update(reconstruction_summary)
    summary.update(metric_summary)
    write_json(metrics_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def _resolve_true_split(config: dict[str, Any], manifest_payload: dict[str, Any], requested_split: str) -> str:
    data_cfg = dict(config.get("data") or {})
    requested_split = str(requested_split or "").strip().lower()
    if requested_split and requested_split in data_cfg:
        return requested_split
    fc_split = str(manifest_payload.get("fc") or "FC01").strip().lower()
    fc_split = "valid_" + fc_split if not fc_split.startswith("valid_") else fc_split
    if fc_split in data_cfg:
        return fc_split
    if "valid_fc01" in data_cfg:
        return "valid_fc01"
    if "valid" in data_cfg:
        return "valid"
    if "train" in data_cfg:
        return "train"
    raise RuntimeError(
        f"Unable to resolve true-valid split for manifest fc={manifest_payload.get('fc')!r}; "
        f"looked for {requested_split!r}, {fc_split!r}, 'valid_fc01', 'valid', and 'train'."
    )


def run_true(args: argparse.Namespace) -> None:
    _normalize_common_args(args)
    started_at = time.time()
    if args.manifest_path is None:
        raise SystemExit("--manifest-path is required for true valid. Use 'build-true-manifest' first.")
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    num_reads = _positive_int(args.num_reads, 2000)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir("true", label=args.label, checkpoint=checkpoint_path, num_reads=num_reads).resolve()
    )
    output_dir = _ensure_tmp_path(output_dir, "--output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(config_path)
    config["_config_dir"] = str(config_path.parent.resolve())
    manifest_payload = load_json(manifest_path)
    manifest_reads = load_manifest_reads(manifest_payload)
    if len(manifest_reads) > num_reads:
        manifest_reads = manifest_reads[:num_reads]
        manifest_payload = dict(manifest_payload)
        manifest_payload["selected_reads"] = manifest_payload.get("selected_reads", [])[:num_reads]

    truth_path = Path(manifest_payload["paths"]["truth_fastq"]).expanduser().resolve()
    truth_entries = read_fastq(truth_path)
    split = _resolve_true_split(config, manifest_payload, args.data_split)
    tail_chunk_mode = str(args.tail_chunk_mode).strip().lower() or "shift_last"
    if tail_chunk_mode not in {"drop", "shift_last", "pad_last"}:
        raise RuntimeError("Unsupported --tail-chunk-mode; choose from ['drop', 'shift_last', 'pad_last'].")

    fastq_dir = output_dir / "fastq"
    metrics_dir = output_dir / "metrics"
    fastq_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    run_meta, reconstruction_summary, original_pod5_path, generated_pod5_path, recon_metrics_path, trimmed_lengths = (
        _prepare_reconstruction(
            config=config,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            manifest_reads=manifest_reads,
            split=split,
            chunk_batch_size=args.chunk_batch_size,
            recon_mode=args.recon_mode,
            hop_override=args.hop_samples,
            trim_mode=args.trim_mode,
            tail_chunk_mode=tail_chunk_mode,
        )
    )

    dorado_model = _resolve_dorado_model(config, config_path, args.dorado_model)
    dorado_bin = resolve_dorado_bin(args.dorado_bin, cfg_dir=config_path.parent, dorado_model=dorado_model)

    original_fastq_path = fastq_dir / "original.fastq"
    generated_fastq_path = fastq_dir / "generated.fastq"
    run_dorado(
        dorado_bin=str(dorado_bin),
        dorado_model=str(dorado_model),
        pod5_path=original_pod5_path,
        out_fastq=original_fastq_path,
        device=args.dorado_device,
    )
    run_dorado(
        dorado_bin=str(dorado_bin),
        dorado_model=str(dorado_model),
        pod5_path=generated_pod5_path,
        out_fastq=generated_fastq_path,
        device=args.dorado_device,
    )

    original_vs_truth_summary, original_vs_truth_records = compare_fastq_to_truth(
        predicted_fastq=original_fastq_path,
        truth_entries=truth_entries,
    )
    generated_vs_truth_summary, generated_vs_truth_records = compare_fastq_to_truth(
        predicted_fastq=generated_fastq_path,
        truth_entries=truth_entries,
    )
    original_vs_generated_summary, original_vs_generated_records = compute_regular_metrics(
        original_fastq_path,
        generated_fastq_path,
    )

    write_jsonl(metrics_dir / "original_vs_truth.jsonl", original_vs_truth_records)
    write_jsonl(metrics_dir / "generated_vs_truth.jsonl", generated_vs_truth_records)
    write_jsonl(metrics_dir / "original_vs_generated.jsonl", original_vs_generated_records)

    per_barcode_summary = compute_group_summaries(
        manifest_payload=manifest_payload,
        original_records=original_vs_truth_records,
        generated_records=generated_vs_truth_records,
    )
    write_json(metrics_dir / "per_barcode_summary.json", per_barcode_summary)

    finished_at = time.time()
    summary = {
        "status": "ok",
        "mode": "true",
        **run_meta,
        "manifest_path": str(manifest_path),
        "truth_mode": str(manifest_payload.get("truth_mode") or TRUTH_MODE_ANALYSIS_HAC_PROXY),
        "selected_read_count": int(len(manifest_reads)),
        "dorado_bin": str(dorado_bin),
        "dorado_model": str(dorado_model),
        "dorado_device": str(args.dorado_device),
        "started_at_unix": float(started_at),
        "finished_at_unix": float(finished_at),
        "elapsed_seconds": float(finished_at - started_at),
        "paths": {
            "original_pod5": str(original_pod5_path),
            "generated_pod5": str(generated_pod5_path),
            "original_fastq": str(original_fastq_path),
            "generated_fastq": str(generated_fastq_path),
            "reconstruction_per_read_metrics": str(recon_metrics_path),
            "original_vs_truth": str(metrics_dir / "original_vs_truth.jsonl"),
            "generated_vs_truth": str(metrics_dir / "generated_vs_truth.jsonl"),
            "original_vs_generated": str(metrics_dir / "original_vs_generated.jsonl"),
            "per_barcode_summary": str(metrics_dir / "per_barcode_summary.json"),
        },
        "selected_trimmed_lengths": trimmed_lengths,
        "original_vs_truth": original_vs_truth_summary,
        "generated_vs_truth": generated_vs_truth_summary,
        "generated_minus_original_vs_truth": compute_summary_delta(generated_vs_truth_summary, original_vs_truth_summary),
        "original_vs_generated": original_vs_generated_summary,
        "warnings": manifest_payload.get("warnings", []),
    }
    summary.update(reconstruction_summary)
    write_json(metrics_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def run_build_true_manifest(args: argparse.Namespace) -> None:
    barcodes = [item.strip() for item in str(args.barcodes).split(",") if item.strip()]
    output_dir = _ensure_tmp_path(args.output_dir, "--output-dir")
    payload = build_true_valid_manifest(
        analysis_root=args.analysis_root,
        raw_root=args.raw_root,
        output_dir=output_dir,
        fc=args.fc,
        barcodes=barcodes,
        selection_mode=args.selection_mode,
        quality_filter_mode=args.quality_filter_mode,
        target_total_reads=args.target_total_reads,
        target_per_barcode=args.target_per_barcode,
        random_seed=args.random_seed,
        min_read_length=args.min_read_length,
        min_acc=args.min_acc,
        min_coverage=args.min_coverage,
        min_mean_quality=args.min_mean_quality,
        max_qstart=args.max_qstart,
        max_tail=args.max_tail,
        truth_mode=args.truth_mode,
        dry_run=args.dry_run,
    )
    summary_payload = dict(payload)
    selected_reads = summary_payload.pop("selected_reads", [])
    summary_payload["selected_read_examples"] = selected_reads[:3]
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "regular":
        run_regular(args)
    elif args.command == "true":
        run_true(args)
    elif args.command == "build-true-manifest":
        run_build_true_manifest(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
