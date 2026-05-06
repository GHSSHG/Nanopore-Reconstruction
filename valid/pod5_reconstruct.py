from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Sequence

import jax
import numpy as np
from flax.core import freeze
from flax.training import checkpoints as flax_ckpt

try:
    import pod5
    from pod5 import Writer
except Exception as exc:  # pragma: no cover - depends on runtime environment
    raise SystemExit(f"pod5 library not available: {exc}") from exc

from codec.data.normalization import normalize_to_pm1_with_stats
from codec.data.pod5_processing import CalibrationParams, NormalizationStats, denormalize_to_adc, parse_calibration
from codec.data.pod5_processing import normalize_adc_signal
from codec.models import build_audio_model
from codec.utils import discover_pod5_files
from valid.common import load_json, progress_markers, summarize
from valid.manifests import ManifestRead

RECON_MODE_DIRECT = "direct_chunk"
RECON_MODE_OVERLAP = "overlap_add"
SUPPORTED_RECON_MODES = frozenset({RECON_MODE_DIRECT, RECON_MODE_OVERLAP})
DEFAULT_OVERLAP_SIZE = 144
CONCAT_CHUNK_HOP = 11688
DIVEQ_VALID_MODEL_APPLY_MODE = "train_true"
DIVEQ_VALID_RNG_SEED = 0
VALID_RECON_DEVICE_COUNT = 4


@dataclass(frozen=True)
class ChunkSpec:
    read_index: int
    read_id: str
    chunk_index: int
    start: int
    stop: int
    center: float
    half_range: float


@dataclass
class ReconstructionRead:
    read_index: int
    source_file: str
    read_id: str
    raw_length: int
    trimmed_length: int
    chunk_count: int
    calibration: CalibrationParams = field(repr=False)
    template_read: Any = field(repr=False)
    trimmed_raw: np.ndarray = field(repr=False)
    trimmed_pa: np.ndarray = field(repr=False)
    chunk_starts: list[int] = field(repr=False)
    overlap_weights: list[np.ndarray] | None = field(default=None, repr=False)
    reconstructed_pa: np.ndarray | None = field(default=None, repr=False)
    reconstructed_adc: np.ndarray | None = field(default=None, repr=False)
    pa_acc: np.ndarray | None = field(default=None, repr=False)
    weight_acc: np.ndarray | None = field(default=None, repr=False)
    chunk_norm_mae: float = 0.0
    chunk_norm_rmse: float = 0.0
    pa_mae: float = 0.0
    pa_rmse: float = 0.0
    adc_mae: float = 0.0
    adc_rmse: float = 0.0


@dataclass(frozen=True)
class ShiftLastWindow:
    start: int
    stop: int
    valid_from: int
    valid_to: int


@dataclass(frozen=True)
class PadLastWindow:
    start: int
    stop: int
    valid_length: int
    is_padded_tail: bool = False


@dataclass(frozen=True)
class SourceChunkSpec:
    source_file: str
    source_read_id: str
    source_start: int
    source_stop: int
    source_length: int
    sample_rate_hz: float


def to_host_tree(tree: Any) -> Any:
    return jax.device_get(tree)


def host_broadcast_tree_for_pmap(tree: Any, replica_count: int) -> Any:
    host_tree = jax.device_get(tree)

    def _broadcast_leaf(value: Any) -> np.ndarray:
        arr = np.asarray(value)
        return np.broadcast_to(arr, (int(replica_count),) + arr.shape).copy()

    return jax.tree_util.tree_map(_broadcast_leaf, host_tree)


def build_model(model_cfg: dict[str, Any] | None):
    return build_audio_model(model_cfg)


def load_generator_variables(checkpoint_path: str | Path) -> dict[str, Any]:
    ckpt = flax_ckpt.restore_checkpoint(str(Path(checkpoint_path).resolve()), target=None)
    if not isinstance(ckpt, dict) or "gen" not in ckpt:
        raise RuntimeError(f"Unexpected checkpoint structure in {checkpoint_path}")
    gen_state = ckpt["gen"]
    params = None
    vq_vars = None
    if hasattr(gen_state, "params"):
        params = getattr(gen_state, "params")
        vq_vars = getattr(gen_state, "vq_vars", None)
    elif isinstance(gen_state, dict):
        params = gen_state.get("params")
        vq_vars = gen_state.get("vq_vars") or gen_state.get("vq")
    if params is None:
        raise RuntimeError(f"Checkpoint missing generator params: {checkpoint_path}")
    return {
        "params": freeze(to_host_tree(params)),
        "vq": freeze(to_host_tree(vq_vars or {})),
    }


def resolve_data_path(path_value: str | Path | None, cfg_dir: Path) -> Path:
    if path_value is None:
        return cfg_dir.resolve()
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (cfg_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def merge_split_cfg(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if override:
        merged.update(override)
    subdirs = merged.get("subdirs", ["."])
    if isinstance(subdirs, str):
        subdirs = [subdirs]
    merged["subdirs"] = list(subdirs or ["."])
    return merged


def resolve_split_cfg(config: dict[str, Any], split_name: str) -> dict[str, Any]:
    cfg_dir = Path(config.get("_config_dir", ".")).resolve()
    data_cfg = dict(config.get("data") or {})
    if split_name not in data_cfg:
        raise ValueError(f"Split {split_name!r} not found in config.data")
    base_cfg = {
        "type": data_cfg.get("type", "pod5"),
        "root": data_cfg.get("root"),
        "subdirs": data_cfg.get("subdirs", ["."]),
        "segment_sec": float(data_cfg.get("segment_sec", 1.0)),
        "segment_samples": data_cfg.get("segment_samples"),
        "segment_hop_samples": data_cfg.get("segment_hop_samples"),
        "sample_rate": float(data_cfg.get("sample_rate", 5000.0)),
    }
    split_cfg = merge_split_cfg(base_cfg, data_cfg.get(split_name))
    split_cfg["root"] = str(resolve_data_path(split_cfg.get("root"), cfg_dir))
    return split_cfg


def resolve_split_files(config: dict[str, Any], split_name: str) -> list[Path]:
    split_cfg = resolve_split_cfg(config, split_name)
    explicit_files = split_cfg.get("files")
    if explicit_files:
        files = [Path(f).expanduser().resolve() for f in explicit_files]
    else:
        if split_cfg.get("type", "pod5") != "pod5":
            raise ValueError(f"Unsupported data.type={split_cfg.get('type')!r}")
        files = discover_pod5_files(Path(split_cfg["root"]), split_cfg.get("subdirs", ["."]))
    if not files:
        raise FileNotFoundError(f"No POD5 files found for split {split_name!r}")
    return files


def resolve_segment_samples(config: dict[str, Any], split_name: str) -> int:
    split_cfg = resolve_split_cfg(config, split_name)
    raw_value = split_cfg.get("segment_samples")
    if raw_value not in (None, ""):
        value = int(raw_value)
        if value > 0:
            return value
    sample_rate = float(split_cfg.get("sample_rate", 5000.0))
    segment_sec = float(split_cfg.get("segment_sec", 1.0))
    value = int(round(segment_sec * sample_rate))
    if value <= 0:
        raise ValueError(f"Invalid segment length for split {split_name!r}")
    return value


def resolve_segment_hop_samples(config: dict[str, Any], split_name: str) -> int:
    split_cfg = resolve_split_cfg(config, split_name)
    raw_value = split_cfg.get("segment_hop_samples")
    if raw_value not in (None, ""):
        value = int(raw_value)
        if value > 0:
            return value
    return resolve_segment_samples(config, split_name)


def resolve_recon_hop(
    recon_mode: str,
    chunk_size: int,
    configured_hop: int | None,
    hop_override: int | None,
) -> int:
    chunk_size = max(1, int(chunk_size))
    if hop_override is not None:
        hop_samples = int(hop_override)
    elif recon_mode == RECON_MODE_DIRECT:
        hop_samples = int(configured_hop) if configured_hop is not None else chunk_size
    else:
        overlap = min(max(0, int(DEFAULT_OVERLAP_SIZE)), max(0, chunk_size - 1))
        hop_samples = chunk_size - overlap
    if hop_samples <= 0:
        raise ValueError(f"Validation chunk hop must be positive, got {hop_samples}.")
    if hop_samples > chunk_size:
        raise ValueError(f"Validation chunk hop ({hop_samples}) cannot exceed chunk size ({chunk_size}).")
    return hop_samples


def trimmed_signal_length(signal_length: int, chunk_size: int, hop_size: int, trim_mode: str) -> int | None:
    n = max(0, int(signal_length))
    chunk_size = max(1, int(chunk_size))
    hop_size = max(1, int(hop_size))
    if hop_size > chunk_size:
        raise ValueError(f"hop_size ({hop_size}) cannot exceed chunk_size ({chunk_size}).")
    if n < chunk_size:
        if trim_mode == "pad" and n > 0:
            return chunk_size
        return None
    if trim_mode == "drop":
        step_count = 1 + ((n - chunk_size) // hop_size)
        return chunk_size + ((step_count - 1) * hop_size)
    if trim_mode == "pad":
        extra = n - chunk_size
        step_count = 1 + (extra // hop_size)
        if (extra % hop_size) != 0:
            step_count += 1
        return chunk_size + ((step_count - 1) * hop_size)
    raise ValueError(f"Unsupported trim_mode={trim_mode!r}")


def window_starts(total_samples: int, chunk_size: int, hop_size: int) -> list[int]:
    trimmed_length = trimmed_signal_length(total_samples, chunk_size, hop_size, "drop")
    if trimmed_length is None:
        return []
    last_start = trimmed_length - int(chunk_size)
    return list(range(0, last_start + 1, int(hop_size)))


def chunk_crossfade_weights(window_index: int, window_count: int, chunk_size: int, hop_size: int) -> np.ndarray:
    weights = np.ones((int(chunk_size),), dtype=np.float32)
    overlap_size = int(chunk_size) - int(hop_size)
    if overlap_size <= 0 or window_count <= 1:
        return weights
    fade_in = np.arange(overlap_size, dtype=np.float32) / float(overlap_size)
    fade_out = 1.0 - fade_in
    if window_index > 0:
        weights[:overlap_size] = fade_in
    if (window_index + 1) < window_count:
        weights[-overlap_size:] = fade_out
    return weights


def trim_signal(signal: np.ndarray, segment_samples: int, hop_samples: int, trim_mode: str) -> np.ndarray | None:
    raw = np.asarray(signal, dtype=np.int16).reshape(-1)
    n = int(raw.size)
    trimmed_length = trimmed_signal_length(n, segment_samples, hop_samples, trim_mode)
    if trimmed_length is None:
        return None
    if trimmed_length <= n:
        return raw[:trimmed_length]
    if n <= 0:
        return None
    pad = np.full((trimmed_length - n,), raw[-1], dtype=raw.dtype)
    return np.concatenate([raw, pad])


def reflect_pad_right_1d(values: np.ndarray, target_length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    target = int(target_length)
    n = int(arr.shape[0])
    if n >= target:
        return arr[:target]
    if n <= 0:
        return np.zeros((target,), dtype=np.float32)
    if n == 1:
        return np.full((target,), float(arr[0]), dtype=np.float32)
    period = 2 * n - 2
    idx = np.arange(target, dtype=np.int64) % period
    idx = np.where(idx < n, idx, period - idx)
    return arr[idx].astype(np.float32, copy=False)


def fetch_records_by_id(reader: Any, ordered_ids: Sequence[str]) -> dict[str, Any]:
    target_ids = [str(read_id) for read_id in ordered_ids]
    records: dict[str, Any] = {}
    try:
        iterator = reader.reads(selection=target_ids)
        for record in iterator:
            records[str(getattr(record, "read_id", ""))] = record
    except Exception:
        target_set = set(target_ids)
        for record in reader.reads():
            read_id = str(getattr(record, "read_id", ""))
            if read_id in target_set:
                records[read_id] = record
                if len(records) >= len(target_set):
                    break
    missing = [read_id for read_id in target_ids if read_id not in records]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} requested reads; first missing read_id={missing[0]}")
    return records


def write_selected_real_pod5(
    *,
    manifest_reads: list[ManifestRead],
    output_path: Path,
    segment_samples: int,
    hop_samples: int,
    trim_mode: str,
) -> tuple[int, dict[str, int]]:
    grouped: dict[str, list[ManifestRead]] = {}
    for item in manifest_reads:
        grouped.setdefault(item.source_file, []).append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    expected = len(manifest_reads)
    read_lengths: dict[str, int] = {}
    written = 0
    markers = progress_markers(expected)
    print(f"[real] writing selected POD5 -> {output_path}", flush=True)
    with Writer(str(output_path)) as writer:
        for file_path_str, items in grouped.items():
            ordered_ids = [item.read_id for item in items]
            with pod5.Reader(str(Path(file_path_str))) as reader:
                fetched = fetch_records_by_id(reader, ordered_ids)
                for item in items:
                    record = fetched[item.read_id]
                    trimmed = trim_signal(
                        np.asarray(record.signal, dtype=np.int16),
                        segment_samples,
                        hop_samples,
                        trim_mode,
                    )
                    if trimmed is None:
                        raise RuntimeError(f"Selected read became empty after trim: {file_path_str}:{item.read_id}")
                    read = record.to_read()
                    read.signal = np.asarray(trimmed, dtype=np.int16)
                    writer.add_read(read)
                    read_lengths[item.read_id] = int(trimmed.size)
                    written += 1
                    if markers and written >= markers[0]:
                        print(f"[real] progress {written}/{expected}", flush=True)
                        markers.pop(0)
    print(f"[real] done. kept {written}/{expected} reads", flush=True)
    return written, read_lengths


def build_shift_last_windows(total_samples: int, chunk_size: int, hop_size: int) -> list[ShiftLastWindow]:
    n = int(total_samples)
    chunk_size = int(chunk_size)
    hop_size = int(hop_size)
    if n < chunk_size:
        return []
    last_start = n - chunk_size
    starts = list(range(0, last_start + 1, hop_size))
    if not starts:
        starts = [0]
    windows = [
        ShiftLastWindow(start=int(start), stop=int(start + chunk_size), valid_from=0, valid_to=chunk_size)
        for start in starts
    ]
    if starts[-1] != last_start:
        prev_start = starts[-1]
        valid_global_start = prev_start + hop_size
        local_valid_from = max(0, int(valid_global_start - last_start))
        windows.append(
            ShiftLastWindow(
                start=int(last_start),
                stop=int(last_start + chunk_size),
                valid_from=local_valid_from,
                valid_to=chunk_size,
            )
        )
    return windows


def build_pad_last_windows(total_samples: int, chunk_size: int, hop_size: int) -> list[PadLastWindow]:
    n = int(total_samples)
    chunk_size = int(chunk_size)
    hop_size = int(hop_size)
    if n < chunk_size:
        return []
    last_full_start = n - chunk_size
    starts = list(range(0, last_full_start + 1, hop_size))
    if not starts:
        starts = [0]
    windows = [
        PadLastWindow(start=int(start), stop=int(start + chunk_size), valid_length=chunk_size)
        for start in starts
    ]
    tail_start = int(starts[-1] + hop_size)
    if tail_start < n:
        tail_valid = n - tail_start
        windows.append(
            PadLastWindow(
                start=int(tail_start),
                stop=int(tail_start + chunk_size),
                valid_length=int(tail_valid),
                is_padded_tail=True,
            )
        )
    return windows


def shift_last_window_weights(
    *,
    window_index: int,
    window_count: int,
    valid_from: int,
    valid_to: int,
    chunk_size: int,
    hop_size: int,
) -> np.ndarray:
    weights = np.zeros((int(chunk_size),), dtype=np.float32)
    valid_from = max(0, int(valid_from))
    valid_to = min(int(chunk_size), int(valid_to))
    if valid_to <= valid_from:
        return weights
    weights[valid_from:valid_to] = 1.0
    overlap_size = int(chunk_size) - int(hop_size)
    active_len = valid_to - valid_from
    if overlap_size <= 0 or window_count <= 1 or active_len <= 0:
        return weights
    fade_len = min(overlap_size, active_len)
    if window_index > 0 and fade_len > 0:
        fade_in = np.arange(fade_len, dtype=np.float32) / float(fade_len)
        weights[valid_from : valid_from + fade_len] = fade_in
    if (window_index + 1) < window_count and fade_len > 0:
        fade_out = 1.0 - (np.arange(fade_len, dtype=np.float32) / float(fade_len))
        weights[valid_to - fade_len : valid_to] = np.minimum(weights[valid_to - fade_len : valid_to], fade_out)
    return weights


def write_selected_real_pod5_shift_last(
    *,
    manifest_reads: list[ManifestRead],
    output_path: Path,
) -> tuple[int, dict[str, int]]:
    grouped: dict[str, list[ManifestRead]] = {}
    for item in manifest_reads:
        grouped.setdefault(item.source_file, []).append(item)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    expected = len(manifest_reads)
    written = 0
    read_lengths: dict[str, int] = {}
    markers = progress_markers(expected)
    print(f"[real] writing selected POD5 -> {output_path}", flush=True)
    with Writer(str(output_path)) as writer:
        for file_path_str, items in grouped.items():
            ordered_ids = [item.read_id for item in items]
            with pod5.Reader(str(Path(file_path_str))) as reader:
                fetched = fetch_records_by_id(reader, ordered_ids)
                for item in items:
                    record = fetched[item.read_id]
                    signal = np.asarray(record.signal, dtype=np.int16)
                    read = record.to_read()
                    read.signal = signal
                    writer.add_read(read)
                    read_lengths[item.read_id] = int(signal.size)
                    written += 1
                    if markers and written >= markers[0]:
                        print(f"[real] progress {written}/{expected}", flush=True)
                        markers.pop(0)
    print(f"[real] done. kept {written}/{expected} reads", flush=True)
    return written, read_lengths


class CheckpointReconstructor:
    def __init__(self, *, model_cfg: dict[str, Any], checkpoint_path: Path, chunk_batch_size: int):
        self.requested_chunk_batch_size = max(1, int(chunk_batch_size))
        self.variables = load_generator_variables(checkpoint_path)
        self.model = build_model(model_cfg)
        self.model_apply_mode = DIVEQ_VALID_MODEL_APPLY_MODE
        self.apply_train = True
        self.apply_rng_seed = DIVEQ_VALID_RNG_SEED
        self.apply_rng = jax.random.PRNGKey(self.apply_rng_seed)
        local_devices = list(jax.local_devices())
        if len(local_devices) < VALID_RECON_DEVICE_COUNT:
            raise RuntimeError(
                f"Validation reconstruction requires {VALID_RECON_DEVICE_COUNT} local JAX devices, "
                f"but only {len(local_devices)} are visible."
            )
        self.devices = local_devices[:VALID_RECON_DEVICE_COUNT]
        self.device_count = len(self.devices)
        self.per_device_chunk_batch_size = max(
            1,
            (self.requested_chunk_batch_size + self.device_count - 1) // self.device_count,
        )
        self.chunk_batch_size = int(self.device_count * self.per_device_chunk_batch_size)
        self.parallelism = "pmap"
        self.replicated_variables = host_broadcast_tree_for_pmap(self.variables, self.device_count)
        self.apply_rngs = np.asarray(jax.random.split(self.apply_rng, self.device_count))

        @partial(jax.pmap, in_axes=(0, 0, 0), out_axes=0, devices=self.devices)
        def _reconstruct(variables: dict[str, Any], batch: jax.Array, rng: jax.Array) -> jax.Array:
            outputs = self.model.apply(
                variables,
                batch,
                train=self.apply_train,
                offset=0,
                rng=rng,
                collect_codebook_stats=False,
            )
            wave_hat = outputs["wave_hat"]
            if wave_hat.ndim == 3 and wave_hat.shape[1] == 1:
                wave_hat = wave_hat[:, 0, :]
            elif wave_hat.ndim != 2:
                wave_hat = wave_hat.reshape(wave_hat.shape[0], -1)
            return wave_hat

        self._reconstruct = _reconstruct

    def reconstruct_chunks(self, chunks: np.ndarray) -> np.ndarray:
        chunks = np.asarray(chunks, dtype=np.float32)
        if chunks.ndim != 2:
            raise ValueError(f"Expected chunk batch with shape [N, T], got {chunks.shape}")
        total_chunks, chunk_size = chunks.shape
        outputs = np.empty((total_chunks, chunk_size), dtype=np.float32)
        for batch_start in range(0, total_chunks, self.chunk_batch_size):
            batch = np.asarray(chunks[batch_start : batch_start + self.chunk_batch_size], dtype=np.float32)
            valid = int(batch.shape[0])
            if valid < self.chunk_batch_size:
                padded = np.zeros((self.chunk_batch_size, chunk_size), dtype=np.float32)
                padded[:valid] = batch
                batch = padded
            batch = batch.reshape(self.device_count, self.per_device_chunk_batch_size, chunk_size)
            reconstructed = np.asarray(
                self._reconstruct(self.replicated_variables, batch, self.apply_rngs),
                dtype=np.float32,
            ).reshape(self.chunk_batch_size, chunk_size)
            outputs[batch_start : batch_start + valid] = reconstructed[:valid]
        return outputs


def load_prepared_reads(
    *,
    source_pod5: Path,
    segment_samples: int,
    hop_samples: int,
    recon_mode: str,
) -> tuple[list[ReconstructionRead], list[ChunkSpec], np.ndarray]:
    reads: list[ReconstructionRead] = []
    chunk_specs: list[ChunkSpec] = []
    chunks: list[np.ndarray] = []
    total_reads = count_reads(source_pod5)
    markers = progress_markers(total_reads)
    print(f"[prep] building chunk pool from {source_pod5}", flush=True)
    with pod5.Reader(str(source_pod5)) as reader:
        for read_index, record in enumerate(reader.reads(), start=0):
            if markers and (read_index + 1) >= markers[0]:
                print(f"[prep] read progress {read_index + 1}/{total_reads}", flush=True)
                markers.pop(0)
            read_id = str(getattr(record, "read_id", ""))
            trimmed_raw = np.asarray(record.signal, dtype=np.int16).reshape(-1)
            calibration = parse_calibration(getattr(record, "calibration", None))
            trimmed_pa = calibration.to_picoamps(trimmed_raw)
            starts = window_starts(int(trimmed_raw.size), segment_samples, hop_samples)
            if not starts:
                raise RuntimeError(f"Selected POD5 read has no reconstructable chunks: {read_id}")
            overlap_weights = None
            if recon_mode == RECON_MODE_OVERLAP:
                overlap_weights = [
                    chunk_crossfade_weights(chunk_index, len(starts), segment_samples, hop_samples)
                    for chunk_index in range(len(starts))
                ]
            state = ReconstructionRead(
                read_index=read_index,
                source_file=str(source_pod5),
                read_id=read_id,
                raw_length=int(trimmed_raw.size),
                trimmed_length=int(trimmed_raw.size),
                chunk_count=len(starts),
                calibration=calibration,
                template_read=record.to_read(),
                trimmed_raw=trimmed_raw,
                trimmed_pa=np.asarray(trimmed_pa, dtype=np.float32),
                chunk_starts=starts,
                overlap_weights=overlap_weights,
            )
            if recon_mode == RECON_MODE_DIRECT:
                state.reconstructed_pa = np.zeros((int(trimmed_raw.size),), dtype=np.float32)
                state.reconstructed_adc = np.zeros((int(trimmed_raw.size),), dtype=np.int16)
            else:
                state.pa_acc = np.zeros((int(trimmed_raw.size),), dtype=np.float32)
                state.weight_acc = np.zeros((int(trimmed_raw.size),), dtype=np.float32)
            reads.append(state)
            for chunk_index, start in enumerate(starts):
                stop = start + int(segment_samples)
                chunk_pa = np.asarray(trimmed_pa[start:stop], dtype=np.float32)
                normalized, center, half_range = normalize_to_pm1_with_stats(chunk_pa)
                chunks.append(np.asarray(normalized, dtype=np.float32))
                chunk_specs.append(
                    ChunkSpec(
                        read_index=read_index,
                        read_id=read_id,
                        chunk_index=chunk_index,
                        start=int(start),
                        stop=int(stop),
                        center=float(center),
                        half_range=float(half_range),
                    )
                )
    if not chunks:
        raise RuntimeError("Prepared chunk pool is empty.")
    print(f"[prep] done. reads={len(reads)} chunks={len(chunks)}", flush=True)
    return reads, chunk_specs, np.stack(chunks, axis=0).astype(np.float32)


def load_prepared_reads_shift_last(
    *,
    source_pod5: Path,
    segment_samples: int,
    hop_samples: int,
    recon_mode: str,
) -> tuple[list[ReconstructionRead], list[ChunkSpec], np.ndarray]:
    reads: list[ReconstructionRead] = []
    chunk_specs: list[ChunkSpec] = []
    chunks: list[np.ndarray] = []
    total_reads = count_reads(source_pod5)
    markers = progress_markers(total_reads)
    print(f"[prep] building chunk pool from {source_pod5}", flush=True)
    with pod5.Reader(str(source_pod5)) as reader:
        for read_index, record in enumerate(reader.reads(), start=0):
            if markers and (read_index + 1) >= markers[0]:
                print(f"[prep] read progress {read_index + 1}/{total_reads}", flush=True)
                markers.pop(0)
            read_id = str(getattr(record, "read_id", ""))
            raw = np.asarray(record.signal, dtype=np.int16).reshape(-1)
            calibration = parse_calibration(getattr(record, "calibration", None))
            signal_pa = calibration.to_picoamps(raw)
            windows = build_shift_last_windows(int(raw.size), segment_samples, hop_samples)
            if not windows:
                raise RuntimeError(f"Selected POD5 read has no reconstructable chunks: {read_id}")
            overlap_weights = None
            if recon_mode == RECON_MODE_OVERLAP:
                overlap_weights = [
                    shift_last_window_weights(
                        window_index=window_index,
                        window_count=len(windows),
                        valid_from=window.valid_from,
                        valid_to=window.valid_to,
                        chunk_size=segment_samples,
                        hop_size=hop_samples,
                    )
                    for window_index, window in enumerate(windows)
                ]
            state = ReconstructionRead(
                read_index=read_index,
                source_file=str(source_pod5),
                read_id=read_id,
                raw_length=int(raw.size),
                trimmed_length=int(raw.size),
                chunk_count=len(windows),
                calibration=calibration,
                template_read=record.to_read(),
                trimmed_raw=raw,
                trimmed_pa=np.asarray(signal_pa, dtype=np.float32),
                chunk_starts=[int(window.start) for window in windows],
                overlap_weights=overlap_weights,
            )
            if recon_mode == RECON_MODE_DIRECT:
                state.reconstructed_pa = np.zeros((int(raw.size),), dtype=np.float32)
                state.reconstructed_adc = np.zeros((int(raw.size),), dtype=np.int16)
            else:
                state.pa_acc = np.zeros((int(raw.size),), dtype=np.float32)
                state.weight_acc = np.zeros((int(raw.size),), dtype=np.float32)
            reads.append(state)
            for chunk_index, window in enumerate(windows):
                chunk_pa = np.asarray(signal_pa[window.start : window.stop], dtype=np.float32)
                normalized, center, half_range = normalize_to_pm1_with_stats(chunk_pa)
                chunks.append(np.asarray(normalized, dtype=np.float32))
                chunk_specs.append(
                    ChunkSpec(
                        read_index=read_index,
                        read_id=read_id,
                        chunk_index=chunk_index,
                        start=int(window.start),
                        stop=int(window.stop),
                        center=float(center),
                        half_range=float(half_range),
                    )
                )
    if not chunks:
        raise RuntimeError("Prepared chunk pool is empty.")
    print(f"[prep] done. reads={len(reads)} chunks={len(chunks)}", flush=True)
    return reads, chunk_specs, np.stack(chunks, axis=0).astype(np.float32)


def load_prepared_reads_pad_last(
    *,
    source_pod5: Path,
    segment_samples: int,
    hop_samples: int,
    recon_mode: str,
) -> tuple[list[ReconstructionRead], list[ChunkSpec], np.ndarray]:
    reads: list[ReconstructionRead] = []
    chunk_specs: list[ChunkSpec] = []
    chunks: list[np.ndarray] = []
    total_reads = count_reads(source_pod5)
    markers = progress_markers(total_reads)
    print(f"[prep] building chunk pool from {source_pod5}", flush=True)
    with pod5.Reader(str(source_pod5)) as reader:
        for read_index, record in enumerate(reader.reads(), start=0):
            if markers and (read_index + 1) >= markers[0]:
                print(f"[prep] read progress {read_index + 1}/{total_reads}", flush=True)
                markers.pop(0)
            read_id = str(getattr(record, "read_id", ""))
            raw = np.asarray(record.signal, dtype=np.int16).reshape(-1)
            calibration = parse_calibration(getattr(record, "calibration", None))
            signal_pa = calibration.to_picoamps(raw)
            windows = build_pad_last_windows(int(raw.size), segment_samples, hop_samples)
            if not windows:
                raise RuntimeError(f"Selected POD5 read has no reconstructable chunks: {read_id}")
            overlap_weights = None
            if recon_mode == RECON_MODE_OVERLAP:
                overlap_weights = [
                    chunk_crossfade_weights(chunk_index, len(windows), segment_samples, hop_samples)
                    for chunk_index in range(len(windows))
                ]
            state = ReconstructionRead(
                read_index=read_index,
                source_file=str(source_pod5),
                read_id=read_id,
                raw_length=int(raw.size),
                trimmed_length=int(raw.size),
                chunk_count=len(windows),
                calibration=calibration,
                template_read=record.to_read(),
                trimmed_raw=raw,
                trimmed_pa=np.asarray(signal_pa, dtype=np.float32),
                chunk_starts=[int(window.start) for window in windows],
                overlap_weights=overlap_weights,
            )
            if recon_mode == RECON_MODE_DIRECT:
                state.reconstructed_pa = np.zeros((int(raw.size),), dtype=np.float32)
                state.reconstructed_adc = np.zeros((int(raw.size),), dtype=np.int16)
            else:
                state.pa_acc = np.zeros((int(raw.size),), dtype=np.float32)
                state.weight_acc = np.zeros((int(raw.size),), dtype=np.float32)
            reads.append(state)
            for chunk_index, window in enumerate(windows):
                valid_stop = min(int(window.start + window.valid_length), int(raw.size))
                chunk_pa = np.asarray(signal_pa[window.start:valid_stop], dtype=np.float32)
                normalized, center, half_range = normalize_to_pm1_with_stats(chunk_pa)
                if int(window.valid_length) < int(segment_samples):
                    normalized = reflect_pad_right_1d(normalized, segment_samples)
                chunks.append(np.asarray(normalized, dtype=np.float32))
                chunk_specs.append(
                    ChunkSpec(
                        read_index=read_index,
                        read_id=read_id,
                        chunk_index=chunk_index,
                        start=int(window.start),
                        stop=int(window.start + window.valid_length),
                        center=float(center),
                        half_range=float(half_range),
                    )
                )
    if not chunks:
        raise RuntimeError("Prepared chunk pool is empty.")
    print(f"[prep] done. reads={len(reads)} chunks={len(chunks)}", flush=True)
    return reads, chunk_specs, np.stack(chunks, axis=0).astype(np.float32)


def finalize_reconstruction(
    *,
    reads: list[ReconstructionRead],
    chunk_specs: Sequence[ChunkSpec],
    chunk_inputs: np.ndarray,
    chunk_outputs: np.ndarray,
    recon_mode: str,
) -> dict[str, Any]:
    read_count = len(reads)
    norm_abs = np.zeros((read_count,), dtype=np.float64)
    norm_sq = np.zeros((read_count,), dtype=np.float64)
    norm_count = np.zeros((read_count,), dtype=np.int64)

    for chunk_index, spec in enumerate(chunk_specs):
        read_state = reads[spec.read_index]
        output_norm = np.asarray(chunk_outputs[chunk_index], dtype=np.float32)
        input_norm = np.asarray(chunk_inputs[chunk_index], dtype=np.float32)
        valid_len = min(int(output_norm.size), int(input_norm.size), max(0, int(spec.stop) - int(spec.start)))
        if valid_len <= 0:
            continue
        diff = output_norm[:valid_len] - input_norm[:valid_len]
        norm_abs[spec.read_index] += float(np.sum(np.abs(diff), dtype=np.float64))
        norm_sq[spec.read_index] += float(np.sum(np.square(diff), dtype=np.float64))
        norm_count[spec.read_index] += int(diff.size)

        stats = NormalizationStats(center=float(spec.center), half_range=float(spec.half_range))
        pa_chunk, adc_chunk = denormalize_to_adc(output_norm, stats, read_state.calibration)
        pa_chunk = np.asarray(pa_chunk[:valid_len], dtype=np.float32)
        adc_chunk = np.asarray(np.clip(np.rint(adc_chunk[:valid_len]), -32768, 32767), dtype=np.int16)
        stop = int(spec.start + valid_len)

        if recon_mode == RECON_MODE_DIRECT:
            if read_state.reconstructed_pa is None or read_state.reconstructed_adc is None:
                raise RuntimeError(f"Direct reconstruction buffers missing for read {read_state.read_id}")
            read_state.reconstructed_pa[spec.start:stop] = pa_chunk
            read_state.reconstructed_adc[spec.start:stop] = adc_chunk
        else:
            if read_state.pa_acc is None or read_state.weight_acc is None or read_state.overlap_weights is None:
                raise RuntimeError(f"Overlap reconstruction buffers missing for read {read_state.read_id}")
            weights = np.asarray(read_state.overlap_weights[spec.chunk_index], dtype=np.float32)[:valid_len]
            read_state.pa_acc[spec.start:stop] += pa_chunk * weights
            read_state.weight_acc[spec.start:stop] += weights

    for read_index, read_state in enumerate(reads):
        if recon_mode == RECON_MODE_OVERLAP:
            if read_state.pa_acc is None or read_state.weight_acc is None:
                raise RuntimeError(f"Overlap accumulators missing for read {read_state.read_id}")
            reconstructed_pa = np.divide(
                read_state.pa_acc,
                np.where(read_state.weight_acc > 0.0, read_state.weight_acc, 1.0),
            )
            reconstructed_adc = read_state.calibration.to_adc(reconstructed_pa)
            read_state.reconstructed_pa = np.asarray(reconstructed_pa, dtype=np.float32)
            read_state.reconstructed_adc = np.asarray(
                np.clip(np.rint(reconstructed_adc), -32768, 32767),
                dtype=np.int16,
            )
        if read_state.reconstructed_pa is None or read_state.reconstructed_adc is None:
            raise RuntimeError(f"Missing reconstruction output for read {read_state.read_id}")

        count = max(1, int(norm_count[read_index]))
        read_state.chunk_norm_mae = float(norm_abs[read_index] / float(count))
        read_state.chunk_norm_rmse = float(np.sqrt(norm_sq[read_index] / float(count)))

        pa_diff = np.asarray(read_state.reconstructed_pa, dtype=np.float32) - np.asarray(read_state.trimmed_pa, dtype=np.float32)
        adc_diff = read_state.reconstructed_adc.astype(np.float32) - read_state.trimmed_raw.astype(np.float32)
        read_state.pa_mae = float(np.mean(np.abs(pa_diff), dtype=np.float64))
        read_state.pa_rmse = float(np.sqrt(np.mean(np.square(pa_diff), dtype=np.float64)))
        read_state.adc_mae = float(np.mean(np.abs(adc_diff), dtype=np.float64))
        read_state.adc_rmse = float(np.sqrt(np.mean(np.square(adc_diff), dtype=np.float64)))

    return {
        "processed_reads": len(reads),
        "total_chunk_count": int(len(chunk_specs)),
        "trimmed_length_summary": summarize(read.trimmed_length for read in reads),
        "chunk_count_summary": summarize(read.chunk_count for read in reads),
        "chunk_norm_mae_summary": summarize(read.chunk_norm_mae for read in reads),
        "chunk_norm_rmse_summary": summarize(read.chunk_norm_rmse for read in reads),
        "pa_mae_summary": summarize(read.pa_mae for read in reads),
        "pa_rmse_summary": summarize(read.pa_rmse for read in reads),
        "adc_mae_summary": summarize(read.adc_mae for read in reads),
        "adc_rmse_summary": summarize(read.adc_rmse for read in reads),
    }


def write_generated_pod5(*, output_path: Path, reads: Sequence[ReconstructionRead]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    expected = len(reads)
    written = 0
    markers = progress_markers(expected)
    print(f"[gen] writing reconstructed POD5 -> {output_path}", flush=True)
    with Writer(str(output_path)) as writer:
        for index, read_state in enumerate(reads, start=1):
            if read_state.reconstructed_adc is None:
                raise RuntimeError(f"Missing reconstructed ADC for read {read_state.read_id}")
            read = read_state.template_read
            read.signal = np.asarray(read_state.reconstructed_adc, dtype=np.int16)
            writer.add_read(read)
            written += 1
            if markers and index >= markers[0]:
                print(f"[gen] progress {index}/{expected}", flush=True)
                markers.pop(0)
    print(f"[gen] done. kept {written}/{expected} reads", flush=True)
    return written


def reconstruction_per_read_metrics(reads: Sequence[ReconstructionRead]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for read in reads:
        records.append(
            {
                "source_file": read.source_file,
                "read_id": read.read_id,
                "raw_length": int(read.raw_length),
                "trimmed_length": int(read.trimmed_length),
                "chunk_count": int(read.chunk_count),
                "chunk_norm_mae": float(read.chunk_norm_mae),
                "chunk_norm_rmse": float(read.chunk_norm_rmse),
                "pa_mae": float(read.pa_mae),
                "pa_rmse": float(read.pa_rmse),
                "adc_mae": float(read.adc_mae),
                "adc_rmse": float(read.adc_rmse),
            }
        )
    return records


def count_reads(path: Path) -> int:
    with pod5.Reader(str(path)) as reader:
        return sum(1 for _ in reader.reads())


def iter_source_specs(
    *,
    files: Sequence[str | Path],
    chunk_size: int,
    chunk_hop: int,
    sample_rate_hz: float,
    target_chunks: int,
    chunks_per_step: int,
) -> tuple[list[SourceChunkSpec], np.ndarray, list[str]]:
    del chunks_per_step
    chunk_size = max(1, int(chunk_size))
    chunk_hop = max(1, int(chunk_hop))
    target_chunks = max(1, int(target_chunks))
    warnings: list[str] = []
    specs: list[SourceChunkSpec] = []
    chunks: list[np.ndarray] = []

    for file_path_raw in files:
        file_path = Path(file_path_raw).expanduser().resolve()
        try:
            with pod5.Reader(str(file_path)) as reader:
                for record in reader.reads():
                    read_id = str(getattr(record, "read_id", "")) or f"{file_path.name}:{len(specs)}"
                    raw_signal = np.asarray(record.signal, dtype=np.int16)
                    if raw_signal.size < chunk_size:
                        continue
                    try:
                        normalized, _, _ = normalize_adc_signal(raw_signal, getattr(record, "calibration", None))
                    except Exception as exc:
                        warnings.append(f"skip {file_path.name}:{read_id} ({exc})")
                        continue
                    last_start = int(normalized.shape[0]) - chunk_size
                    for start in range(0, last_start + 1, chunk_hop):
                        stop = start + chunk_size
                        specs.append(
                            SourceChunkSpec(
                                source_file=str(file_path),
                                source_read_id=read_id,
                                source_start=start,
                                source_stop=stop,
                                source_length=int(raw_signal.size),
                                sample_rate_hz=float(sample_rate_hz),
                            )
                        )
                        chunks.append(np.asarray(normalized[start:stop], dtype=np.float32))
                        if len(chunks) >= target_chunks:
                            break
                    if len(chunks) >= target_chunks:
                        break
        except Exception as exc:
            warnings.append(f"failed to scan {file_path}: {exc}")
        if len(chunks) >= target_chunks:
            break

    if not chunks:
        raise RuntimeError("Unable to extract any normalized chunks from the requested POD5 sources.")

    if len(chunks) < target_chunks:
        original_specs = list(specs)
        original_chunks = [np.asarray(chunk, dtype=np.float32) for chunk in chunks]
        warnings.append(f"Only found {len(chunks)} chunks; repeating cached chunks to reach requested {target_chunks}.")
        repeat_idx = 0
        while len(chunks) < target_chunks:
            specs.append(original_specs[repeat_idx % len(original_specs)])
            chunks.append(np.array(original_chunks[repeat_idx % len(original_chunks)], copy=True))
            repeat_idx += 1

    stacked = np.stack(chunks[:target_chunks], axis=0).astype(np.float32)
    return specs[:target_chunks], stacked, warnings


_build_model = build_model
_iter_source_specs = iter_source_specs
_load_generator_variables = load_generator_variables
_load_json = load_json
_resolve_segment_hop_samples = resolve_segment_hop_samples
_resolve_segment_samples = resolve_segment_samples
_resolve_split_files = resolve_split_files
_to_host_tree = to_host_tree
