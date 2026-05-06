#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import pod5
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(f"pod5 is required: {exc}") from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from codec.data.normalization import normalize_to_pm1_with_stats
from codec.data.pod5_dataset import _reflect_pad_right_1d
from codec.data.pod5_processing import CalibrationError, parse_calibration


DEFAULT_DATASET_DIR = Path("/data_nvme/simvqgan_posttrain_sup_truth_shiftlast_5to8/FC01_barcode01_id_gt_99")
DEFAULT_POD5_ROOT = Path("/data/nanopore/hereditary_cancer_2025.09/raw/FC01/pod5")
BASE_TO_TOKEN = {
    "N": 0,
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
}


@dataclass
class TruthRecord:
    sequence: str
    qscores: np.ndarray


@dataclass
class BucketBuffer:
    bucket_chunks: int
    max_truth_len: int
    shard_reads: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    chunks_norm: list[np.ndarray] = field(default_factory=list)
    chunk_center: list[np.ndarray] = field(default_factory=list)
    chunk_half_range: list[np.ndarray] = field(default_factory=list)
    chunk_valid_mask: list[np.ndarray] = field(default_factory=list)
    chunk_start: list[np.ndarray] = field(default_factory=list)
    chunk_valid_length: list[np.ndarray] = field(default_factory=list)
    truth_tokens: list[np.ndarray] = field(default_factory=list)
    truth_qscores: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)

    def append(self, sample: dict[str, Any]) -> None:
        self.rows.append(sample["row"])
        self.chunks_norm.append(sample["chunks_norm"])
        self.chunk_center.append(sample["chunk_center"])
        self.chunk_half_range.append(sample["chunk_half_range"])
        self.chunk_valid_mask.append(sample["chunk_valid_mask"])
        self.chunk_start.append(sample["chunk_start"])
        self.chunk_valid_length.append(sample["chunk_valid_length"])
        self.truth_tokens.append(sample["truth_tokens"])
        self.truth_qscores.append(sample["truth_qscores"])

    def should_flush(self) -> bool:
        return len(self) >= int(self.shard_reads)

    def clear(self) -> None:
        self.rows.clear()
        self.chunks_norm.clear()
        self.chunk_center.clear()
        self.chunk_half_range.clear()
        self.chunk_valid_mask.clear()
        self.chunk_start.clear()
        self.chunk_valid_length.clear()
        self.truth_tokens.clear()
        self.truth_qscores.clear()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_candidate_rows(
    path: Path,
    *,
    max_reads: int | None,
    selection_mode: str,
    random_seed: int,
) -> list[dict[str, Any]]:
    rows = [row for row in _iter_jsonl(path) if str(row.get("read_id") or "")]
    if selection_mode == "pod5_first":
        max_reads = None
    elif selection_mode == "random":
        rng = random.Random(int(random_seed))
        rng.shuffle(rows)
    elif selection_mode != "first":
        raise ValueError(f"Unsupported selection mode: {selection_mode}")
    if max_reads is not None:
        rows = rows[: max(1, int(max_reads))]
    seen: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    for row in rows:
        read_id = str(row.get("read_id") or "")
        if not read_id or read_id in seen:
            continue
        seen.add(read_id)
        unique_rows.append(row)
    return unique_rows


def _load_truth_fastq(path: Path, target_ids: set[str]) -> dict[str, TruthRecord]:
    truth: dict[str, TruthRecord] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        while True:
            header = handle.readline()
            if not header:
                break
            seq = handle.readline()
            plus = handle.readline()
            qual = handle.readline()
            if not qual:
                break
            if not header.startswith("@") or not plus.startswith("+"):
                raise ValueError(f"Malformed FASTQ around header: {header.strip()!r}")
            read_id = header[1:].strip().split()[0]
            if read_id not in target_ids:
                continue
            sequence = seq.strip().upper()
            quality = qual.strip()
            qscores = np.fromiter((max(0, ord(ch) - 33) for ch in quality), dtype=np.uint8, count=len(quality))
            if qscores.shape[0] != len(sequence):
                qscores = np.full((len(sequence),), 40, dtype=np.uint8)
            truth[read_id] = TruthRecord(sequence=sequence, qscores=qscores)
            if len(truth) >= len(target_ids):
                break
    return truth


def _discover_pod5_files(root: Path, glob_pattern: str) -> list[Path]:
    if root.is_file():
        return [root.resolve()]
    files = sorted(root.rglob(glob_pattern))
    return [path.resolve() for path in files if path.is_file()]


def _select_pod5_first_candidates(
    *,
    pod5_files: list[Path],
    row_by_id: dict[str, dict[str, Any]],
    max_reads: int,
    progress_every_files: int,
    progress_every_reads: int,
) -> tuple[list[dict[str, Any]], list[Path]]:
    selected_ids: list[str] = []
    selected_set: set[str] = set()
    source_files: list[Path] = []
    source_seen: set[Path] = set()
    candidate_ids = set(row_by_id)
    scanned_reads = 0
    started_at = time.time()
    progress_every_files = max(1, int(progress_every_files))
    progress_every_reads = max(1, int(progress_every_reads))
    for file_idx, pod5_file in enumerate(pod5_files, start=1):
        if len(selected_ids) >= int(max_reads):
            break
        if file_idx == 1 or file_idx % progress_every_files == 0:
            print(
                f"[select] file={file_idx}/{len(pod5_files)} selected={len(selected_ids)}/{int(max_reads)}",
                flush=True,
            )
        with pod5.Reader(str(pod5_file)) as reader:
            for record in reader.reads():
                scanned_reads += 1
                if scanned_reads % progress_every_reads == 0:
                    elapsed = max(1e-6, time.time() - started_at)
                    print(
                        f"[select] reads={scanned_reads} selected={len(selected_ids)} "
                        f"rate={scanned_reads / elapsed:.0f}/s",
                        flush=True,
                    )
                read_id = str(getattr(record, "read_id", ""))
                if read_id not in candidate_ids or read_id in selected_set:
                    continue
                selected_ids.append(read_id)
                selected_set.add(read_id)
                if pod5_file not in source_seen:
                    source_files.append(pod5_file)
                    source_seen.add(pod5_file)
                if len(selected_ids) >= int(max_reads):
                    break
    rows = [row_by_id[read_id] for read_id in selected_ids]
    print(
        f"[select] done selected={len(rows)} scanned_reads={scanned_reads} source_files={len(source_files)}",
        flush=True,
    )
    return rows, source_files


def _parse_bucket_chunks(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in str(raw).split(",") if part.strip()}))
    if not values or any(value <= 0 for value in values):
        raise ValueError("--bucket-chunks must contain positive integers.")
    return values


def _bucket_for_chunk_count(n_chunks: int, bucket_chunks: tuple[int, ...]) -> int | None:
    for bucket in bucket_chunks:
        if int(n_chunks) <= int(bucket):
            return int(bucket)
    return None


def _truth_to_tokens(sequence: str, max_len: int) -> tuple[np.ndarray, np.ndarray, int]:
    seq = str(sequence).upper()
    truth_len = int(len(seq))
    tokens = np.zeros((int(max_len),), dtype=np.int16)
    mask = np.zeros((int(max_len),), dtype=np.uint8)
    n = min(truth_len, int(max_len))
    if n > 0:
        tokens[:n] = np.asarray([BASE_TO_TOKEN.get(base, 0) for base in seq[:n]], dtype=np.int16)
        mask[:n] = 1
    return tokens, mask, truth_len


def _pad_qscores(qscores: np.ndarray, max_len: int) -> np.ndarray:
    out = np.zeros((int(max_len),), dtype=np.uint8)
    n = min(int(qscores.shape[0]), int(max_len))
    if n > 0:
        out[:n] = np.asarray(qscores[:n], dtype=np.uint8)
    return out


def _build_windows(
    total_samples: int,
    chunk_size: int,
    hop_size: int,
    *,
    tail_chunk_mode: str,
) -> list[tuple[int, int]]:
    n = int(total_samples)
    chunk_size = int(chunk_size)
    hop_size = int(hop_size)
    tail_chunk_mode = str(tail_chunk_mode).strip().lower()
    if n <= 0:
        return []
    if n <= chunk_size:
        return [(0, n)]
    last_full_start = n - chunk_size
    starts = list(range(0, last_full_start + 1, hop_size))
    if not starts:
        starts = [0]
    if tail_chunk_mode == "shift_last":
        if starts[-1] != last_full_start:
            starts.append(last_full_start)
        return [(int(start), chunk_size) for start in starts]
    if tail_chunk_mode != "pad_last":
        raise ValueError(f"Unsupported tail_chunk_mode: {tail_chunk_mode}")
    windows = [(int(start), chunk_size) for start in starts]
    tail_start = int(starts[-1] + hop_size)
    if tail_start < n:
        windows.append((tail_start, int(n - tail_start)))
    return windows


def _materialize_read(
    *,
    row: dict[str, Any],
    truth: TruthRecord,
    signal_pa: np.ndarray,
    source_pod5: Path,
    bucket_chunks: int,
    max_truth_len: int,
    chunk_size: int,
    hop_size: int,
    min_signal_samples: int,
    tail_chunk_mode: str,
) -> dict[str, Any] | None:
    raw_len = int(signal_pa.shape[0])
    if raw_len < int(min_signal_samples):
        return None
    windows = _build_windows(raw_len, chunk_size, hop_size, tail_chunk_mode=tail_chunk_mode)
    if not windows:
        return None
    if len(windows) > int(bucket_chunks):
        return None

    chunks = np.zeros((int(bucket_chunks), int(chunk_size)), dtype=np.float32)
    centers = np.zeros((int(bucket_chunks),), dtype=np.float32)
    half_ranges = np.ones((int(bucket_chunks),), dtype=np.float32)
    valid_mask = np.zeros((int(bucket_chunks), int(chunk_size)), dtype=np.uint8)
    starts_arr = np.full((int(bucket_chunks),), -1, dtype=np.int32)
    valid_lengths = np.zeros((int(bucket_chunks),), dtype=np.int32)

    for idx, (start, valid_len) in enumerate(windows):
        valid_len = max(0, min(int(valid_len), int(chunk_size), raw_len - int(start)))
        chunk_pa = np.asarray(signal_pa[int(start) : int(start) + valid_len], dtype=np.float32)
        normalized, center, half_range = normalize_to_pm1_with_stats(chunk_pa)
        normalized = np.asarray(normalized, dtype=np.float32)
        if valid_len < int(chunk_size):
            normalized = _reflect_pad_right_1d(normalized, int(chunk_size))
        chunks[idx] = normalized[: int(chunk_size)]
        centers[idx] = float(center)
        half_ranges[idx] = float(half_range) if math.isfinite(float(half_range)) else 0.0
        valid_mask[idx, :valid_len] = 1
        starts_arr[idx] = int(start)
        valid_lengths[idx] = int(valid_len)

    padded_chunk_count = int(sum(1 for _start, valid_len in windows if int(valid_len) < int(chunk_size)))
    shift_last_applied = bool(
        str(tail_chunk_mode).strip().lower() == "shift_last"
        and raw_len > int(chunk_size)
        and ((raw_len - int(chunk_size)) % int(hop_size)) != 0
    )
    tokens, _token_mask, truth_len = _truth_to_tokens(truth.sequence, max_truth_len)
    if truth_len > int(max_truth_len):
        return None
    qscore_arr = _pad_qscores(truth.qscores, max_truth_len)
    read_id = str(row["read_id"])
    out_row = {
        "read_id": read_id,
        "source_pod5": str(source_pod5),
        "barcode": row.get("barcode"),
        "flowcell": row.get("flowcell"),
        "read_length_bases": int(row.get("read_length_bases") or row.get("read_length") or 0),
        "identity": float(row.get("identity")) if row.get("identity") is not None else None,
        "accuracy": float(row.get("accuracy")) if row.get("accuracy") is not None else None,
        "mean_quality": float(row.get("mean_quality")) if row.get("mean_quality") is not None else None,
        "raw_signal_length_samples": int(raw_len),
        "chunk_count": int(len(windows)),
        "bucket_chunks": int(bucket_chunks),
        "chunk_size_samples": int(chunk_size),
        "chunk_hop_samples": int(hop_size),
        "tail_chunk_mode": str(tail_chunk_mode),
        "padded_chunk_count": int(padded_chunk_count),
        "shift_last_applied": bool(shift_last_applied),
        "truth_length": int(truth_len),
        "max_truth_len": int(max_truth_len),
    }
    return {
        "row": out_row,
        "chunks_norm": chunks,
        "chunk_center": centers,
        "chunk_half_range": half_ranges,
        "chunk_valid_mask": valid_mask,
        "chunk_start": starts_arr,
        "chunk_valid_length": valid_lengths,
        "truth_tokens": tokens,
        "truth_qscores": qscore_arr,
    }


def _flush_bucket(
    *,
    buffer: BucketBuffer,
    bucket_dir: Path,
    shard_index: int,
    summary: dict[str, Any],
    manifest_handle: Any,
    args: argparse.Namespace,
) -> int:
    if len(buffer) <= 0:
        return shard_index
    bucket_dir.mkdir(parents=True, exist_ok=True)
    shard_name = f"shard_{shard_index:05d}.npz"
    shard_path = bucket_dir / shard_name
    metadata = {
        "schema_version": 1,
        "bucket_chunks": int(buffer.bucket_chunks),
        "max_truth_len": int(buffer.max_truth_len),
        "chunk_size_samples": int(args.chunk_size),
        "chunk_hop_samples": int(args.chunk_hop_samples),
        "overlap_samples": int(args.chunk_size - args.chunk_hop_samples),
        "tail_chunk_mode": str(args.tail_chunk_mode),
        "min_chunks": int(args.min_chunks),
        "max_chunks": None if args.max_chunks is None else int(args.max_chunks),
        "token_alphabet": {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4},
    }
    rows = list(buffer.rows)
    np.savez(
        shard_path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        read_id=np.asarray([str(row["read_id"]) for row in rows], dtype="U64"),
        source_pod5=np.asarray([str(row["source_pod5"]) for row in rows], dtype="U512"),
        barcode=np.asarray([str(row.get("barcode") or "") for row in rows], dtype="U32"),
        flowcell=np.asarray([str(row.get("flowcell") or "") for row in rows], dtype="U32"),
        read_length_bases=np.asarray([int(row["read_length_bases"]) for row in rows], dtype=np.int32),
        identity=np.asarray([float(row["identity"] or 0.0) for row in rows], dtype=np.float32),
        accuracy=np.asarray([float(row["accuracy"] or 0.0) for row in rows], dtype=np.float32),
        mean_quality=np.asarray([float(row["mean_quality"] or 0.0) for row in rows], dtype=np.float32),
        raw_signal_length_samples=np.asarray(
            [int(row["raw_signal_length_samples"]) for row in rows], dtype=np.int32
        ),
        chunk_count=np.asarray([int(row["chunk_count"]) for row in rows], dtype=np.int16),
        padded_chunk_count=np.asarray([int(row["padded_chunk_count"]) for row in rows], dtype=np.int16),
        shift_last_applied=np.asarray([int(bool(row["shift_last_applied"])) for row in rows], dtype=np.uint8),
        chunks_norm=np.stack(buffer.chunks_norm, axis=0).astype(np.float32, copy=False),
        chunk_center=np.stack(buffer.chunk_center, axis=0).astype(np.float32, copy=False),
        chunk_half_range=np.stack(buffer.chunk_half_range, axis=0).astype(np.float32, copy=False),
        chunk_valid_mask=np.stack(buffer.chunk_valid_mask, axis=0).astype(np.uint8, copy=False),
        chunk_start=np.stack(buffer.chunk_start, axis=0).astype(np.int32, copy=False),
        chunk_valid_length=np.stack(buffer.chunk_valid_length, axis=0).astype(np.int32, copy=False),
        valid_length=np.asarray([int(row["raw_signal_length_samples"]) for row in rows], dtype=np.int32),
        truth_tokens=np.stack(buffer.truth_tokens, axis=0).astype(np.int16, copy=False),
        truth_qscores=np.stack(buffer.truth_qscores, axis=0).astype(np.uint8, copy=False),
        truth_length=np.asarray([int(row["truth_length"]) for row in rows], dtype=np.int32),
        read_mask=np.ones((len(rows),), dtype=np.uint8),
    )
    rel_shard = shard_path.relative_to(Path(args.output_dir).expanduser().resolve())
    for row_idx, row in enumerate(rows):
        manifest_row = dict(row)
        manifest_row["shard"] = str(rel_shard)
        manifest_row["row_index"] = int(row_idx)
        manifest_handle.write(json.dumps(manifest_row, sort_keys=True) + "\n")
    bucket_key = f"bucket_{int(buffer.bucket_chunks):03d}"
    summary["bucket_counts"][bucket_key] = int(summary["bucket_counts"].get(bucket_key, 0)) + len(rows)
    summary["bucket_shards"][bucket_key] = int(summary["bucket_shards"].get(bucket_key, 0)) + 1
    summary["written_reads"] += len(rows)
    summary["written_shards"] += 1
    print(f"[write] {shard_path} reads={len(rows)} bucket={bucket_key}", flush=True)
    buffer.clear()
    return shard_index + 1


def _pending_buffer_reads(buffers: dict[int, BucketBuffer]) -> int:
    return int(sum(len(buffer) for buffer in buffers.values()))


def _flush_all_buffers(
    *,
    buffers: dict[int, BucketBuffer],
    shard_indices: dict[int, int],
    output_dir: Path,
    summary: dict[str, Any],
    manifest_handle: Any,
    args: argparse.Namespace,
) -> None:
    for bucket, buffer in buffers.items():
        shard_indices[bucket] = _flush_bucket(
            buffer=buffer,
            bucket_dir=output_dir / "shards" / f"bucket_{bucket:03d}",
            shard_index=shard_indices[bucket],
            summary=summary,
            manifest_handle=manifest_handle,
            args=args,
        )


def _default_output_dir(args: argparse.Namespace) -> Path:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    subset = f"subset{int(args.max_reads)}" if args.max_reads is not None else "all"
    overlap = int(args.chunk_size) - int(args.chunk_hop_samples)
    return dataset_dir / "materialized" / f"chunk{int(args.chunk_size)}_overlap{overlap}_{subset}"


def _resolve_paths(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    dataset_path = dataset_dir / "dataset.json"
    dataset = _read_json(dataset_path) if dataset_path.exists() else {"paths": {}}
    if args.manifest is None:
        args.manifest = dataset.get("paths", {}).get("candidates_with_truth") or dataset_dir / "manifests" / "candidates_with_truth.jsonl"
    if args.truth_fastq is None:
        args.truth_fastq = dataset.get("paths", {}).get("truth_fastq") or dataset_dir / "truth" / "truth.fastq.gz"
    args.manifest = str(Path(args.manifest).expanduser().resolve())
    args.truth_fastq = str(Path(args.truth_fastq).expanduser().resolve())
    args.pod5_root = str(Path(args.pod5_root).expanduser().resolve())
    if args.chunk_hop_samples is None:
        args.chunk_hop_samples = int(args.chunk_size) - int(args.overlap_samples)
    args.chunk_hop_samples = max(1, int(args.chunk_hop_samples))
    if args.chunk_hop_samples > int(args.chunk_size):
        raise ValueError("--chunk-hop-samples cannot exceed --chunk-size.")
    if args.output_dir is None:
        args.output_dir = str(_default_output_dir(args))
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    _resolve_paths(args)
    manifest_path = Path(args.manifest)
    truth_fastq_path = Path(args.truth_fastq)
    pod5_root = Path(args.pod5_root)
    output_dir = Path(args.output_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not truth_fastq_path.exists():
        raise FileNotFoundError(f"Truth FASTQ not found: {truth_fastq_path}")
    if not pod5_root.exists():
        raise FileNotFoundError(f"POD5 root not found: {pod5_root}")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}; pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    bucket_chunks = _parse_bucket_chunks(args.bucket_chunks)
    candidates = _load_candidate_rows(
        manifest_path,
        max_reads=args.max_reads,
        selection_mode=str(args.selection_mode),
        random_seed=int(args.random_seed),
    )
    row_by_id = {str(row["read_id"]): row for row in candidates}
    pod5_files = _discover_pod5_files(pod5_root, str(args.pod5_glob))
    if args.max_pod5_files is not None:
        pod5_files = pod5_files[: max(1, int(args.max_pod5_files))]
    if not pod5_files:
        raise FileNotFoundError(f"No POD5 files found under {pod5_root}")
    if args.selection_mode == "pod5_first":
        if args.max_reads is None:
            raise ValueError("--selection-mode pod5_first requires --max-reads.")
        candidates, selected_pod5_files = _select_pod5_first_candidates(
            pod5_files=pod5_files,
            row_by_id=row_by_id,
            max_reads=int(args.max_reads),
            progress_every_files=int(args.progress_every_files),
            progress_every_reads=int(args.progress_every_reads),
        )
        row_by_id = {str(row["read_id"]): row for row in candidates}
        pod5_files = selected_pod5_files
    target_ids = {str(row["read_id"]) for row in candidates}
    stop_after_written = int(args.max_reads) if args.max_reads is not None and args.selection_mode == "pod5_first" else None
    print(f"[setup] candidate_reads={len(candidates)} selection={args.selection_mode}", flush=True)
    truth = _load_truth_fastq(truth_fastq_path, target_ids)
    missing_truth_ids = sorted(target_ids - set(truth))
    if missing_truth_ids:
        print(f"[warn] missing truth for {len(missing_truth_ids)} selected reads; first={missing_truth_ids[0]}", flush=True)
    target_ids &= set(truth)
    print(f"[setup] pod5_files={len(pod5_files)} target_reads={len(target_ids)}", flush=True)

    buffers = {
        bucket: BucketBuffer(
            bucket_chunks=bucket,
            max_truth_len=int(bucket) * int(args.max_truth_bases_per_chunk),
            shard_reads=max(1, int(args.shard_reads)),
        )
        for bucket in bucket_chunks
    }
    shard_indices = {bucket: 0 for bucket in bucket_chunks}
    summary: dict[str, Any] = {
        "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()),
        "output_dir": str(output_dir),
        "source_manifest": str(manifest_path),
        "source_truth_fastq": str(truth_fastq_path),
        "source_pod5_root": str(pod5_root),
        "chunk_size_samples": int(args.chunk_size),
        "chunk_hop_samples": int(args.chunk_hop_samples),
        "overlap_samples": int(args.chunk_size - args.chunk_hop_samples),
        "tail_chunk_mode": str(args.tail_chunk_mode),
        "min_chunks": int(args.min_chunks),
        "max_chunks": None if args.max_chunks is None else int(args.max_chunks),
        "bucket_chunks": list(bucket_chunks),
        "max_truth_bases_per_chunk": int(args.max_truth_bases_per_chunk),
        "selected_reads": int(len(candidates)),
        "target_reads_with_truth": int(len(target_ids)),
        "missing_truth_reads": int(len(missing_truth_ids)),
        "scanned_pod5_files": 0,
        "scanned_pod5_reads": 0,
        "matched_pod5_reads": 0,
        "written_reads": 0,
        "written_shards": 0,
        "bucket_counts": {},
        "bucket_shards": {},
        "skipped": {
            "missing_truth": int(len(missing_truth_ids)),
            "too_short_signal": 0,
            "too_few_chunks": 0,
            "too_many_chunks": 0,
            "truth_too_long": 0,
            "calibration_error": 0,
            "materialize_error": 0,
        },
        "tail": {
            "padded_chunk_reads": 0,
            "padded_chunks": 0,
            "shift_last_reads": 0,
        },
    }

    reads_manifest_path = output_dir / "manifests" / "reads.jsonl"
    reads_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    remaining = set(target_ids)
    progress_every_reads = max(1, int(args.progress_every_reads))
    progress_every_files = max(1, int(args.progress_every_files))

    with reads_manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for file_idx, pod5_file in enumerate(pod5_files, start=1):
            if not remaining:
                break
            if stop_after_written is not None and int(summary["written_reads"]) >= stop_after_written:
                break
            if file_idx == 1 or file_idx % progress_every_files == 0:
                print(
                    f"[scan] file={file_idx}/{len(pod5_files)} remaining={len(remaining)} "
                    f"written={summary['written_reads']}",
                    flush=True,
                )
            summary["scanned_pod5_files"] += 1
            try:
                with pod5.Reader(str(pod5_file)) as reader:
                    for record in reader.reads():
                        if stop_after_written is not None and int(summary["written_reads"]) >= stop_after_written:
                            break
                        summary["scanned_pod5_reads"] += 1
                        if summary["scanned_pod5_reads"] % progress_every_reads == 0:
                            elapsed = max(1e-6, time.time() - started_at)
                            print(
                                f"[scan] reads={summary['scanned_pod5_reads']} "
                                f"matched={summary['matched_pod5_reads']} written={summary['written_reads']} "
                                f"rate={summary['scanned_pod5_reads'] / elapsed:.0f}/s",
                                flush=True,
                            )
                        read_id = str(getattr(record, "read_id", ""))
                        if read_id not in remaining:
                            continue
                        summary["matched_pod5_reads"] += 1
                        row = row_by_id[read_id]
                        truth_record = truth[read_id]
                        try:
                            raw = np.asarray(record.signal, dtype=np.int16).reshape(-1)
                            calibration = parse_calibration(getattr(record, "calibration", None))
                            signal_pa = np.asarray(calibration.to_picoamps(raw), dtype=np.float32)
                        except CalibrationError:
                            summary["skipped"]["calibration_error"] += 1
                            remaining.discard(read_id)
                            continue
                        except Exception:
                            summary["skipped"]["materialize_error"] += 1
                            remaining.discard(read_id)
                            continue
                        windows = _build_windows(
                            int(signal_pa.shape[0]),
                            int(args.chunk_size),
                            int(args.chunk_hop_samples),
                            tail_chunk_mode=str(args.tail_chunk_mode),
                        )
                        n_chunks = int(len(windows))
                        bucket = _bucket_for_chunk_count(len(windows), bucket_chunks)
                        if int(signal_pa.shape[0]) < int(args.min_signal_samples):
                            summary["skipped"]["too_short_signal"] += 1
                            remaining.discard(read_id)
                            continue
                        if n_chunks < int(args.min_chunks):
                            summary["skipped"]["too_few_chunks"] += 1
                            remaining.discard(read_id)
                            continue
                        if args.max_chunks is not None and n_chunks > int(args.max_chunks):
                            summary["skipped"]["too_many_chunks"] += 1
                            remaining.discard(read_id)
                            continue
                        if bucket is None:
                            summary["skipped"]["too_many_chunks"] += 1
                            remaining.discard(read_id)
                            continue
                        max_truth_len = buffers[bucket].max_truth_len
                        if len(truth_record.sequence) > max_truth_len:
                            summary["skipped"]["truth_too_long"] += 1
                            remaining.discard(read_id)
                            continue
                        sample = _materialize_read(
                            row=row,
                            truth=truth_record,
                            signal_pa=signal_pa,
                            source_pod5=pod5_file,
                            bucket_chunks=bucket,
                            max_truth_len=max_truth_len,
                            chunk_size=int(args.chunk_size),
                            hop_size=int(args.chunk_hop_samples),
                            min_signal_samples=int(args.min_signal_samples),
                            tail_chunk_mode=str(args.tail_chunk_mode),
                        )
                        remaining.discard(read_id)
                        if sample is None:
                            summary["skipped"]["materialize_error"] += 1
                            continue
                        padded_chunk_count = int(sample["row"].get("padded_chunk_count") or 0)
                        if padded_chunk_count > 0:
                            summary["tail"]["padded_chunk_reads"] += 1
                            summary["tail"]["padded_chunks"] += padded_chunk_count
                        if bool(sample["row"].get("shift_last_applied")):
                            summary["tail"]["shift_last_reads"] += 1
                        buffers[bucket].append(sample)
                        if buffers[bucket].should_flush():
                            shard_indices[bucket] = _flush_bucket(
                                buffer=buffers[bucket],
                                bucket_dir=output_dir / "shards" / f"bucket_{bucket:03d}",
                                shard_index=shard_indices[bucket],
                                summary=summary,
                                manifest_handle=manifest_handle,
                                args=args,
                            )
                        if (
                            stop_after_written is not None
                            and int(summary["written_reads"]) + _pending_buffer_reads(buffers) >= stop_after_written
                        ):
                            _flush_all_buffers(
                                buffers=buffers,
                                shard_indices=shard_indices,
                                output_dir=output_dir,
                                summary=summary,
                                manifest_handle=manifest_handle,
                                args=args,
                            )
                            break
            except Exception as exc:
                print(f"[warn] skip POD5 file {pod5_file}: {exc}", flush=True)

        _flush_all_buffers(
            buffers=buffers,
            shard_indices=shard_indices,
            output_dir=output_dir,
            summary=summary,
            manifest_handle=manifest_handle,
            args=args,
        )

    summary["missing_pod5_reads"] = int(len(remaining))
    summary["paths"] = {
        "reads_manifest": str(reads_manifest_path),
        "shards": str(output_dir / "shards"),
        "summary": str(output_dir / "materialize_summary.json"),
    }
    elapsed = time.time() - started_at
    summary["elapsed_seconds"] = float(elapsed)
    summary["pod5_reads_per_second"] = float(summary["scanned_pod5_reads"] / max(1e-6, elapsed))
    _write_json(output_dir / "materialize_summary.json", summary)
    _write_json(output_dir / "dataset.json", summary)
    print(
        f"[done] written_reads={summary['written_reads']} shards={summary['written_shards']} "
        f"missing_pod5={summary['missing_pod5_reads']} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize POD5-backed post-training shards.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--truth-fastq", type=Path, default=None)
    parser.add_argument("--pod5-root", type=Path, default=DEFAULT_POD5_ROOT)
    parser.add_argument("--pod5-glob", type=str, default="*.pod5")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=6144)
    parser.add_argument("--overlap-samples", type=int, default=144)
    parser.add_argument("--chunk-hop-samples", type=int, default=None)
    parser.add_argument("--tail-chunk-mode", choices=("shift_last", "pad_last"), default="shift_last")
    parser.add_argument("--bucket-chunks", type=str, default="2,4,8,16,32")
    parser.add_argument("--min-chunks", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--max-truth-bases-per-chunk", type=int, default=1024)
    parser.add_argument("--min-signal-samples", type=int, default=2048)
    parser.add_argument("--shard-reads", type=int, default=128)
    parser.add_argument("--max-reads", type=int, default=None)
    parser.add_argument("--selection-mode", choices=("first", "random", "pod5_first"), default="first")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-pod5-files", type=int, default=None)
    parser.add_argument("--progress-every-files", type=int, default=25)
    parser.add_argument("--progress-every-reads", type=int, default=100000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    materialize(parse_args())


if __name__ == "__main__":
    main()
