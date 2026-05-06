from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


ARRAY_KEYS = (
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
    "read_id",
    "source_pod5",
    "read_length_bases",
    "identity",
    "accuracy",
    "mean_quality",
    "raw_signal_length_samples",
    "chunk_count",
    "padded_chunk_count",
    "shift_last_applied",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bucket_from_name(path: Path) -> int | None:
    name = path.parent.name
    if not name.startswith("bucket_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return None


def _load_npz_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        batch = {key: np.asarray(payload[key]) for key in ARRAY_KEYS if key in payload}
        metadata_raw = payload.get("metadata_json")
        if metadata_raw is not None:
            batch["metadata_json"] = np.asarray(metadata_raw)
    if "chunk_valid_mask" in batch:
        batch["chunk_valid_mask"] = batch["chunk_valid_mask"].astype(np.float32, copy=False)
    if "read_mask" in batch:
        batch["read_mask"] = batch["read_mask"].astype(np.float32, copy=False)
    return batch


def _take_rows(batch: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[indices] for key, value in batch.items() if key != "metadata_json"}


def _slice_rows(batch: dict[str, np.ndarray], start: int, stop: int) -> dict[str, np.ndarray]:
    return {key: value[start:stop] for key, value in batch.items() if key != "metadata_json"}


def _concat_batches(items: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not items:
        raise ValueError("Cannot concatenate an empty batch list.")
    keys = items[0].keys()
    return {key: np.concatenate([item[key] for item in items], axis=0) for key in keys}


@dataclass
class PosttrainShardDataset:
    root: Path
    shards_by_bucket: dict[int, list[Path]]
    metadata: dict[str, Any]

    @classmethod
    def from_dir(cls, root: str | Path) -> "PosttrainShardDataset":
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            raise FileNotFoundError(f"Posttrain materialized dataset not found: {root_path}")
        metadata_path = root_path / "dataset.json"
        metadata = _read_json(metadata_path) if metadata_path.exists() else {}
        shards_root = root_path / "shards"
        shards_by_bucket: dict[int, list[Path]] = {}
        for shard in sorted(shards_root.glob("bucket_*/*.npz")):
            bucket = _bucket_from_name(shard)
            if bucket is None:
                continue
            shards_by_bucket.setdefault(bucket, []).append(shard.resolve())
        if not shards_by_bucket:
            raise FileNotFoundError(f"No posttrain shards found under {shards_root}")
        return cls(root=root_path, shards_by_bucket=shards_by_bucket, metadata=metadata)

    @property
    def buckets(self) -> tuple[int, ...]:
        return tuple(sorted(self.shards_by_bucket))

    def shard_paths(self, bucket_chunks: int | None = None) -> list[Path]:
        if bucket_chunks is None:
            paths: list[Path] = []
            for bucket in self.buckets:
                paths.extend(self.shards_by_bucket[bucket])
            return paths
        bucket = int(bucket_chunks)
        if bucket not in self.shards_by_bucket:
            raise ValueError(f"Bucket {bucket} not found; available buckets={self.buckets}")
        return list(self.shards_by_bucket[bucket])

    def iter_shards(
        self,
        *,
        bucket_chunks: int | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterator[dict[str, np.ndarray]]:
        paths = self.shard_paths(bucket_chunks)
        if shuffle:
            rng = random.Random(int(seed))
            rng.shuffle(paths)
        for path in paths:
            batch = _load_npz_shard(path)
            batch["shard_path"] = np.asarray([str(path)] * int(batch["chunks_norm"].shape[0]), dtype="U512")
            yield batch

    def batches(
        self,
        *,
        bucket_chunks: int,
        batch_size: int,
        shuffle_shards: bool = False,
        shuffle_rows: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ) -> Iterator[dict[str, np.ndarray]]:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        pending: dict[str, np.ndarray] | None = None
        row_rng = np.random.default_rng(int(seed))
        for shard in self.iter_shards(bucket_chunks=bucket_chunks, shuffle=shuffle_shards, seed=seed):
            n = int(shard["chunks_norm"].shape[0])
            if shuffle_rows and n > 1:
                shard = _take_rows(shard, row_rng.permutation(n))
            pending = shard if pending is None else _concat_batches((pending, shard))
            while int(pending["chunks_norm"].shape[0]) >= batch_size:
                yield _slice_rows(pending, 0, batch_size)
                remaining = int(pending["chunks_norm"].shape[0]) - batch_size
                if remaining <= 0:
                    pending = None
                    break
                pending = _slice_rows(pending, batch_size, batch_size + remaining)
        if pending is not None and int(pending["chunks_norm"].shape[0]) > 0 and not drop_last:
            yield pending
