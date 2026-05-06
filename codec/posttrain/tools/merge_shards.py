#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


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


def _link_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    if mode == "symlink":
        dst.symlink_to(src)
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    raise ValueError(f"Unsupported link mode: {mode}")


def _reads_manifest_path(dataset_dir: Path, summary: dict[str, Any]) -> Path:
    value = summary.get("paths", {}).get("reads_manifest")
    if value:
        return Path(value).expanduser().resolve()
    return dataset_dir / "manifests" / "reads.jsonl"


def merge_shards(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}; pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_bucket_dir = output_dir / "shards" / f"bucket_{int(args.bucket):03d}"
    output_manifest = output_dir / "manifests" / "reads.jsonl"
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, Any]] = []
    shard_map: dict[str, str] = {}
    shard_index = 0
    total_reads = 0
    skipped_manifest_rows = 0
    chunk_hist = Counter()
    first_summary: dict[str, Any] | None = None

    for raw_input in args.inputs:
        dataset_dir = Path(raw_input).expanduser().resolve()
        summary_path = dataset_dir / "dataset.json"
        if not summary_path.exists():
            summary_path = dataset_dir / "materialize_summary.json"
        summary = _read_json(summary_path)
        if first_summary is None:
            first_summary = summary
        bucket_dir = dataset_dir / "shards" / f"bucket_{int(args.bucket):03d}"
        shards = sorted(bucket_dir.glob("*.npz"))
        if not shards:
            raise FileNotFoundError(f"No bucket {args.bucket} shards under {bucket_dir}")
        local_map: dict[str, str] = {}
        for src in shards:
            dst = output_bucket_dir / f"shard_{shard_index:05d}.npz"
            _link_file(src.resolve(), dst, str(args.link_mode))
            old_rel = str(src.relative_to(dataset_dir))
            new_rel = str(dst.relative_to(output_dir))
            local_map[old_rel] = new_rel
            shard_map[f"{dataset_dir}:{old_rel}"] = new_rel
            shard_index += 1

        manifest = _reads_manifest_path(dataset_dir, summary)
        source_reads = 0
        with output_manifest.open("a", encoding="utf-8") as out:
            for row in _iter_jsonl(manifest):
                old_shard = str(row.get("shard") or "")
                if old_shard not in local_map:
                    skipped_manifest_rows += 1
                    continue
                row = dict(row)
                row["source_materialized_dataset"] = str(dataset_dir)
                row["shard"] = local_map[old_shard]
                out.write(json.dumps(row, sort_keys=True) + "\n")
                total_reads += 1
                source_reads += 1
                chunk_hist[int(row.get("chunk_count") or 0)] += 1
        sources.append(
            {
                "dataset_dir": str(dataset_dir),
                "summary": str(summary_path),
                "reads": int(source_reads),
                "shards": int(len(shards)),
            }
        )

    if first_summary is None:
        raise ValueError("No input datasets provided.")
    bucket_key = f"bucket_{int(args.bucket):03d}"
    merged = {
        "dataset_dir": str(output_dir),
        "output_dir": str(output_dir),
        "source_datasets": sources,
        "chunk_size_samples": int(first_summary.get("chunk_size_samples", 6144)),
        "chunk_hop_samples": int(first_summary.get("chunk_hop_samples", 6000)),
        "overlap_samples": int(first_summary.get("overlap_samples", 144)),
        "tail_chunk_mode": first_summary.get("tail_chunk_mode", "shift_last"),
        "min_chunks": first_summary.get("min_chunks"),
        "max_chunks": first_summary.get("max_chunks"),
        "bucket_chunks": [int(args.bucket)],
        "bucket_counts": {bucket_key: int(total_reads)},
        "bucket_shards": {bucket_key: int(shard_index)},
        "chunk_count_hist": {str(key): int(value) for key, value in sorted(chunk_hist.items())},
        "written_reads": int(total_reads),
        "written_shards": int(shard_index),
        "skipped_manifest_rows": int(skipped_manifest_rows),
        "paths": {
            "reads_manifest": str(output_manifest),
            "shards": str(output_dir / "shards"),
            "summary": str(output_dir / "materialize_summary.json"),
        },
    }
    _write_json(output_dir / "dataset.json", merged)
    _write_json(output_dir / "materialize_summary.json", merged)
    print(json.dumps({"written_reads": total_reads, "written_shards": shard_index}, indent=2), flush=True)
    print(f"[output] {output_dir}", flush=True)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge materialized post-training shard datasets.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--bucket", type=int, default=8)
    parser.add_argument("--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    merge_shards(parse_args())


if __name__ == "__main__":
    main()
