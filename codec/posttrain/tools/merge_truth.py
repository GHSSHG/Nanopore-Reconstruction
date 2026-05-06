#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import shutil
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


def _iter_fastq(path: Path) -> Iterable[tuple[str, str, str]]:
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
            yield read_id, seq.strip(), qual.strip()


def _resolve_input_paths(dataset_dir: Path) -> tuple[Path, Path]:
    dataset_path = dataset_dir / "dataset.json"
    dataset = _read_json(dataset_path) if dataset_path.exists() else {"paths": {}}
    manifest = Path(dataset.get("paths", {}).get("candidates_with_truth") or dataset_dir / "manifests" / "candidates_with_truth.jsonl")
    truth_fastq = Path(dataset.get("paths", {}).get("truth_fastq") or dataset_dir / "truth" / "truth.fastq.gz")
    return manifest.expanduser().resolve(), truth_fastq.expanduser().resolve()


def merge_truth(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}; pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    manifest_out = output_dir / "manifests" / "candidates_with_truth.jsonl"
    missing_out = output_dir / "manifests" / "missing_truth_read_ids.txt"
    truth_out = output_dir / "truth" / "truth.fastq.gz"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    truth_out.parent.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, Any]] = []
    read_ids: set[str] = set()
    duplicate_manifest_reads = 0
    manifest_rows = 0
    with manifest_out.open("w", encoding="utf-8") as out:
        for raw_input in args.inputs:
            dataset_dir = Path(raw_input).expanduser().resolve()
            manifest, truth_fastq = _resolve_input_paths(dataset_dir)
            dataset = _read_json(dataset_dir / "dataset.json") if (dataset_dir / "dataset.json").exists() else {}
            if not manifest.exists():
                raise FileNotFoundError(f"Manifest not found: {manifest}")
            if not truth_fastq.exists():
                raise FileNotFoundError(f"Truth FASTQ not found: {truth_fastq}")
            source_count = 0
            for row in _iter_jsonl(manifest):
                read_id = str(row.get("read_id") or "")
                if not read_id:
                    continue
                if read_id in read_ids:
                    duplicate_manifest_reads += 1
                    continue
                read_ids.add(read_id)
                out.write(json.dumps(row, sort_keys=True) + "\n")
                manifest_rows += 1
                source_count += 1
            sources.append(
                {
                    "dataset_dir": str(dataset_dir),
                    "manifest": str(manifest),
                    "truth_fastq": str(truth_fastq),
                    "rows": int(source_count),
                    "flowcell": dataset.get("flowcell"),
                    "barcode": dataset.get("barcode"),
                }
            )

    remaining = set(read_ids)
    truth_reads = 0
    with gzip.open(truth_out, "wt", encoding="utf-8") as out:
        for source in sources:
            truth_fastq = Path(str(source["truth_fastq"]))
            for read_id, seq, qual in _iter_fastq(truth_fastq):
                if read_id not in remaining:
                    continue
                out.write(f"@{read_id}\n{seq}\n+\n{qual}\n")
                remaining.remove(read_id)
                truth_reads += 1

    missing = sorted(remaining)
    missing_out.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    summary = {
        "name": output_dir.name,
        "dataset_dir": str(output_dir),
        "sources": sources,
        "counts": {
            "manifest_reads": int(manifest_rows),
            "truth_reads": int(truth_reads),
            "missing_truth_reads": int(len(missing)),
            "duplicate_manifest_reads": int(duplicate_manifest_reads),
        },
        "paths": {
            "candidates_with_truth": str(manifest_out),
            "truth_fastq": str(truth_out),
            "missing_truth_read_ids": str(missing_out),
        },
        "truth_enrichment": {
            "source": "merged_haplotagged_bam_query_sequence",
            "primary_only": True,
        },
    }
    _write_json(output_dir / "dataset.json", summary)
    _write_json(output_dir / "truth" / "truth_summary.json", summary)
    print(json.dumps(summary["counts"], indent=2, sort_keys=True), flush=True)
    print(f"[output] {output_dir}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge post-training candidate/truth datasets.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    merge_truth(parse_args())


if __name__ == "__main__":
    main()
