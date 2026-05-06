#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from pathlib import Path
from typing import Any

try:
    import pysam
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(f"pysam is required: {exc}") from exc


DEFAULT_DATASET_DIR = Path("/data_nvme/simvqgan_posttrain_sup_truth_shiftlast_5to8/FC01_barcode01_id_gt_99")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_candidates(path: Path, *, max_candidates: int | None) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            read_id = str(row.get("read_id") or "")
            if not read_id or read_id in candidates:
                continue
            candidates[read_id] = row
            if max_candidates is not None and len(candidates) >= max(1, int(max_candidates)):
                break
    return candidates


def _mean_qscore(quality: str) -> float:
    if not quality:
        return 0.0
    vals = [max(0, ord(ch) - 33) for ch in quality]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _quality_string(record: Any, sequence_length: int) -> str:
    try:
        qualities = record.get_forward_qualities()
    except Exception:
        qualities = None
    if qualities is None:
        return "I" * int(sequence_length)
    text = "".join(chr(max(0, int(q)) + 33) for q in qualities)
    if len(text) != int(sequence_length):
        return "I" * int(sequence_length)
    return text


def _output_paths(args: argparse.Namespace, dataset: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    candidates_path = (
        Path(args.candidates).expanduser().resolve()
        if args.candidates is not None
        else Path(dataset["paths"]["candidates"]).expanduser().resolve()
    )
    enriched_path = (
        Path(args.output_manifest).expanduser().resolve()
        if args.output_manifest is not None
        else dataset_dir / "manifests" / "candidates_with_truth.jsonl"
    )
    truth_fastq_path = (
        Path(args.truth_fastq).expanduser().resolve()
        if args.truth_fastq is not None
        else dataset_dir / "truth" / "truth.fastq.gz"
    )
    missing_path = (
        Path(args.missing_output).expanduser().resolve()
        if args.missing_output is not None
        else dataset_dir / "manifests" / "missing_truth_read_ids.txt"
    )
    return candidates_path, enriched_path, truth_fastq_path, missing_path


def enrich_truth(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    dataset_path = dataset_dir / "dataset.json"
    dataset = _read_json(dataset_path) if dataset_path.exists() else {"paths": {}}
    candidates_path, enriched_path, truth_fastq_path, missing_path = _output_paths(args, dataset)
    bam_path = (
        Path(args.bam).expanduser().resolve()
        if args.bam is not None
        else Path(dataset.get("source_bam") or "").expanduser().resolve()
    )

    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidate manifest not found: {candidates_path}")
    if not bam_path.exists():
        raise FileNotFoundError(f"BAM not found: {bam_path}")
    for path in (enriched_path, truth_fastq_path, missing_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {path}; pass --overwrite to replace it.")

    candidates = _load_candidates(candidates_path, max_candidates=args.max_candidates)
    target_ids = set(candidates)
    found_ids: set[str] = set()
    skipped_secondary = 0
    skipped_empty_sequence = 0
    skipped_duplicate = 0
    scanned_records = 0
    matched_candidate_records = 0
    started_at = time.time()

    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    truth_fastq_path.parent.mkdir(parents=True, exist_ok=True)
    missing_path.parent.mkdir(parents=True, exist_ok=True)

    progress_every = max(1, int(args.progress_every))
    with pysam.AlignmentFile(str(bam_path), "rb") as bam, enriched_path.open(
        "w", encoding="utf-8"
    ) as manifest_handle, gzip.open(truth_fastq_path, "wt", encoding="utf-8") as fastq_handle:
        for record in bam.fetch(until_eof=True):
            scanned_records += 1
            if scanned_records % progress_every == 0:
                elapsed = max(1e-6, time.time() - started_at)
                rate = scanned_records / elapsed
                print(
                    "[scan] "
                    f"records={scanned_records} found={len(found_ids)}/{len(target_ids)} "
                    f"rate={rate:.0f}/s",
                    flush=True,
                )
            read_id = str(record.query_name)
            if read_id not in target_ids:
                continue
            matched_candidate_records += 1
            if read_id in found_ids:
                skipped_duplicate += 1
                continue
            if not args.include_secondary and (record.is_secondary or record.is_supplementary):
                skipped_secondary += 1
                continue
            sequence = record.get_forward_sequence()
            if not sequence:
                skipped_empty_sequence += 1
                continue
            sequence = str(sequence)
            quality = _quality_string(record, len(sequence))
            candidate = dict(candidates[read_id])
            candidate.update(
                {
                    "truth_length": int(len(sequence)),
                    "truth_mean_qscore": float(_mean_qscore(quality)),
                    "truth_fastq": str(truth_fastq_path),
                    "truth_source": "haplotagged_bam_query_sequence",
                    "truth_from_primary_only": not bool(args.include_secondary),
                    "bam_is_reverse": bool(record.is_reverse),
                    "bam_is_secondary": bool(record.is_secondary),
                    "bam_is_supplementary": bool(record.is_supplementary),
                    "bam_mapping_quality": int(record.mapping_quality),
                }
            )
            manifest_handle.write(json.dumps(candidate, sort_keys=True) + "\n")
            fastq_handle.write(f"@{read_id}\n{sequence}\n+\n{quality}\n")
            found_ids.add(read_id)
            if len(found_ids) >= len(target_ids):
                break

    missing = sorted(target_ids - found_ids)
    missing_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")

    elapsed = time.time() - started_at
    truth_lengths = []
    truth_qscores = []
    # Keep summary computation lightweight by reading only the enriched manifest.
    with enriched_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            truth_lengths.append(int(row["truth_length"]))
            qscore = row.get("truth_mean_qscore")
            if qscore is not None and math.isfinite(float(qscore)):
                truth_qscores.append(float(qscore))

    summary = {
        "dataset_dir": str(dataset_dir),
        "source_candidates": str(candidates_path),
        "source_bam": str(bam_path),
        "paths": {
            "candidates_with_truth": str(enriched_path),
            "truth_fastq": str(truth_fastq_path),
            "missing_truth_read_ids": str(missing_path),
        },
        "counts": {
            "candidate_reads": int(len(target_ids)),
            "truth_reads": int(len(found_ids)),
            "missing_truth_reads": int(len(missing)),
            "scanned_bam_records": int(scanned_records),
            "matched_candidate_records": int(matched_candidate_records),
            "skipped_secondary_or_supplementary": int(skipped_secondary),
            "skipped_empty_sequence": int(skipped_empty_sequence),
            "skipped_duplicate_candidate_records": int(skipped_duplicate),
        },
        "truth_length": _summary_stats(truth_lengths),
        "truth_mean_qscore": _summary_stats(truth_qscores),
        "elapsed_seconds": float(elapsed),
        "records_per_second": float(scanned_records / max(1e-6, elapsed)),
    }

    summary_path = dataset_dir / "truth" / "truth_summary.json"
    _write_json(summary_path, summary)
    summary["paths"]["truth_summary"] = str(summary_path)

    should_update_dataset = (
        args.max_candidates is None
        and args.output_manifest is None
        and args.truth_fastq is None
        and args.missing_output is None
    )
    if should_update_dataset and dataset_path.exists():
        dataset = dict(dataset)
        dataset.setdefault("paths", {})
        dataset["paths"].update(summary["paths"])
        dataset.setdefault("counts", {})
        dataset["counts"].update(
            {
                "truth_reads": int(len(found_ids)),
                "missing_truth_reads": int(len(missing)),
            }
        )
        dataset["truth_enrichment"] = {
            "source": "haplotagged_bam_query_sequence",
            "primary_only": not bool(args.include_secondary),
            "summary": str(summary_path),
        }
        _write_json(dataset_path, dataset)

    return summary


def _summary_stats(values: list[int] | list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p1": None,
            "p5": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.percentile(arr, 0)),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.percentile(arr, 100)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich post-training candidates with BAM query truth sequences.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--bam", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--truth-fastq", type=Path, default=None)
    parser.add_argument("--missing-output", type=Path, default=None)
    parser.add_argument("--include-secondary", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=200000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = enrich_truth(parse_args())
    print(json.dumps(summary["counts"], indent=2, sort_keys=True), flush=True)
    print(f"[output] {summary['paths']['candidates_with_truth']}", flush=True)
    print(f"[output] {summary['paths']['truth_fastq']}", flush=True)


if __name__ == "__main__":
    main()
