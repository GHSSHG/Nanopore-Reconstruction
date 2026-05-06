#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANALYSIS_DIR = Path("/data/nanopore/hereditary_cancer_2025.09/analysis/FC01/barcode01")
DEFAULT_OUTPUT_ROOT = Path("/data_nvme/simvqgan_posttrain_sup_truth_shiftlast_5to8")

LENGTH_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("0-100", 0, 100),
    ("100-250", 100, 250),
    ("250-500", 250, 500),
    ("500-750", 500, 750),
    ("750-1000", 750, 1000),
    ("1000-1500", 1000, 1500),
    ("1500-2000", 1500, 2000),
    ("2000-4000", 2000, 4000),
    ("4000-6000", 4000, 6000),
    ("6000-8000", 6000, 8000),
    ("8000-10000", 8000, 10000),
    ("10000-12000", 10000, 12000),
    ("12000-15000", 12000, 15000),
    ("15000-20000", 15000, 20000),
    ("20000-30000", 20000, 30000),
    ("30000+", 30000, None),
)

LENGTH_WINDOWS: tuple[tuple[str, int, int | None], ...] = (
    ("250-1600", 250, 1600),
    ("350-1300", 350, 1300),
    ("450-1000", 450, 1000),
    ("500-1500", 500, 1500),
    ("1000-3000", 1000, 3000),
    ("3000-6000", 3000, 6000),
    ("6000-12000", 6000, 12000),
    ("12000-15000", 12000, 15000),
    ("15000+", 15000, None),
)

IDENTITY_THRESHOLDS = (95.0, 98.0, 99.0, 99.5)
ACCURACY_THRESHOLDS = (90.0, 95.0, 98.0, 99.0)


def _parse_float(value: Any) -> float:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "na", "none", "null"}:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def _parse_int(value: Any) -> int | None:
    parsed = _parse_float(value)
    if not math.isfinite(parsed):
        return None
    return int(round(parsed))


def _identity_pass(identity: float, *, threshold: float, op: str) -> bool:
    if not math.isfinite(identity):
        return False
    if op == "gt":
        return identity > threshold
    if op == "ge":
        return identity >= threshold
    raise ValueError(f"Unsupported identity op: {op}")


def _bucket_label(value: int, bins: tuple[tuple[str, int, int | None], ...]) -> str | None:
    for label, lo, hi in bins:
        if value >= lo and (hi is None or value < hi):
            return label
    return None


def _in_window(value: int, lo: int, hi: int | None) -> bool:
    return value >= lo and (hi is None or value <= hi)


def _summarize(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
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
        "std": float(np.std(arr)),
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


def _n50(values: list[int]) -> int | None:
    if not values:
        return None
    import numpy as np

    arr = np.sort(np.asarray(values, dtype=np.int64))[::-1]
    csum = np.cumsum(arr)
    idx = int(np.searchsorted(csum, float(arr.sum()) / 2.0, side="left"))
    return int(arr[min(idx, arr.size - 1)])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _default_output_dir(*, output_root: Path, fc: str, barcode: str, identity_op: str, threshold: float) -> Path:
    op_label = "gt" if identity_op == "gt" else "ge"
    threshold_label = str(float(threshold)).rstrip("0").rstrip(".").replace(".", "p")
    return output_root / f"{fc}_{barcode}_id_{op_label}_{threshold_label}"


def build_candidates(args: argparse.Namespace) -> dict[str, Any]:
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    fc = str(args.fc)
    barcode = str(args.barcode)
    readstats_path = Path(args.readstats).expanduser().resolve() if args.readstats else analysis_dir / f"{barcode}.readstats.tsv.gz"
    bam_path = Path(args.bam).expanduser().resolve() if args.bam else analysis_dir / f"{barcode}.haplotagged.bam"
    output_root = Path(args.output_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir(
            output_root=output_root,
            fc=fc,
            barcode=barcode,
            identity_op=str(args.identity_op),
            threshold=float(args.identity_threshold),
        )
    )

    if not readstats_path.exists():
        raise FileNotFoundError(f"readstats not found: {readstats_path}")
    if not bam_path.exists():
        raise FileNotFoundError(f"BAM not found: {bam_path}")
    candidates_path = output_dir / "manifests" / "candidates.jsonl"
    if candidates_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {candidates_path}; pass --overwrite to replace it.")

    total_rows = 0
    unique_names: set[str] = set()
    valid_alignment_rows = 0
    duplicate_name_rows = 0
    unmapped_or_no_ref_rows = 0
    invalid_identity_rows = 0
    missing_read_length_rows = 0

    valid_read_lengths: list[int] = []
    valid_identities: list[float] = []
    valid_accuracies: list[float] = []
    valid_qualities: list[float] = []

    candidate_read_lengths: list[int] = []
    candidate_identities: list[float] = []
    candidate_accuracies: list[float] = []
    candidate_qualities: list[float] = []

    length_bin_valid = Counter()
    length_bin_candidate = Counter()
    window_valid = Counter()
    window_candidate = Counter()
    identity_counts_valid = Counter()
    identity_counts_candidate = Counter()
    accuracy_counts_valid = Counter()
    accuracy_counts_candidate = Counter()

    candidates: list[dict[str, Any]] = []
    max_candidates = None if args.max_candidates is None else max(1, int(args.max_candidates))

    with gzip.open(readstats_path, "rt", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            total_rows += 1
            read_id = str(row.get("name") or "")
            if read_id in unique_names:
                duplicate_name_rows += 1
            unique_names.add(read_id)

            ref = str(row.get("ref") or "")
            identity = _parse_float(row.get("iden"))
            accuracy = _parse_float(row.get("acc"))
            read_length = _parse_int(row.get("read_length"))
            mean_quality = _parse_float(row.get("mean_quality"))
            if not ref or ref == "*":
                unmapped_or_no_ref_rows += 1
            if not math.isfinite(identity):
                invalid_identity_rows += 1
            if read_length is None or read_length <= 0:
                missing_read_length_rows += 1

            valid_alignment = bool(ref and ref != "*" and math.isfinite(identity))
            if not valid_alignment:
                continue
            valid_alignment_rows += 1

            if read_length is not None and read_length > 0:
                valid_read_lengths.append(read_length)
                label = _bucket_label(read_length, LENGTH_BINS)
                if label is not None:
                    length_bin_valid[label] += 1
                for window_label, lo, hi in LENGTH_WINDOWS:
                    if _in_window(read_length, lo, hi):
                        window_valid[window_label] += 1
            valid_identities.append(identity)
            if math.isfinite(accuracy):
                valid_accuracies.append(accuracy)
            if math.isfinite(mean_quality):
                valid_qualities.append(mean_quality)
            for threshold in IDENTITY_THRESHOLDS:
                if identity >= threshold:
                    identity_counts_valid[f"identity_ge_{threshold:g}"] += 1
            for threshold in ACCURACY_THRESHOLDS:
                if accuracy >= threshold:
                    accuracy_counts_valid[f"accuracy_ge_{threshold:g}"] += 1

            if read_length is None or read_length <= 0:
                continue
            if read_length < int(args.min_read_length):
                continue
            if args.max_read_length is not None and read_length > int(args.max_read_length):
                continue
            if not _identity_pass(identity, threshold=float(args.identity_threshold), op=str(args.identity_op)):
                continue
            if args.min_accuracy is not None and (not math.isfinite(accuracy) or accuracy < float(args.min_accuracy)):
                continue
            if args.min_mean_quality is not None and (
                not math.isfinite(mean_quality) or mean_quality < float(args.min_mean_quality)
            ):
                continue

            candidate = {
                "read_id": read_id,
                "flowcell": fc,
                "barcode": barcode,
                "read_length_bases": int(read_length),
                "identity": float(identity),
                "accuracy": None if not math.isfinite(accuracy) else float(accuracy),
                "mean_quality": None if not math.isfinite(mean_quality) else float(mean_quality),
                "ref": ref,
                "qstart": _parse_int(row.get("qstart")),
                "qend": _parse_int(row.get("qend")),
                "rstart": _parse_int(row.get("rstart")),
                "rend": _parse_int(row.get("rend")),
                "direction": str(row.get("direction") or ""),
                "source_analysis_dir": str(analysis_dir),
                "source_readstats": str(readstats_path),
                "source_bam": str(bam_path),
            }
            candidates.append(candidate)

            candidate_read_lengths.append(read_length)
            candidate_identities.append(identity)
            if math.isfinite(accuracy):
                candidate_accuracies.append(accuracy)
            if math.isfinite(mean_quality):
                candidate_qualities.append(mean_quality)
            label = _bucket_label(read_length, LENGTH_BINS)
            if label is not None:
                length_bin_candidate[label] += 1
            for window_label, lo, hi in LENGTH_WINDOWS:
                if _in_window(read_length, lo, hi):
                    window_candidate[window_label] += 1
            for threshold in IDENTITY_THRESHOLDS:
                if identity >= threshold:
                    identity_counts_candidate[f"identity_ge_{threshold:g}"] += 1
            for threshold in ACCURACY_THRESHOLDS:
                if accuracy >= threshold:
                    accuracy_counts_candidate[f"accuracy_ge_{threshold:g}"] += 1

            if max_candidates is not None and len(candidates) >= max_candidates:
                break

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(candidates_path, candidates)

    length_bin_rows = [
        {
            "bin": label,
            "valid_aligned": int(length_bin_valid[label]),
            "candidates": int(length_bin_candidate[label]),
        }
        for label, _, _ in LENGTH_BINS
    ]
    window_rows = [
        {
            "window": label,
            "valid_aligned": int(window_valid[label]),
            "candidates": int(window_candidate[label]),
        }
        for label, _, _ in LENGTH_WINDOWS
    ]
    _write_tsv(output_dir / "manifests" / "read_length_bins.tsv", length_bin_rows, ["bin", "valid_aligned", "candidates"])
    _write_tsv(
        output_dir / "manifests" / "read_length_windows.tsv",
        window_rows,
        ["window", "valid_aligned", "candidates"],
    )

    summary = {
        "name": output_dir.name,
        "flowcell": fc,
        "barcode": barcode,
        "source_analysis_dir": str(analysis_dir),
        "source_readstats": str(readstats_path),
        "source_bam": str(bam_path),
        "filter": {
            "identity_threshold": float(args.identity_threshold),
            "identity_op": str(args.identity_op),
            "min_read_length": int(args.min_read_length),
            "max_read_length": None if args.max_read_length is None else int(args.max_read_length),
            "min_accuracy": None if args.min_accuracy is None else float(args.min_accuracy),
            "min_mean_quality": None if args.min_mean_quality is None else float(args.min_mean_quality),
        },
        "counts": {
            "readstats_rows": int(total_rows),
            "unique_read_ids": int(len(unique_names)),
            "duplicate_name_rows": int(duplicate_name_rows),
            "valid_alignment_rows": int(valid_alignment_rows),
            "unmapped_or_no_ref_rows": int(unmapped_or_no_ref_rows),
            "invalid_identity_rows": int(invalid_identity_rows),
            "missing_read_length_rows": int(missing_read_length_rows),
            "candidates": int(len(candidates)),
        },
        "valid_aligned_stats": {
            "read_length": _summarize(valid_read_lengths),
            "read_length_n50": _n50(valid_read_lengths),
            "identity": _summarize(valid_identities),
            "accuracy": _summarize(valid_accuracies),
            "mean_quality": _summarize(valid_qualities),
            "identity_threshold_counts": dict(identity_counts_valid),
            "accuracy_threshold_counts": dict(accuracy_counts_valid),
        },
        "candidate_stats": {
            "read_length": _summarize(candidate_read_lengths),
            "read_length_n50": _n50(candidate_read_lengths),
            "identity": _summarize(candidate_identities),
            "accuracy": _summarize(candidate_accuracies),
            "mean_quality": _summarize(candidate_qualities),
            "identity_threshold_counts": dict(identity_counts_candidate),
            "accuracy_threshold_counts": dict(accuracy_counts_candidate),
        },
        "paths": {
            "candidates": str(candidates_path),
            "read_length_bins": str(output_dir / "manifests" / "read_length_bins.tsv"),
            "read_length_windows": str(output_dir / "manifests" / "read_length_windows.tsv"),
        },
        "next_stage": {
            "description": "Use this manifest to fetch truth sequences from BAM and materialize selected reads from POD5.",
            "touches_pod5": False,
        },
    }
    _write_json(output_dir / "dataset.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build readstats-only post-training candidate manifests.")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--readstats", type=Path, default=None)
    parser.add_argument("--bam", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fc", type=str, default="FC01")
    parser.add_argument("--barcode", type=str, default="barcode01")
    parser.add_argument("--identity-threshold", type=float, default=99.0)
    parser.add_argument("--identity-op", choices=("gt", "ge"), default="gt")
    parser.add_argument("--min-read-length", type=int, default=1)
    parser.add_argument("--max-read-length", type=int, default=None)
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--min-mean-quality", type=float, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = build_candidates(parse_args())
    print(json.dumps(summary["counts"], indent=2, sort_keys=True), flush=True)
    print(f"[output] {summary['paths']['candidates']}", flush=True)


if __name__ == "__main__":
    main()
