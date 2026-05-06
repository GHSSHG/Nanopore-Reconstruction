from __future__ import annotations

import csv
import gzip
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pod5

from codec.data.pod5_processing import CalibrationError, parse_calibration
from valid.common import progress_markers, write_fastq_gz, write_json, write_jsonl


@dataclass(frozen=True)
class ManifestRead:
    source_file: str
    read_id: str
    raw_length: int


def load_manifest_reads(payload: dict[str, Any]) -> list[ManifestRead]:
    return [
        ManifestRead(
            source_file=str(item["source_file"]),
            read_id=str(item["read_id"]),
            raw_length=int(item["raw_length"]),
        )
        for item in payload.get("selected_reads", [])
    ]


def load_or_create_regular_manifest(
    *,
    manifest_path: Path,
    split: str,
    files: Sequence[Path],
    num_reads: int,
    min_read_length: int,
    max_read_length: int | None,
) -> tuple[list[ManifestRead], Path, list[str]]:
    current_file_set = {str(path.resolve()) for path in files}
    if manifest_path.exists():
        import json

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload_min = payload.get("min_read_length")
        payload_max = payload.get("max_read_length")
        if payload_min not in (None, int(min_read_length)):
            raise RuntimeError(
                f"Manifest {manifest_path} was created for min_read_length={payload_min}, requested {min_read_length}."
            )
        if payload_max != (None if max_read_length is None else int(max_read_length)):
            raise RuntimeError(
                f"Manifest {manifest_path} was created for max_read_length={payload_max}, requested {max_read_length}."
            )
        items = load_manifest_reads(payload)
        for item in items:
            if str(Path(item.source_file).resolve()) not in current_file_set:
                raise RuntimeError(
                    f"Manifest {manifest_path} includes source file outside current input set: {item.source_file}"
                )
        if len(items) < int(num_reads):
            warnings = list(payload.get("warnings", []))
            warnings.append(
                f"Manifest contains {len(items)} reads but {num_reads} were requested; proceeding with manifest contents."
            )
            return items, manifest_path, warnings
        return items[: int(num_reads)], manifest_path, list(payload.get("warnings", []))

    warnings: list[str] = []
    selected: list[ManifestRead] = []
    markers = progress_markers(len(files))
    print(f"[select] scanning {len(files)} POD5 files for split={split}", flush=True)
    for file_idx, file_path in enumerate(files, start=1):
        if markers and file_idx >= markers[0]:
            print(f"[select] file progress {file_idx}/{len(files)}", flush=True)
            markers.pop(0)
        with pod5.Reader(str(file_path)) as reader:
            for record in reader.reads():
                read_id = str(getattr(record, "read_id", ""))
                raw = record.signal
                raw_length = int(len(raw))
                if raw_length < int(min_read_length):
                    continue
                if max_read_length is not None and raw_length > int(max_read_length):
                    continue
                try:
                    parse_calibration(getattr(record, "calibration", None))
                except CalibrationError as exc:
                    if len(warnings) < 64:
                        warnings.append(f"skip {file_path.name}:{read_id} ({exc})")
                    continue
                selected.append(
                    ManifestRead(
                        source_file=str(file_path),
                        read_id=read_id,
                        raw_length=raw_length,
                    )
                )
                if len(selected) >= num_reads:
                    break
        if len(selected) >= num_reads:
            break

    if not selected:
        raise RuntimeError(
            f"No reads found for split={split!r} with min_read_length={min_read_length} and max_read_length={max_read_length}."
        )
    if len(selected) < num_reads:
        warnings.append(
            f"Requested {num_reads} reads but only found {len(selected)} matching reads; proceeding with the smaller set."
        )

    payload = {
        "status": "ok",
        "kind": "regular_valid",
        "split": split,
        "requested_read_count": int(num_reads),
        "selected_read_count": len(selected),
        "min_read_length": int(min_read_length),
        "max_read_length": None if max_read_length is None else int(max_read_length),
        "source_files": [str(path.resolve()) for path in files],
        "selected_reads": [item.__dict__ for item in selected],
        "warnings": warnings,
        "created_at_unix": float(time.time()),
    }
    write_json(manifest_path, payload)
    return selected, manifest_path, warnings


def _float_value(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    result = float(value)
    return 0.0 if math.isnan(result) else result


def _int_value(row: dict[str, str], key: str) -> int:
    return int(round(_float_value(row, key)))


def _candidate_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(item["analysis_acc"]),
        -float(item["analysis_coverage"]),
        -float(item["analysis_mean_quality"]),
        -int(item["analysis_read_length"]),
        str(item["read_id"]),
    )


def _stable_sample(*, items: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size <= 0 or not items:
        return []
    ordered = sorted(items, key=lambda item: (str(item["barcode"]), str(item["read_id"])))
    if sample_size >= len(ordered):
        return ordered
    rng = random.Random(int(seed))
    sampled_indices = sorted(rng.sample(range(len(ordered)), k=int(sample_size)))
    return [ordered[idx] for idx in sampled_indices]


def _stable_shuffle(*, items: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    ordered = sorted(items, key=lambda item: (str(item["barcode"]), str(item["read_id"])))
    rng = random.Random(int(seed))
    rng.shuffle(ordered)
    return ordered


def _load_candidates(
    *,
    readstats_path: Path,
    fc: str,
    barcode: str,
    min_read_length: int,
    quality_filter_mode: str,
    min_acc: float,
    min_coverage: float,
    min_mean_quality: float,
    max_qstart: int,
    max_tail: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    with gzip.open(readstats_path, "rt", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            read_length = _int_value(row, "read_length")
            qend = _int_value(row, "qend")
            qstart = _int_value(row, "qstart")
            tail = read_length - qend
            acc = _float_value(row, "acc")
            coverage = _float_value(row, "coverage")
            mean_quality = _float_value(row, "mean_quality")
            if read_length < int(min_read_length):
                continue
            if quality_filter_mode == "strict":
                if acc < float(min_acc):
                    continue
                if coverage < float(min_coverage):
                    continue
                if mean_quality < float(min_mean_quality):
                    continue
                if qstart > int(max_qstart):
                    continue
                if tail > int(max_tail):
                    continue
            candidates.append(
                {
                    "fc": fc,
                    "barcode": barcode,
                    "read_id": str(row["name"]),
                    "analysis_ref": str(row["ref"]),
                    "analysis_direction": str(row["direction"]),
                    "analysis_rstart": _int_value(row, "rstart"),
                    "analysis_rend": _int_value(row, "rend"),
                    "analysis_qstart": qstart,
                    "analysis_qend": qend,
                    "analysis_read_length": read_length,
                    "analysis_length": _int_value(row, "length"),
                    "analysis_acc": acc,
                    "analysis_iden": _float_value(row, "iden"),
                    "analysis_coverage": coverage,
                    "analysis_ref_coverage": _float_value(row, "ref_coverage"),
                    "analysis_mean_quality": mean_quality,
                    "analysis_match": _int_value(row, "match"),
                    "analysis_ins": _int_value(row, "ins"),
                    "analysis_del": _int_value(row, "del"),
                    "analysis_sub": _int_value(row, "sub"),
                    "analysis_duplex": int(_float_value(row, "duplex")),
                    "analysis_start_time": str(row.get("start_time") or ""),
                    "analysis_runid": str(row.get("runid") or ""),
                    "analysis_sample_name": str(row.get("sample_name") or ""),
                }
            )
    return candidates


def _map_read_ids_to_pod5(*, pod5_dir: Path, target_ids: set[str]) -> dict[str, dict[str, Any]]:
    remaining = set(target_ids)
    found: dict[str, dict[str, Any]] = {}
    files = sorted(pod5_dir.glob("*.pod5"))
    if not files:
        raise FileNotFoundError(f"No POD5 files found in {pod5_dir}")
    for file_path in files:
        if not remaining:
            break
        ordered = sorted(remaining)
        with pod5.Reader(str(file_path)) as reader:
            try:
                records = reader.reads(selection=ordered, missing_ok=True)
            except TypeError:
                records = reader.reads()
            for record in records:
                read_id = str(getattr(record, "read_id", ""))
                if read_id not in remaining:
                    continue
                found[read_id] = {
                    "source_file": str(file_path.resolve()),
                    "raw_length": int(len(record.signal)),
                }
                remaining.discard(read_id)
    return found


def _extract_truth_from_analysis_bam(*, bam_path: Path, candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    import pysam

    found: dict[str, dict[str, Any]] = {}

    def _record_to_truth_payload(record: Any, bam: Any) -> dict[str, Any] | None:
        sequence = record.get_forward_sequence()
        try:
            qualities = record.get_forward_qualities()
        except TypeError:
            qualities = None
        if not sequence:
            return None
        quality_string = "".join(chr(int(q) + 33) for q in (qualities or []))
        if len(quality_string) != len(sequence):
            quality_string = "I" * len(sequence)
        return {
            "truth_seq": sequence,
            "truth_qual": quality_string,
            "truth_length": len(sequence),
            "analysis_hp": int(record.get_tag("HP")) if record.has_tag("HP") else None,
            "analysis_mapq": int(record.mapping_quality),
            "analysis_bam_ref": bam.get_reference_name(record.reference_id) if record.reference_id >= 0 else None,
            "analysis_bam_start": int(record.reference_start) if record.reference_start >= 0 else None,
            "analysis_bam_end": int(record.reference_end) if record.reference_end is not None else None,
            "analysis_bam_is_reverse": bool(record.is_reverse),
        }

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for item in candidates:
            read_id = str(item["read_id"])
            if read_id in found:
                continue
            ref_name = str(item["analysis_ref"])
            start = max(0, int(item["analysis_rstart"]) - 512)
            stop = max(start + 1, int(item["analysis_rend"]) + 512)
            try:
                iterator = bam.fetch(ref_name, start, stop)
            except ValueError:
                iterator = ()
            for record in iterator:
                if str(record.query_name) != read_id:
                    continue
                payload = _record_to_truth_payload(record, bam)
                if payload is None:
                    continue
                found[read_id] = payload
                break

        missing_ids = {str(item["read_id"]) for item in candidates if str(item["read_id"]) not in found}
        if missing_ids:
            for record in bam.fetch(until_eof=True):
                read_id = str(record.query_name)
                if read_id not in missing_ids:
                    continue
                payload = _record_to_truth_payload(record, bam)
                if payload is None:
                    continue
                found[read_id] = payload
                missing_ids.discard(read_id)
                if not missing_ids:
                    break
    return found


def _merge_usable_rows(
    *,
    candidate_rows: Sequence[dict[str, Any]],
    pod5_map: dict[str, dict[str, Any]],
    truth_map: dict[str, dict[str, Any]],
    min_truth_length: int,
    warnings: list[str],
) -> list[dict[str, Any]]:
    usable: list[dict[str, Any]] = []
    for item in candidate_rows:
        read_id = str(item["read_id"])
        barcode = str(item["barcode"])
        pod5_entry = pod5_map.get(read_id)
        truth_entry = truth_map.get(read_id)
        if pod5_entry is None:
            if len(warnings) < 64:
                warnings.append(f"Missing POD5 mapping for {barcode}:{read_id}")
            continue
        if truth_entry is None:
            if len(warnings) < 64:
                warnings.append(f"Missing truth entry for {barcode}:{read_id}")
            continue
        truth_length = int(truth_entry.get("truth_length") or 0)
        if truth_length < int(min_truth_length):
            if len(warnings) < 64:
                warnings.append(
                    f"Truth sequence shorter than {int(min_truth_length)} for {barcode}:{read_id} "
                    f"(truth_length={truth_length})"
                )
            continue
        merged = dict(item)
        merged.update(pod5_entry)
        merged.update(truth_entry)
        usable.append(merged)
    return usable


def _select_global_random_usable_reads(
    *,
    raw_root: Path,
    analysis_root: Path,
    fc: str,
    barcode_list: Sequence[str],
    candidate_rows_by_barcode: dict[str, list[dict[str, Any]]],
    target_total_reads: int,
    random_seed: int,
    min_truth_length: int,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], int]:
    all_candidates: list[dict[str, Any]] = []
    for barcode in barcode_list:
        all_candidates.extend(candidate_rows_by_barcode.get(barcode, []))

    ordered_candidates = _stable_shuffle(items=all_candidates, seed=random_seed)
    target_total_reads = max(0, int(target_total_reads))
    if target_total_reads <= 0:
        return [], {barcode: [] for barcode in barcode_list}, 0

    selected_reads: list[dict[str, Any]] = []
    usable_rows_by_barcode: dict[str, list[dict[str, Any]]] = {barcode: [] for barcode in barcode_list}
    batch_size = max(5000, target_total_reads * 3)
    processed_candidate_count = 0

    for batch_start in range(0, len(ordered_candidates), batch_size):
        if len(selected_reads) >= target_total_reads:
            break
        batch = ordered_candidates[batch_start : batch_start + batch_size]
        processed_candidate_count += len(batch)
        batch_ids = {str(item["read_id"]) for item in batch}
        print(
            "[manifest] probing candidates "
            f"{batch_start + 1}-{batch_start + len(batch)}/{len(ordered_candidates)} "
            f"(selected={len(selected_reads)}/{target_total_reads})",
            flush=True,
        )
        pod5_map = _map_read_ids_to_pod5(pod5_dir=raw_root / fc / "pod5", target_ids=batch_ids)

        truth_map: dict[str, dict[str, Any]] = {}
        for barcode in barcode_list:
            barcode_batch = [
                item
                for item in batch
                if str(item["barcode"]) == str(barcode) and str(item["read_id"]) in pod5_map
            ]
            if not barcode_batch:
                continue
            bam_path = analysis_root / fc / str(barcode) / f"{barcode}.haplotagged.bam"
            truth_map.update(_extract_truth_from_analysis_bam(bam_path=bam_path, candidates=barcode_batch))

        usable_batch = _merge_usable_rows(
            candidate_rows=batch,
            pod5_map=pod5_map,
            truth_map=truth_map,
            min_truth_length=min_truth_length,
            warnings=warnings,
        )
        for item in usable_batch:
            barcode = str(item["barcode"])
            usable_rows_by_barcode.setdefault(barcode, []).append(item)
            if len(selected_reads) < target_total_reads:
                selected_reads.append(item)

    if len(selected_reads) < target_total_reads:
        warnings.append(
            f"Requested {target_total_reads} reads in global_random mode but only found {len(selected_reads)} usable reads "
            f"after probing {processed_candidate_count} candidates."
        )
    return selected_reads, usable_rows_by_barcode, processed_candidate_count


def build_true_valid_manifest(
    *,
    analysis_root: Path,
    raw_root: Path,
    output_dir: Path,
    fc: str = "FC01",
    barcodes: Sequence[str] = ("barcode01", "barcode02", "barcode03"),
    selection_mode: str = "global_random",
    quality_filter_mode: str = "len_only",
    target_total_reads: int = 2000,
    target_per_barcode: int = 200,
    random_seed: int = 42,
    min_read_length: int = 6144,
    min_acc: float = 99.0,
    min_coverage: float = 98.0,
    min_mean_quality: float = 16.0,
    max_qstart: int = 100,
    max_tail: int = 100,
    truth_mode: str = "analysis_hac_proxy",
    dry_run: bool = False,
) -> dict[str, Any]:
    started_at = time.time()
    analysis_root = Path(analysis_root).expanduser().resolve()
    raw_root = Path(raw_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    truth_mode = str(truth_mode).strip().lower()
    if truth_mode != "analysis_hac_proxy":
        raise ValueError("Unsupported truth_mode; currently only 'analysis_hac_proxy' is implemented.")

    selection_mode = str(selection_mode).strip().lower()
    if selection_mode not in {"global_random", "per_barcode_top"}:
        raise ValueError("Unsupported selection_mode; choose from ['global_random', 'per_barcode_top'].")

    quality_filter_mode = str(quality_filter_mode).strip().lower()
    if quality_filter_mode not in {"len_only", "strict"}:
        raise ValueError("Unsupported quality_filter_mode; choose from ['len_only', 'strict'].")

    fc = str(fc).strip().upper()
    barcode_list = [str(item).strip() for item in barcodes if str(item).strip()]
    if not barcode_list:
        raise ValueError("At least one barcode is required.")

    candidate_rows_by_barcode: dict[str, list[dict[str, Any]]] = {}
    candidate_summary: dict[str, Any] = {}
    for barcode in barcode_list:
        readstats_path = analysis_root / fc / barcode / f"{barcode}.readstats.tsv.gz"
        bam_path = analysis_root / fc / barcode / f"{barcode}.haplotagged.bam"
        if not readstats_path.exists():
            raise FileNotFoundError(f"Missing readstats: {readstats_path}")
        if not bam_path.exists():
            raise FileNotFoundError(f"Missing haplotagged BAM: {bam_path}")
        candidates = _load_candidates(
            readstats_path=readstats_path,
            fc=fc,
            barcode=barcode,
            min_read_length=min_read_length,
            quality_filter_mode=quality_filter_mode,
            min_acc=min_acc,
            min_coverage=min_coverage,
            min_mean_quality=min_mean_quality,
            max_qstart=max_qstart,
            max_tail=max_tail,
        )
        for item in candidates:
            item["analysis_bam"] = str(bam_path.resolve())
        candidate_rows_by_barcode[barcode] = candidates
        candidate_summary[barcode] = {"candidate_read_count": int(len(candidates))}

    usable_rows_by_barcode: dict[str, list[dict[str, Any]]] = {}
    usable_rows: list[dict[str, Any]] = []
    truth_fastq_records: list[dict[str, str]] = []
    truth_metadata_rows: list[dict[str, Any]] = []
    group_summary: dict[str, Any] = {}
    warnings: list[str] = []
    processed_candidate_count = 0

    if selection_mode == "global_random":
        selected_reads, usable_rows_by_barcode, processed_candidate_count = _select_global_random_usable_reads(
            raw_root=raw_root,
            analysis_root=analysis_root,
            fc=fc,
            barcode_list=barcode_list,
            candidate_rows_by_barcode=candidate_rows_by_barcode,
            target_total_reads=target_total_reads,
            random_seed=random_seed,
            min_truth_length=min_read_length,
            warnings=warnings,
        )
        usable_rows = [item for items in usable_rows_by_barcode.values() for item in items]
    else:
        all_candidate_ids = {
            str(item["read_id"])
            for items in candidate_rows_by_barcode.values()
            for item in items
        }
        pod5_map = _map_read_ids_to_pod5(pod5_dir=raw_root / fc / "pod5", target_ids=all_candidate_ids)
        truth_map: dict[str, dict[str, Any]] = {}
        for barcode in barcode_list:
            bam_path = analysis_root / fc / barcode / f"{barcode}.haplotagged.bam"
            barcode_candidates = list(candidate_rows_by_barcode.get(barcode, []))
            truth_map.update(_extract_truth_from_analysis_bam(bam_path=bam_path, candidates=barcode_candidates))
        for barcode in barcode_list:
            barcode_candidates = list(candidate_rows_by_barcode.get(barcode, []))
            barcode_usable = _merge_usable_rows(
                candidate_rows=barcode_candidates,
                pod5_map=pod5_map,
                truth_map=truth_map,
                min_truth_length=min_read_length,
                warnings=warnings,
            )
            barcode_usable.sort(key=_candidate_sort_key)
            usable_rows_by_barcode[barcode] = barcode_usable
            usable_rows.extend(barcode_usable)
        selected_reads = []
        for barcode in barcode_list:
            barcode_usable = list(usable_rows_by_barcode.get(barcode, []))
            barcode_selected = barcode_usable[:target_per_barcode]
            if len(barcode_selected) < target_per_barcode:
                warnings.append(
                    f"{barcode}: requested {target_per_barcode} reads but only found {len(barcode_selected)} usable reads."
                )
            selected_reads.extend(barcode_selected)

    selected_reads.sort(key=lambda item: (str(item["barcode"]), str(item["read_id"])))
    selected_count_by_barcode = Counter(str(item["barcode"]) for item in selected_reads)

    for item in selected_reads:
        truth_fastq_records.append(
            {
                "read_id": str(item["read_id"]),
                "seq": str(item["truth_seq"]),
                "qual": str(item["truth_qual"]),
            }
        )
        truth_metadata_rows.append({key: value for key, value in item.items() if key not in {"truth_seq", "truth_qual"}})

    for barcode in barcode_list:
        barcode_selected = [item for item in selected_reads if str(item["barcode"]) == barcode]
        group_summary[barcode] = {
            "candidate_read_count": int(candidate_summary[barcode]["candidate_read_count"]),
            "usable_read_count": int(len(usable_rows_by_barcode.get(barcode, []))),
            "selected_read_count": int(len(barcode_selected)),
            "min_raw_length": int(min((item["raw_length"] for item in barcode_selected), default=0)),
            "max_raw_length": int(max((item["raw_length"] for item in barcode_selected), default=0)),
            "min_truth_length": int(min((item["truth_length"] for item in barcode_selected), default=0)),
            "max_truth_length": int(max((item["truth_length"] for item in barcode_selected), default=0)),
        }

    truth_dir = output_dir / "truth"
    truth_fastq_path = truth_dir / "analysis_hac_proxy_truth.fastq.gz"
    truth_metadata_path = truth_dir / "truth_metadata.jsonl"
    manifest_path = output_dir / "manifest" / f"true_valid_{fc.lower()}_{len(selected_reads)}reads_manifest.json"

    manifest_payload = {
        "status": "ok",
        "kind": "true_valid",
        "truth_mode": truth_mode,
        "analysis_root": str(analysis_root),
        "raw_root": str(raw_root),
        "fc": fc,
        "barcodes": barcode_list,
        "selection": {
            "selection_mode": selection_mode,
            "quality_filter_mode": quality_filter_mode,
            "random_seed": int(random_seed),
            "target_total_reads": int(target_total_reads),
            "target_per_barcode": int(target_per_barcode),
            "min_read_length": int(min_read_length),
            "min_acc": float(min_acc),
            "min_coverage": float(min_coverage),
            "min_mean_quality": float(min_mean_quality),
            "max_qstart": int(max_qstart),
            "max_tail": int(max_tail),
            "applied_filters": ["min_read_length"]
            if quality_filter_mode == "len_only"
            else ["min_read_length", "min_acc", "min_coverage", "min_mean_quality", "max_qstart", "max_tail"],
            "sort_order": (
                ["analysis_acc", "analysis_coverage", "analysis_mean_quality", "analysis_read_length"]
                if selection_mode == "per_barcode_top"
                else ["barcode", "read_id"]
            ),
        },
        "candidate_read_count": int(sum(len(items) for items in candidate_rows_by_barcode.values())),
        "usable_read_count": int(len(usable_rows)),
        "processed_candidate_count": int(processed_candidate_count),
        "selected_read_count": int(len(selected_reads)),
        "selected_barcode_counts": {barcode: int(selected_count_by_barcode.get(barcode, 0)) for barcode in barcode_list},
        "group_summary": group_summary,
        "selected_reads": [
            {key: value for key, value in item.items() if key not in {"truth_seq", "truth_qual"}}
            for item in selected_reads
        ],
        "paths": {
            "truth_fastq": str(truth_fastq_path),
            "truth_metadata": str(truth_metadata_path),
        },
        "warnings": warnings,
        "started_at_unix": float(started_at),
        "finished_at_unix": float(time.time()),
    }
    if dry_run:
        dry_run_payload = dict(manifest_payload)
        dry_run_payload["selected_read_examples"] = dry_run_payload.get("selected_reads", [])[:10]
        dry_run_payload.pop("selected_reads", None)
        return dry_run_payload

    write_fastq_gz(truth_fastq_path, truth_fastq_records)
    write_jsonl(truth_metadata_path, truth_metadata_rows)
    write_json(manifest_path, manifest_payload)
    manifest_payload["manifest_path"] = str(manifest_path)
    return manifest_payload
