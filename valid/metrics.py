from __future__ import annotations

import difflib
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from valid.common import mean_qscore, read_fastq, summarize

try:
    import edlib
except Exception:  # pragma: no cover - optional dependency
    edlib = None

EMPIRICAL_QSCORE_CAP = 60.0


def alignment_identity(a: str, b: str) -> tuple[float, int, int, str]:
    if not a and not b:
        return 1.0, 0, 0, "empty"
    if not a or not b:
        denom = max(len(a), len(b))
        return 0.0, denom, denom, "trivial"
    denom = max(len(a), len(b))
    if edlib is not None:
        res = edlib.align(a, b, mode="NW", task="distance")
        dist = int(res["editDistance"])
        identity = 1.0 - (float(dist) / float(denom))
        return identity, dist, denom, "edlib"
    ratio = difflib.SequenceMatcher(a=a, b=b).ratio()
    identity = float(ratio)
    approx_dist = int(round((1.0 - identity) * float(denom)))
    return identity, approx_dist, denom, "difflib"


def compute_regular_metrics(real_fastq: Path, generated_fastq: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    real = read_fastq(real_fastq)
    generated = read_fastq(generated_fastq)
    shared_ids = sorted(set(real) & set(generated))
    original_only = sorted(set(real) - set(generated))
    generated_only = sorted(set(generated) - set(real))

    per_read: list[dict[str, Any]] = []
    identities: list[float] = []
    length_deltas: list[float] = []
    qscore_deltas: list[float] = []
    exact_matches = 0
    total_weight = 0
    total_identity_weight = 0.0
    backend = "none"

    for read_id in shared_ids:
        real_entry = real[read_id]
        generated_entry = generated[read_id]
        identity, dist, denom, backend = alignment_identity(real_entry.seq, generated_entry.seq)
        real_len = len(real_entry.seq)
        generated_len = len(generated_entry.seq)
        qscore_real = mean_qscore(real_entry.qual)
        qscore_generated = mean_qscore(generated_entry.qual)
        qscore_delta = qscore_generated - qscore_real
        length_delta = float(generated_len - real_len)
        if real_entry.seq == generated_entry.seq:
            exact_matches += 1
        total_weight += denom
        total_identity_weight += identity * float(denom)
        identities.append(float(identity))
        length_deltas.append(length_delta)
        qscore_deltas.append(float(qscore_delta))
        per_read.append(
            {
                "read_id": read_id,
                "real_length": real_len,
                "generated_length": generated_len,
                "identity": float(identity),
                "edit_distance": int(dist),
                "identity_weight": int(denom),
                "qscore_real": float(qscore_real),
                "qscore_generated": float(qscore_generated),
                "qscore_delta": float(qscore_delta),
                "length_delta": length_delta,
                "exact_match": bool(real_entry.seq == generated_entry.seq),
            }
        )

    shared_count = len(shared_ids)
    summary = {
        "shared_read_count": int(shared_count),
        "original_read_count": int(len(real)),
        "reconstructed_read_count": int(len(generated)),
        "original_only_read_count": int(len(original_only)),
        "reconstructed_only_read_count": int(len(generated_only)),
        "exact_match_rate": float(exact_matches / shared_count) if shared_count else 0.0,
        "length_weighted_identity": float(total_identity_weight / total_weight) if total_weight else 0.0,
        "mean_qscore_delta": float(np.mean(np.asarray(qscore_deltas, dtype=np.float64))) if qscore_deltas else 0.0,
        "mean_qscore_delta_abs": (
            float(np.mean(np.abs(np.asarray(qscore_deltas, dtype=np.float64)))) if qscore_deltas else 0.0
        ),
        "identity_summary": summarize(identities),
        "length_delta_summary": summarize(length_deltas),
        "qscore_delta_summary": summarize(qscore_deltas),
        "identity_backend": backend,
    }
    return summary, per_read


def empirical_qscore(edit_distance: int, denom: int, *, cap: float = EMPIRICAL_QSCORE_CAP) -> float:
    denom = max(1, int(denom))
    errors = max(0, int(edit_distance))
    if errors <= 0:
        return float(cap)
    rate = max(float(errors) / float(denom), 1e-12)
    return float(min(cap, -10.0 * math.log10(rate)))


def metric_summary_from_records(
    records: Sequence[dict[str, Any]],
    *,
    truth_read_count: int,
    predicted_read_count: int,
) -> dict[str, Any]:
    identities = [float(record["identity"]) for record in records]
    empirical_qscores = [float(record["empirical_qscore"]) for record in records]
    predicted_qscores = [float(record["predicted_qscore"]) for record in records]
    truth_proxy_qscores = [float(record["truth_proxy_qscore"]) for record in records]
    length_deltas = [float(record["length_delta"]) for record in records]
    total_weight = int(sum(int(record["identity_weight"]) for record in records))
    total_identity_weight = float(
        sum(float(record["identity"]) * float(record["identity_weight"]) for record in records)
    )
    exact_matches = int(sum(1 for record in records if bool(record["exact_match"])))
    shared_count = int(len(records))
    return {
        "shared_read_count": shared_count,
        "truth_read_count": int(truth_read_count),
        "predicted_read_count": int(predicted_read_count),
        "truth_only_read_count": int(max(0, truth_read_count - shared_count)),
        "predicted_only_read_count": int(max(0, predicted_read_count - shared_count)),
        "exact_match_rate": (float(exact_matches) / float(shared_count)) if shared_count else 0.0,
        "length_weighted_identity": (float(total_identity_weight) / float(total_weight)) if total_weight else 0.0,
        "mean_predicted_qscore": float(np.mean(np.asarray(predicted_qscores, dtype=np.float64))) if predicted_qscores else 0.0,
        "mean_truth_proxy_qscore": (
            float(np.mean(np.asarray(truth_proxy_qscores, dtype=np.float64))) if truth_proxy_qscores else 0.0
        ),
        "mean_empirical_qscore": (
            float(np.mean(np.asarray(empirical_qscores, dtype=np.float64))) if empirical_qscores else 0.0
        ),
        "identity_summary": summarize(identities),
        "empirical_qscore_summary": summarize(empirical_qscores),
        "predicted_qscore_summary": summarize(predicted_qscores),
        "truth_proxy_qscore_summary": summarize(truth_proxy_qscores),
        "length_delta_summary": summarize(length_deltas),
    }


def compare_fastq_to_truth(
    *,
    predicted_fastq: Path,
    truth_entries: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predicted_entries = read_fastq(predicted_fastq)
    shared_ids = sorted(set(predicted_entries) & set(truth_entries))
    records: list[dict[str, Any]] = []
    for read_id in shared_ids:
        truth_entry = truth_entries[read_id]
        predicted_entry = predicted_entries[read_id]
        identity, edit_distance, denom, backend = alignment_identity(truth_entry.seq, predicted_entry.seq)
        record = {
            "read_id": read_id,
            "truth_length": int(len(truth_entry.seq)),
            "predicted_length": int(len(predicted_entry.seq)),
            "identity": float(identity),
            "edit_distance": int(edit_distance),
            "identity_weight": int(denom),
            "predicted_qscore": float(mean_qscore(predicted_entry.qual)),
            "truth_proxy_qscore": float(mean_qscore(truth_entry.qual)),
            "empirical_qscore": float(empirical_qscore(edit_distance, denom)),
            "length_delta": float(len(predicted_entry.seq) - len(truth_entry.seq)),
            "exact_match": bool(truth_entry.seq == predicted_entry.seq),
            "alignment_backend": backend,
        }
        records.append(record)
    summary = metric_summary_from_records(
        records,
        truth_read_count=len(truth_entries),
        predicted_read_count=len(predicted_entries),
    )
    summary["alignment_backend"] = records[0]["alignment_backend"] if records else "none"
    return summary, records


def summarize_group_metric_records(
    *,
    metric_records: Sequence[dict[str, Any]],
    grouped_read_ids: Sequence[str],
) -> dict[str, Any]:
    read_id_set = set(grouped_read_ids)
    selected = [record for record in metric_records if record["read_id"] in read_id_set]
    return metric_summary_from_records(
        selected,
        truth_read_count=len(read_id_set),
        predicted_read_count=len(selected),
    )


def compute_summary_delta(new_summary: dict[str, Any], old_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "delta_shared_read_count": int(new_summary.get("shared_read_count", 0) - old_summary.get("shared_read_count", 0)),
        "delta_length_weighted_identity": float(
            float(new_summary.get("length_weighted_identity", 0.0))
            - float(old_summary.get("length_weighted_identity", 0.0))
        ),
        "delta_mean_empirical_qscore": float(
            float(new_summary.get("mean_empirical_qscore", 0.0))
            - float(old_summary.get("mean_empirical_qscore", 0.0))
        ),
        "delta_mean_predicted_qscore": float(
            float(new_summary.get("mean_predicted_qscore", 0.0))
            - float(old_summary.get("mean_predicted_qscore", 0.0))
        ),
        "delta_exact_match_rate": float(
            float(new_summary.get("exact_match_rate", 0.0)) - float(old_summary.get("exact_match_rate", 0.0))
        ),
    }


def group_manifest_reads(manifest_payload: dict[str, Any], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in manifest_payload.get("selected_reads", []):
        group_value = str(item.get(key) or "unknown")
        grouped.setdefault(group_value, []).append(item)
    return grouped


def compute_group_summaries(
    *,
    manifest_payload: dict[str, Any],
    original_records: list[dict[str, Any]],
    generated_records: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped = group_manifest_reads(manifest_payload, "barcode")
    summary: dict[str, Any] = {}
    for barcode, items in grouped.items():
        read_ids = [str(item["read_id"]) for item in items]
        original_summary = summarize_group_metric_records(metric_records=original_records, grouped_read_ids=read_ids)
        generated_summary = summarize_group_metric_records(metric_records=generated_records, grouped_read_ids=read_ids)
        summary[barcode] = {
            "selected_manifest_read_count": int(len(read_ids)),
            "original_vs_truth": original_summary,
            "generated_vs_truth": generated_summary,
            "generated_minus_original": compute_summary_delta(generated_summary, original_summary),
        }
    return summary


_alignment_identity = alignment_identity
_mean_qscore = mean_qscore
_read_fastq = read_fastq
_summarize = summarize
