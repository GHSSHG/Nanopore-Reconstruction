#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from codec.runtime import configure_runtime_env, enable_jax_compilation_cache

configure_runtime_env()
enable_jax_compilation_cache()

import jax
import jax.numpy as jnp
import numpy as np

from codec.data import PosttrainShardDataset
from codec.dorado import dorado_crf_nll, extract_dorado_crf_logits, load_dorado_encoder_state
from codec.train import dorado_logit_lengths, stitch_chunks_to_pa


NUMERIC_KEYS = {
    "chunks_norm",
    "chunk_center",
    "chunk_half_range",
    "chunk_valid_mask",
    "chunk_start",
    "valid_length",
    "truth_tokens",
    "truth_length",
    "read_mask",
}


def _numeric_batch(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for key in NUMERIC_KEYS:
        arr = np.asarray(batch[key])
        if key in {"truth_tokens", "truth_length", "valid_length", "chunk_start"}:
            out[key] = arr.astype(np.int32, copy=False)
        else:
            out[key] = arr.astype(np.float32, copy=False)
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Dorado SUP CRF NLL on materialized post-train shards.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dorado-model", type=Path, default=Path("~/Download/dorado/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0"))
    parser.add_argument("--bucket", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--noise-std-pa", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = PosttrainShardDataset.from_dir(args.dataset)
    dorado_state = load_dorado_encoder_state(model_path=args.dorado_model, layers=("crf_logits",))
    output_length = int(args.bucket) * int(dataset.metadata.get("chunk_size_samples", 6144))
    rng = jax.random.PRNGKey(0)

    @jax.jit
    def _batch_nll(batch: dict[str, jnp.ndarray], noise_key: jax.Array) -> dict[str, jnp.ndarray]:
        read_pa, _weights = stitch_chunks_to_pa(
            batch["chunks_norm"],
            batch["chunk_center"],
            batch["chunk_half_range"],
            batch["chunk_valid_mask"],
            batch["chunk_start"],
            output_length=output_length,
        )
        noise = jax.random.normal(noise_key, read_pa.shape, dtype=jnp.float32) * jnp.asarray(
            float(args.noise_std_pa), dtype=jnp.float32
        )
        read_mask = jnp.asarray(batch["read_mask"], dtype=jnp.float32)
        logit_lengths = dorado_logit_lengths(batch["valid_length"])

        def _score(signal_pa: jnp.ndarray) -> jnp.ndarray:
            dorado_input = (signal_pa - jnp.asarray(dorado_state.pa_mean, dtype=jnp.float32)) / jnp.asarray(
                dorado_state.pa_std, dtype=jnp.float32
            )
            scores = extract_dorado_crf_logits(dorado_input, dorado_state)
            return dorado_crf_nll(
                scores,
                batch["truth_tokens"],
                batch["truth_length"],
                logit_lengths,
                normalise_scores=True,
            )

        original = _score(read_pa)
        noisy = _score(read_pa + noise)
        denom = jnp.maximum(jnp.sum(read_mask), 1.0)
        return {
            "original_nll": jnp.sum(original * read_mask) / denom,
            "noisy_nll": jnp.sum(noisy * read_mask) / denom,
            "mean_truth_length": jnp.sum(batch["truth_length"].astype(jnp.float32) * read_mask) / denom,
            "mean_logit_length": jnp.sum(logit_lengths.astype(jnp.float32) * read_mask) / denom,
        }

    rows = []
    for idx, raw_batch in enumerate(
        dataset.batches(bucket_chunks=args.bucket, batch_size=args.batch_size, drop_last=True),
        start=1,
    ):
        if idx > int(args.max_batches):
            break
        rng, use_rng = jax.random.split(rng)
        metrics = _batch_nll(_numeric_batch(raw_batch), use_rng)
        row = {key: float(jax.device_get(value)) for key, value in metrics.items()}
        row["batch"] = idx
        row["passes_original_lt_noisy"] = bool(row["original_nll"] < row["noisy_nll"])
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    payload = {"dataset": str(args.dataset), "bucket": int(args.bucket), "rows": rows}
    if args.output is not None:
        _write_json(args.output.expanduser().resolve(), payload)
    if not rows:
        raise SystemExit("No batches checked.")
    if not all(row["passes_original_lt_noisy"] for row in rows):
        raise SystemExit("NLL sanity check failed: original_nll is not lower than noisy_nll for every checked batch.")


if __name__ == "__main__":
    main()
