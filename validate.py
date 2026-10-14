#!/usr/bin/env python3
"""Validate a checkpoint on the held-out POD5 set.

Every read is reconstructed from its codewords on one GPU (windows of segment_samples with
validation.overlap_samples of linear cross-fade, the last window moved to the read end, each window
min-max normalized in pA as in training), basecalled with Dorado, and compared with the truth
sequences: length-weighted identity 1 - edit distance / max(len) (edlib, global) and the
empirical Q -10 log10(errors / length), overall and per flow cell. For the first
validation.band_reads reads it also reports, per frequency band, how much of the original
signal the reconstruction brings back: the reconstruction's band is split into a part
proportional to the original (`correct`, its amplitude relative to the original) and a part
uncorrelated with it (`wrong`, its energy relative to the original band energy).

train.py starts this on GPU validation.gpu after each checkpoint. By hand:
  python validate.py --config configs/offline-high.json --checkpoint <checkpoint dir> --out <dir>
Results: <out>/summary.json and one line appended to <out>/../summary.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # shares the GPU with training

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from codec.runtime import configure_runtime_env

configure_runtime_env()

import edlib
import jax
import numpy as np
import pod5
from flax.training import checkpoints as flax_ckpt

from codec.data.normalization import normalize_to_pm1_with_stats
from codec.models import build_audio_model

BANDS_HZ = ((0, 200), (200, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500))


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _path(value: str) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else _ROOT / path


def _windows(n: int, seg: int, hop: int) -> list[tuple[int, np.ndarray]]:
    """(start, weight) per window; the weights sum to 1 on every sample of the read."""
    if n < seg:
        return [(0, np.ones(seg, np.float32))]  # reflect-padded; only the first n samples are kept
    starts = list(range(0, n - seg + 1, hop))
    spans = [(s, 0, seg) for s in starts]
    if starts[-1] != n - seg:  # last window ends at the read end; its new part starts after the previous window's
        spans.append((n - seg, starts[-1] + hop - (n - seg), seg))
    fade = seg - hop
    out = []
    for k, (start, lo, hi) in enumerate(spans):
        w = np.zeros(seg, np.float32)
        w[lo:hi] = 1.0
        f = min(fade, hi - lo)
        if len(spans) > 1 and k > 0:
            w[lo:lo + f] = np.arange(f, dtype=np.float32) / f
        if len(spans) > 1 and k + 1 < len(spans):
            w[hi - f:hi] = np.minimum(w[hi - f:hi], 1.0 - np.arange(f, dtype=np.float32) / f)
        out.append((start, w))
    return out


def reconstruct(model_cfg: dict, checkpoint: Path, reads: list, seg: int, hop: int, batch: int) -> list[np.ndarray]:
    """ADC signal reconstructed from the codewords, one array per read."""
    model = build_audio_model(model_cfg)
    gen = flax_ckpt.restore_checkpoint(str(checkpoint.resolve()), target=None)["gen"]
    variables = jax.device_put({"params": gen["params"], "vq": gen["vq_vars"]}, jax.devices()[0])
    forward = jax.jit(lambda v, x: model.apply(v, x)["wave_hat"])

    chunks, specs, acc = [], [], []
    for i, read in enumerate(reads):
        cal = read.calibration
        pa = (read.signal.astype(np.float32) + np.float32(cal.offset)) * np.float32(cal.scale)
        acc.append(np.zeros(max(len(pa), seg), np.float32))
        for start, w in _windows(len(pa), seg, hop):
            norm, center, half = normalize_to_pm1_with_stats(pa[start:start + seg])
            if len(norm) < seg:
                norm = np.pad(norm, (0, seg - len(norm)), mode="reflect")
            chunks.append(norm)
            specs.append((i, start, w, center, half))
    chunks = np.stack(chunks)
    _log(f"{len(reads)} reads, {len(chunks)} windows")
    for b in range(0, len(chunks), batch):
        x = chunks[b:b + batch]
        n = len(x)
        if n < batch:
            x = np.concatenate([x, np.zeros((batch - n, seg), np.float32)])
        y = np.asarray(forward(variables, x), np.float32).reshape(batch, -1)[:n]
        for (i, start, w, center, half), yy in zip(specs[b:b + n], y):
            acc[i][start:start + seg] += (yy * half + center) * w
    out = []
    for read, pa in zip(reads, acc):
        cal = read.calibration
        adc = np.rint(pa[:len(read.signal)] / cal.scale - cal.offset)
        out.append(np.clip(adc, -32768, 32767).astype(np.int16))
    return out


def read_fastq(path: Path) -> dict[str, str]:
    seqs = {}
    with path.open() as fh:
        for i, line in enumerate(fh):
            if i % 4 == 0:
                read_id = line[1:].split()[0]
            elif i % 4 == 1:
                seqs[read_id] = line.strip()
    return seqs


def identity_metrics(pred: dict[str, str], truth: dict[str, str], groups: dict[str, str]) -> dict:
    per_group: dict[str, list[float]] = {}
    num = den = 0.0
    emp_q = []
    for read_id in sorted(set(pred) & set(truth)):
        t, p = truth[read_id], pred[read_id]
        denom = max(len(t), len(p), 1)
        dist = edlib.align(t, p, mode="NW", task="distance")["editDistance"] if t and p else denom
        identity = 1.0 - dist / denom
        num += identity * denom
        den += denom
        emp_q.append(60.0 if dist == 0 else min(60.0, -10.0 * math.log10(dist / denom)))
        per_group.setdefault(groups.get(read_id, "?"), []).append(identity)
    return {
        "reads": len(emp_q),
        "identity": num / den if den else 0.0,
        "empirical_q": float(np.mean(emp_q)) if emp_q else 0.0,
        "identity_by_flow_cell": {g: float(np.mean(v)) for g, v in sorted(per_group.items())},
    }


def band_analysis(reads: list, recon: list[np.ndarray], n_reads: int, seg: int, sample_rate: float) -> dict:
    freqs = np.fft.rfftfreq(seg, 1.0 / sample_rate)
    window = np.hanning(seg)
    p_orig = np.zeros(len(freqs))
    p_recon = np.zeros(len(freqs))
    p_err = np.zeros(len(freqs))
    cross = np.zeros(len(freqs))
    for read, rec in zip(reads[:n_reads], recon[:n_reads]):
        a_all, r_all = read.signal.astype(np.float64), rec.astype(np.float64)
        for s in range(0, len(a_all) - seg + 1, seg):
            a, r = a_all[s:s + seg], r_all[s:s + seg]
            fa = np.fft.rfft((a - a.mean()) * window)
            fr = np.fft.rfft((r - r.mean()) * window)
            p_orig += np.abs(fa) ** 2
            p_recon += np.abs(fr) ** 2
            p_err += np.abs(fa - fr) ** 2
            cross += np.real(fa * np.conj(fr))
    out = {}
    for low, high in BANDS_HZ:
        m = (freqs >= low) & (freqs < high) if high < sample_rate / 2 else (freqs >= low)
        po = p_orig[m].sum()
        correct = cross[m].sum() / po
        energy = p_recon[m].sum() / po
        out[f"{low}-{high}"] = {
            "orig_share": float(po / p_orig.sum()),
            "err_over_orig": float(p_err[m].sum() / po),
            "energy_over_orig": float(energy),
            "correct": float(correct),
            "wrong": float(energy - correct**2),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    v = cfg["validation"]
    seg = int(cfg["data"]["segment_samples"])
    hop = seg - int(v.get("overlap_samples", 144))
    sample_rate = float(cfg["data"]["sample_rate"])
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    with pod5.Reader(str(_path(v["source_pod5"]))) as reader:
        reads = [rec.to_read() for rec in reader.reads()]
    recon = reconstruct(cfg["model"], args.checkpoint, reads, seg, hop, int(v.get("batch_size", 64)))

    pod5_path, fastq = out / "recon.pod5", out / "recon.fastq"
    pod5_path.unlink(missing_ok=True)
    with pod5.Writer(str(pod5_path)) as writer:
        for read, signal in zip(reads, recon):
            original = read.signal
            read.signal = signal
            writer.add_read(read)
            read.signal = original
    _log("dorado")
    with fastq.open("w") as fh:
        subprocess.run(
            [str(_path(v["dorado_bin"])), "basecaller", str(_path(v["dorado_model"])), str(pod5_path),
             "--device", "cuda:0", "--emit-fastq"],
            stdout=fh, stderr=subprocess.DEVNULL, check=True,
        )
    pod5_path.unlink()

    groups = {}
    with _path(v["truth_meta"]).open() as fh:
        for line in fh:
            row = json.loads(line)
            groups[row["read_id"]] = row["fc"]
    summary = {
        "checkpoint": str(args.checkpoint),
        **identity_metrics(read_fastq(fastq), read_fastq(_path(v["truth_fastq"])), groups),
        "adc_mae": float(np.mean([np.mean(np.abs(r.signal.astype(np.int32) - s)) for r, s in zip(reads, recon)])),
        "bands": band_analysis(reads, recon, int(v.get("band_reads", 100)), seg, sample_rate),
        "minutes": (time.perf_counter() - t0) / 60.0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out.parent / "summary.jsonl").open("a") as fh:
        fh.write(json.dumps({"step": out.name, **summary}) + "\n")
    bands = "  ".join(f"{k}: correct {b['correct']:.2f} wrong {b['wrong']:.2f}" for k, b in summary["bands"].items())
    _log(f"identity {summary['identity']:.4f}  empirical Q {summary['empirical_q']:.2f}  "
         f"ADC MAE {summary['adc_mae']:.2f}  reads {summary['reads']}\n  {bands}")


if __name__ == "__main__":
    main()
