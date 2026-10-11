"""POD5 training chunks.

Each read is converted to picoamps with its POD5 calibration, cut into fixed windows (an extra
right-aligned last window covers the tail, "shift_last") and every window is normalized to
[-1, 1] by its own min/max, as the NanoRecon app does. Reads shorter than one window are
normalized on their real samples and reflect-padded, with a valid mask for the losses.

Files are visited in a seeded random order each epoch, several at a time in round robin, and
windows pass through a shuffle buffer, so a batch mixes many reads and files while the order
stays reproducible from the seed. One background thread prepares the batches.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pod5 as p5

from .normalization import normalize_to_pm1_with_stats


def _reflect_pad_right_1d(values: np.ndarray, target_length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    target = int(target_length)
    n = int(arr.shape[0])
    if n >= target:
        return arr[:target]
    if n == 1:
        return np.full((target,), float(arr[0]), dtype=np.float32)
    period = 2 * n - 2
    idx = np.arange(target, dtype=np.int64) % period
    idx = np.where(idx < n, idx, period - idx)
    return arr[idx].astype(np.float32, copy=False)


class Pod5ChunkDataset:
    def __init__(
        self,
        files: Sequence[Path],
        *,
        chunk_samples: int,
        hop_samples: int,
        sample_rate: float,
        pad_short_reads: bool,
        min_read_length_for_pad: int,
        shuffle_buffer: int,
        files_in_flight: int,
        seed: int,
    ) -> None:
        if not files:
            raise FileNotFoundError("no POD5 files to train on")
        self.files = [Path(f) for f in files]
        self.chunk_samples = int(chunk_samples)
        self.hop_samples = int(hop_samples)
        self.sample_rate = float(sample_rate)
        self.pad_short_reads = bool(pad_short_reads)
        self.min_read_length_for_pad = int(min_read_length_for_pad)
        self.shuffle_buffer = max(1, int(shuffle_buffer))
        self.files_in_flight = max(1, int(files_in_flight))
        self.seed = int(seed)

    def _reads(self, path: Path) -> Iterator[np.ndarray]:
        """Calibrated pA signal of each usable read in one file."""
        try:
            with p5.Reader(path) as reader:
                for read in reader.reads():
                    if abs(float(read.run_info.sample_rate) - self.sample_rate) > 1.0:
                        continue
                    offset, scale = float(read.calibration.offset), float(read.calibration.scale)
                    if not (np.isfinite(offset) and np.isfinite(scale)) or scale == 0.0:
                        continue
                    yield (read.signal.astype(np.float32) + np.float32(offset)) * np.float32(scale)
        except Exception as exc:  # damaged POD5 files raise from the reader
            print(f"[warn] skipping the rest of {path}: {exc}", flush=True)

    def _chunks(self, pa: np.ndarray) -> Iterator[tuple[np.ndarray, float, int]]:
        """(normalized window, pA half range, valid samples) for one read."""
        size, n = self.chunk_samples, int(pa.shape[0])
        if n < size:
            if self.pad_short_reads and n >= self.min_read_length_for_pad:
                norm, _, half = normalize_to_pm1_with_stats(pa)
                yield _reflect_pad_right_1d(norm, size), half, n
            return
        starts = list(range(0, n - size + 1, self.hop_samples))
        if starts[-1] != n - size:
            starts.append(n - size)
        for start in starts:
            norm, _, half = normalize_to_pm1_with_stats(pa[start : start + size])
            yield norm, half, size

    def _windows(self, rng: np.random.Generator) -> Iterator[tuple[np.ndarray, float, int]]:
        pending = iter([self.files[i] for i in rng.permutation(len(self.files))])
        active = [self._reads(p) for _, p in zip(range(self.files_in_flight), pending)]
        k = 0
        while active:
            k %= len(active)
            pa = next(active[k], None)
            if pa is None:
                path = next(pending, None)
                if path is None:
                    active.pop(k)
                else:
                    active[k] = self._reads(path)
                continue
            yield from self._chunks(pa)
            k += 1

    def _collate(self, items: list[tuple[np.ndarray, float, int]]) -> dict[str, np.ndarray]:
        valid = np.asarray([v for _, _, v in items], dtype=np.int32)
        return {
            "signal": np.stack([s for s, _, _ in items]).astype(np.float32, copy=False),
            "pa_half_range": np.asarray([h for _, h, _ in items], dtype=np.float32),
            "valid_mask": (np.arange(self.chunk_samples)[None, :] < valid[:, None]).astype(np.float32),
        }

    def _batches(self, batch_size: int, epoch: int) -> Iterator[dict[str, np.ndarray]]:
        rng = np.random.default_rng([self.seed, int(epoch)])
        buf: list[tuple[np.ndarray, float, int]] = []
        out: list[tuple[np.ndarray, float, int]] = []
        for item in self._windows(rng):
            if len(buf) < self.shuffle_buffer:
                buf.append(item)
                continue
            j = int(rng.integers(len(buf)))
            out.append(buf[j])
            buf[j] = item
            if len(out) == batch_size:
                yield self._collate(out)
                out = []
        out.extend(buf[i] for i in rng.permutation(len(buf)))
        for start in range(0, len(out) - batch_size + 1, batch_size):  # the last partial batch is dropped
            yield self._collate(out[start : start + batch_size])

    def batches(self, batch_size: int, epoch: int, prefetch: int = 8) -> Iterator[dict[str, np.ndarray]]:
        """Batches of one epoch, prepared by a background thread up to `prefetch` ahead."""
        return _background(self._batches(int(batch_size), epoch), prefetch)


def _background(source: Iterator[Any], size: int) -> Iterator[Any]:
    q: queue.Queue = queue.Queue(maxsize=max(1, int(size)))
    done = object()
    stop = threading.Event()

    def put(item: Any) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=1.0)
                return True
            except queue.Full:
                pass
        return False

    def run() -> None:
        try:
            for item in source:
                if not put(item):
                    return
            put(done)
        except BaseException as exc:
            put(exc)

    threading.Thread(target=run, daemon=True, name="pod5-batches").start()
    try:
        while True:
            item = q.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
