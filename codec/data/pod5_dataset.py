from __future__ import annotations

import queue
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np
import pod5 as p5

from .normalization import normalize_to_pm1_with_stats
from .pod5_processing import (
    parse_calibration,
    resolve_sample_rate,
    NormalizationStats,
    CalibrationParams,
    CalibrationError,
)


def _calibrate_read_signal(signal: np.ndarray, calibration: Any) -> tuple[np.ndarray, CalibrationParams]:
    """Convert int16 ADC to picoamps once per read using POD5 calibration."""
    cal = parse_calibration(calibration)
    return cal.to_picoamps(signal), cal


def _normalize_chunk_signal(chunk: np.ndarray) -> tuple[np.ndarray, NormalizationStats]:
    """Normalize a single chunk to [-1, 1] and keep reversible stats."""
    normalized, center, half_range = normalize_to_pm1_with_stats(chunk)
    return normalized, NormalizationStats(center=center, half_range=half_range)


def _reflect_pad_right_1d(values: np.ndarray, target_length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    target = int(target_length)
    n = int(arr.shape[0])
    if n >= target:
        return arr[:target]
    if n <= 0:
        return np.zeros((target,), dtype=np.float32)
    if n == 1:
        return np.full((target,), float(arr[0]), dtype=np.float32)
    period = 2 * n - 2
    idx = np.arange(target, dtype=np.int64) % period
    idx = np.where(idx < n, idx, period - idx)
    return arr[idx].astype(np.float32, copy=False)


def _should_skip_pod5(exc: Exception) -> bool:
    msg = str(exc)
    keywords = (
        "Invalid signature in file",
        "Failed to open pod5 file",
        "Pod5ApiException",
        "Bad address",
        "Error writing bytes to file",
    )
    return any(k in msg for k in keywords)


"""POD5 dataset utilities.

Each read is streamed from POD5, converted to picoamps via calibration, sliced
into fixed windows, and then each chunk is normalized independently to [-1, 1]
using its own center/half-range statistics. Sample-rate hints stored in POD5 are preferred,
falling back to configured defaults only when metadata is absent.
"""


@dataclass
class NanoporeSignalDataset:
    pod5_files: List[Path]
    window_ms: int = 1000
    window_samples: Optional[int] = None
    window_hop_samples: Optional[int] = None
    tail_chunk_mode: str = "shift_last"
    pad_short_reads: bool = False
    min_read_length_for_pad: int = 2048
    tail_min_valid_samples: int = 2048
    tail_min_valid_ratio: float = 0.33
    max_padded_tail_fraction: float = 1.0
    sample_rate_hz_default: Optional[float] = None
    return_metadata: bool = False
    read_ids_per_file: Optional[Dict[Path, Sequence[str]]] = None
    loader_workers: int = 1
    loader_prefetch_chunks: int = 128
    _invalid_files: set[Path] = field(default_factory=set, init=False, repr=False)
    _cached_length: Optional[int] = field(default=None, init=False, repr=False)
    _calibration_warned_files: set[Path] = field(default_factory=set, init=False, repr=False)
    _tail_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _emitted_chunks: int = field(default=0, init=False, repr=False)
    _emitted_padded_tail_chunks: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_paths(
        cls,
        files: Iterable[Union[str, Path]],
        window_ms: int = 1000,
        window_samples: Optional[int] = None,
        window_hop_samples: Optional[int] = None,
        tail_chunk_mode: str = "shift_last",
        pad_short_reads: bool = False,
        min_read_length_for_pad: int = 2048,
        tail_min_valid_samples: int = 2048,
        tail_min_valid_ratio: float = 0.33,
        max_padded_tail_fraction: float = 1.0,
        sample_rate_hz_default: Optional[float] = None,
        return_metadata: bool = False,
        read_ids_per_file: Optional[Dict[Union[str, Path], Sequence[str]]] = None,
        loader_workers: int = 1,
        loader_prefetch_chunks: int = 128,
    ) -> "NanoporeSignalDataset":
        paths = [Path(f).expanduser().resolve() for f in files]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"POD5 not found: {p}")
        rid_map: Optional[Dict[Path, Sequence[str]]] = None
        if read_ids_per_file:
            rid_map = {Path(k).expanduser().resolve(): v for k, v in read_ids_per_file.items()}
        workers = max(1, int(loader_workers))
        prefetch_chunks = max(1, int(loader_prefetch_chunks))
        window_samples_int: Optional[int] = None
        if window_samples is not None:
            ws = int(window_samples)
            if ws > 0:
                window_samples_int = ws
        window_hop_samples_int: Optional[int] = None
        if window_hop_samples is not None:
            hs = int(window_hop_samples)
            if hs > 0:
                window_hop_samples_int = hs
        tail_chunk_mode_text = str(tail_chunk_mode).strip().lower() or "shift_last"
        if tail_chunk_mode_text not in {"drop", "shift_last", "pad_last"}:
            raise ValueError(
                f"Unsupported tail_chunk_mode={tail_chunk_mode!r}; choose from ['drop', 'shift_last', 'pad_last']"
            )
        return cls(
            pod5_files=paths,
            window_ms=int(window_ms),
            window_samples=window_samples_int,
            window_hop_samples=window_hop_samples_int,
            tail_chunk_mode=tail_chunk_mode_text,
            pad_short_reads=bool(pad_short_reads),
            min_read_length_for_pad=max(1, int(min_read_length_for_pad)),
            tail_min_valid_samples=max(1, int(tail_min_valid_samples)),
            tail_min_valid_ratio=max(0.0, min(1.0, float(tail_min_valid_ratio))),
            max_padded_tail_fraction=max(0.0, min(1.0, float(max_padded_tail_fraction))),
            sample_rate_hz_default=sample_rate_hz_default,
            return_metadata=bool(return_metadata),
            read_ids_per_file=rid_map,
            loader_workers=workers,
            loader_prefetch_chunks=prefetch_chunks,
        )

    def _record_emitted_chunk(self, *, is_padded_tail: bool = False) -> None:
        with self._tail_lock:
            self._emitted_chunks += 1
            if is_padded_tail:
                self._emitted_padded_tail_chunks += 1

    def _allow_padded_tail(self) -> bool:
        fraction = float(self.max_padded_tail_fraction)
        if fraction >= 1.0:
            return True
        if fraction <= 0.0:
            return False
        with self._tail_lock:
            projected_padded = self._emitted_padded_tail_chunks + 1
            projected_total = max(1, self._emitted_chunks + 1)
            return (float(projected_padded) / float(projected_total)) <= fraction

    def _emit_normalized_chunk(
        self,
        chunk: np.ndarray,
        *,
        cal: CalibrationParams,
        valid_length: int | None = None,
        is_padded_tail: bool = False,
    ) -> Any:
        real = np.asarray(chunk, dtype=np.float32).reshape(-1)
        chunk_size = int(self.window_samples) if self.window_samples is not None else int(real.shape[0])
        if valid_length is None:
            valid_length = int(real.shape[0])
        valid_length = max(0, min(int(valid_length), int(real.shape[0])))
        real_part = real[:valid_length]
        arr, stats = _normalize_chunk_signal(real_part)
        arr = np.asarray(arr, dtype=np.float32)
        if valid_length < chunk_size:
            arr = _reflect_pad_right_1d(arr, chunk_size)
        valid_mask = np.zeros((chunk_size,), dtype=np.float32)
        valid_mask[:valid_length] = 1.0
        self._record_emitted_chunk(is_padded_tail=is_padded_tail)
        if not self.return_metadata:
            return arr
        return (arr, stats, cal, valid_mask, int(valid_length), bool(is_padded_tail))

    def _iter_chunks_from_file(self, file_path: Path) -> Iterator[np.ndarray]:
        if file_path in self._invalid_files:
            return
        selection = None
        if self.read_ids_per_file and file_path in self.read_ids_per_file:
            selection = self.read_ids_per_file[file_path]

        def _mark_bad_file(exc: Exception, context: str) -> None:
            if file_path in self._invalid_files:
                return
            print(f"[warn] {context}，永久跳过 {file_path}: {exc}")
            self._invalid_files.add(file_path)

        warned_sr_mismatch = False
        try:
            with p5.Reader(str(file_path)) as reader:
                run_info = getattr(reader, "run_info", None)
                gen = reader.reads(selection=selection) if selection else reader.reads()
                for read in gen:
                    try:
                        read_sr = getattr(read, "sample_rate", None)
                        run_sr = getattr(run_info, "sample_rate", None)
                        measured_sr = None
                        if read_sr is not None:
                            try:
                                measured_sr = float(read_sr)
                            except Exception:
                                measured_sr = None
                        if measured_sr is None and run_sr is not None:
                            try:
                                measured_sr = float(run_sr)
                            except Exception:
                                measured_sr = None

                        target_sr = resolve_sample_rate(
                            read_obj=read,
                            run_info=run_info,
                            configured_hz=self.sample_rate_hz_default,
                        )
                        # Only gate when we actually have a measured value from metadata.
                        if measured_sr is not None and self.sample_rate_hz_default is not None:
                            mismatch = abs(measured_sr - float(self.sample_rate_hz_default))
                            tol = max(1.0, 0.001 * float(self.sample_rate_hz_default))
                            if mismatch > tol:
                                if not warned_sr_mismatch:
                                    print(
                                        f"[warn] read sample_rate={measured_sr:.2f}Hz differs from configured {float(self.sample_rate_hz_default):.2f}Hz; skipping reads in {file_path}",
                                        flush=True,
                                    )
                                    warned_sr_mismatch = True
                                continue
                        if self.window_samples is not None and self.window_samples > 0:
                            chunk_size = int(self.window_samples)
                        else:
                            chunk_size = int(round(self.window_ms * float(target_sr) / 1000.0))
                        if chunk_size <= 0:
                            chunk_size = 1
                        if self.window_hop_samples is not None and self.window_hop_samples > 0:
                            chunk_hop = int(self.window_hop_samples)
                        else:
                            chunk_hop = chunk_size
                        raw_signal = read.signal
                        try:
                            pa_signal, cal = _calibrate_read_signal(
                                raw_signal, getattr(read, "calibration", None)
                            )
                        except CalibrationError as cal_exc:
                            if file_path not in self._calibration_warned_files:
                                read_id = getattr(read, "read_id", "unknown")
                                warnings.warn(
                                    f"[pod5] Skipping reads in {file_path.name} (first failing read {read_id}): {cal_exc}",
                                    RuntimeWarning,
                                )
                                self._calibration_warned_files.add(file_path)
                            continue
                        n = int(pa_signal.shape[0])
                        if n < chunk_size:
                            if self.pad_short_reads and n >= int(self.min_read_length_for_pad):
                                yield self._emit_normalized_chunk(
                                    pa_signal,
                                    cal=cal,
                                    valid_length=n,
                                    is_padded_tail=True,
                                )
                            continue
                        last_full_start = n - chunk_size
                        starts = list(range(0, last_full_start + 1, chunk_hop))
                        if not starts:
                            starts = [0]
                        for start in starts:
                            stop = start + chunk_size
                            yield self._emit_normalized_chunk(pa_signal[start:stop], cal=cal)
                        if self.tail_chunk_mode == "shift_last" and starts[-1] != last_full_start:
                            start = last_full_start
                            yield self._emit_normalized_chunk(pa_signal[start : start + chunk_size], cal=cal)
                        elif self.tail_chunk_mode == "pad_last":
                            tail_start = starts[-1] + chunk_hop
                            if tail_start < n:
                                tail_valid = n - tail_start
                                tail_ratio = float(tail_valid) / float(chunk_size)
                                if (
                                    tail_valid >= int(self.tail_min_valid_samples)
                                    or tail_ratio >= float(self.tail_min_valid_ratio)
                                ) and self._allow_padded_tail():
                                    yield self._emit_normalized_chunk(
                                        pa_signal[tail_start:n],
                                        cal=cal,
                                        valid_length=tail_valid,
                                        is_padded_tail=True,
                                    )
                    except Exception as read_exc:
                        if _should_skip_pod5(read_exc):
                            _mark_bad_file(read_exc, "读取 read 失败")
                            return
                        read_id = getattr(read, "read_id", "unknown")
                        raise RuntimeError(f"读取 {file_path} 中 read {read_id} 失败: {read_exc}") from read_exc
        except Exception as open_exc:
            if _should_skip_pod5(open_exc):
                _mark_bad_file(open_exc, "POD5 文件损坏")
                return
            raise RuntimeError(f"打开 POD5 {file_path} 失败: {open_exc}") from open_exc

    @staticmethod
    def _flush_batch(buf: List[Any]) -> Any:
        if buf and isinstance(buf[0], tuple):
            signals = []
            pa_centers = []
            pa_half_ranges = []
            cal_offsets = []
            cal_scales = []
            valid_masks = []
            valid_lengths = []
            is_padded_tails = []
            for item in buf:
                if len(item) == 3:
                    chunk, stats, cal = item
                    valid_mask = np.ones_like(np.asarray(chunk, dtype=np.float32), dtype=np.float32)
                    valid_length = int(valid_mask.shape[0])
                    is_padded_tail = False
                else:
                    chunk, stats, cal, valid_mask, valid_length, is_padded_tail = item
                signals.append(np.asarray(chunk, dtype=np.float32))
                pa_centers.append(float(stats.center))
                pa_half_ranges.append(float(stats.half_range))
                cal_offsets.append(float(cal.offset))
                cal_scales.append(float(cal.scale))
                valid_masks.append(np.asarray(valid_mask, dtype=np.float32))
                valid_lengths.append(int(valid_length))
                is_padded_tails.append(float(bool(is_padded_tail)))
            return {
                "signal": np.asarray(np.stack(signals, axis=0), dtype=np.float32),
                "pa_center": np.asarray(pa_centers, dtype=np.float32),
                "pa_half_range": np.asarray(pa_half_ranges, dtype=np.float32),
                "calibration_offset": np.asarray(cal_offsets, dtype=np.float32),
                "calibration_scale": np.asarray(cal_scales, dtype=np.float32),
                "valid_mask": np.asarray(np.stack(valid_masks, axis=0), dtype=np.float32),
                "valid_length": np.asarray(valid_lengths, dtype=np.int32),
                "is_padded_tail": np.asarray(is_padded_tails, dtype=np.float32),
            }
        batch = np.asarray(np.stack(buf, axis=0), dtype=np.float32)
        return batch[:, np.newaxis, :]

    def iter_chunks(self, files_cycle: bool = False) -> Iterator[np.ndarray]:
        while True:
            for fp in self.pod5_files:
                if fp in self._invalid_files:
                    continue
                yield from self._iter_chunks_from_file(fp)
            if not files_cycle:
                break

    def batches(
        self,
        batch_size: int,
        drop_last: bool = True,
        files_cycle: bool = False,
        num_workers: Optional[int] = None,
        max_chunk_queue: Optional[int] = None,
    ) -> Iterator[Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        worker_count = int(num_workers if num_workers is not None else self.loader_workers)
        queue_cap = int(max_chunk_queue if max_chunk_queue is not None else self.loader_prefetch_chunks)
        # Threaded path engages whenever >1 worker is requested; finite epochs still benefit
        if worker_count <= 1:
            while True:
                buf: List[Any] = []
                for fp in self.pod5_files:
                    if fp in self._invalid_files:
                        continue
                    for chunk in self._iter_chunks_from_file(fp):
                        buf.append(chunk)
                        if len(buf) == batch_size:
                            yield self._flush_batch(buf)
                            buf.clear()
                if not drop_last and len(buf) > 0:
                    yield self._flush_batch(buf)
                    buf.clear()
                if not files_cycle:
                    break
            return
        yield from self._threaded_batches(
            batch_size=batch_size,
            drop_last=drop_last,
            files_cycle=files_cycle,
            worker_count=worker_count,
            max_chunk_queue=max(1, queue_cap),
        )

    class _FileIterator:
        def __init__(self, files: List[Path], files_cycle: bool):
            self._files = files
            self._files_cycle = files_cycle
            self._idx = 0
            self._lock = threading.Lock()

        def next(self) -> Optional[Path]:
            with self._lock:
                if not self._files:
                    return None
                if self._idx >= len(self._files):
                    if not self._files_cycle:
                        return None
                    self._idx = 0
                fp = self._files[self._idx]
                self._idx += 1
                return fp

    def _threaded_batches(
        self,
        *,
        batch_size: int,
        drop_last: bool,
        files_cycle: bool,
        worker_count: int,
        max_chunk_queue: int,
    ) -> Iterator[Any]:
        valid_files = [fp for fp in self.pod5_files if fp not in self._invalid_files]
        if not valid_files:
            raise FileNotFoundError("No valid POD5 files to stream from.")
        sentinel = object()
        chunk_queue: "queue.Queue[object]" = queue.Queue(maxsize=max_chunk_queue)
        stop_event = threading.Event()
        file_iter = self._FileIterator(valid_files, files_cycle)
        workers: list[threading.Thread] = []

        def worker_main() -> None:
            try:
                while not stop_event.is_set():
                    fp = file_iter.next()
                    if fp is None:
                        break
                    for chunk in self._iter_chunks_from_file(fp):
                        if stop_event.is_set():
                            break
                        chunk_queue.put(chunk, block=True)
            except Exception as exc:
                chunk_queue.put(exc)
            finally:
                chunk_queue.put(sentinel)

        for _ in range(worker_count):
            t = threading.Thread(target=worker_main, daemon=True)
            t.start()
            workers.append(t)

        def generator() -> Iterator[Any]:
            active = worker_count
            buf: List[Any] = []
            try:
                while True:
                    try:
                        item = chunk_queue.get(timeout=0.5)
                    except queue.Empty:
                        if stop_event.is_set() and (not files_cycle or active == 0):
                            break
                        continue
                    if item is sentinel:
                        active -= 1
                        # When streaming forever we only break once stop_event is set (consumer closed) or workers drained
                        if active == 0 and (not files_cycle or stop_event.is_set()):
                            break
                        continue
                    if isinstance(item, Exception):
                        raise item
                    buf.append(item)  # type: ignore[arg-type]
                    if len(buf) == batch_size:
                        yield self._flush_batch(buf)
                        buf.clear()
            finally:
                stop_event.set()
                # Drain remaining sentinels so worker threads can exit
                while active > 0:
                    try:
                        item = chunk_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if item is sentinel:
                        active -= 1
                for t in workers:
                    t.join(timeout=0.5)
            if not drop_last and buf:
                yield self._flush_batch(buf)

        yield from generator()

    def __len__(self) -> int:
        if self._cached_length is not None:
            return self._cached_length
        if self.read_ids_per_file:
            total = sum(len(ids) for ids in self.read_ids_per_file.values())
        else:
            valid = [fp for fp in self.pod5_files if fp not in self._invalid_files]
            total = len(valid)
        self._cached_length = max(0, total)
        return self._cached_length
