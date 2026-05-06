from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = REPO_ROOT / "tmp"

TRUTH_MODE_ANALYSIS_HAC_PROXY = "analysis_hac_proxy"


@dataclass(frozen=True)
class FastqEntry:
    seq: str
    qual: str


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_fastq_gz(path: Path, records: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(f"@{record['read_id']}\n")
            handle.write(f"{record['seq']}\n+\n")
            handle.write(f"{record['qual']}\n")


def progress_markers(total: int) -> list[int]:
    total = max(0, int(total))
    if total <= 0:
        return []
    markers = {1, total}
    for pct in (10, 25, 50, 75, 90):
        markers.add(max(1, int(round(total * pct / 100.0))))
    return sorted(markers)


def summarize(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def sanitize_tag(text: str) -> str:
    cleaned = [ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(text)]
    return "".join(cleaned).strip("_") or "data"


def default_tmp_path(*parts: str | int) -> Path:
    return TMP_ROOT.joinpath(*(str(part) for part in parts))


def resolve_repo_path(path_value: str | Path | None, cfg_dir: Path) -> Path | None:
    if path_value in (None, ""):
        return None
    raw_text = str(path_value).strip()
    if not raw_text:
        return None
    candidate = Path(raw_text).expanduser()
    has_explicit_path = candidate.is_absolute() or raw_text.startswith(".") or raw_text.startswith("~")
    has_explicit_path = has_explicit_path or any(sep in raw_text for sep in (os.sep, "/", "\\"))
    if not has_explicit_path and not candidate.exists():
        return candidate
    if not candidate.is_absolute():
        candidate = (cfg_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _is_executable_file(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.is_file() and os.access(path, os.X_OK)
    except OSError:
        return False


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def resolve_dorado_bin(
    dorado_bin_value: str | Path | None,
    *,
    cfg_dir: Path,
    dorado_model: Path | None,
) -> Path:
    raw_value = "" if dorado_bin_value is None else str(dorado_bin_value).strip()
    if raw_value:
        explicit = resolve_repo_path(raw_value, cfg_dir)
        if explicit is not None and _is_executable_file(explicit):
            return explicit
        if explicit is None:
            which_match = shutil.which(raw_value)
            if which_match:
                return Path(which_match).resolve()

    candidate_paths: list[Path] = []
    if dorado_model is not None:
        model_dir = dorado_model.resolve()
        search_roots = [model_dir]
        search_roots.extend(model_dir.parents)
        for root in search_roots:
            candidate_paths.append(root / "dorado")
            candidate_paths.append(root / "bin" / "dorado")
            candidate_paths.extend(sorted(root.glob("dorado-*/bin/dorado")))
            if root.name == "models":
                parent = root.parent
                candidate_paths.append(parent / "bin" / "dorado")
                candidate_paths.extend(sorted(parent.glob("dorado-*/bin/dorado")))

    default_roots = [
        Path("~/Download/dorado").expanduser(),
        Path("~/dorado").expanduser(),
    ]
    for root in default_roots:
        candidate_paths.append(root / "dorado")
        candidate_paths.append(root / "bin" / "dorado")
        candidate_paths.extend(sorted(root.glob("dorado-*/bin/dorado")))

    for candidate in _dedupe_paths(path.resolve() for path in candidate_paths if _is_executable_file(path)):
        return candidate

    fallback = shutil.which("dorado")
    if fallback:
        return Path(fallback).resolve()

    searched = []
    if raw_value:
        searched.append(raw_value)
    if dorado_model is not None:
        searched.append(str(dorado_model))
    raise FileNotFoundError(
        "Unable to locate the Dorado executable. "
        "Set --dorado-bin or install Dorado into a standard location. "
        f"Searched from: {searched}"
    )


def run_dorado(*, dorado_bin: str, dorado_model: str, pod5_path: Path, out_fastq: Path, device: str) -> None:
    out_fastq.parent.mkdir(parents=True, exist_ok=True)
    if out_fastq.exists():
        out_fastq.unlink()
    cmd = [dorado_bin, "basecaller", dorado_model, str(pod5_path), "--device", device, "--emit-fastq"]
    print(f"[dorado] {' '.join(cmd)}", flush=True)
    with out_fastq.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd,
            stdout=handle,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Dorado failed on {pod5_path}: {proc.stderr}")
    print(f"[dorado] finished {pod5_path.name}", flush=True)


def open_fastq(path: Path):
    text_kwargs = {"encoding": "utf-8", "errors": "ignore"}
    if path.suffix == ".gz":
        return gzip.open(path, "rt", **text_kwargs)
    try:
        with path.open("rb") as handle:
            magic = handle.read(2)
        if magic == b"\x1f\x8b":
            return gzip.open(path, "rt", **text_kwargs)
    except FileNotFoundError:
        raise
    return path.open("r", **text_kwargs)


def read_fastq(path: Path) -> dict[str, FastqEntry]:
    records: dict[str, FastqEntry] = {}
    with open_fastq(path) as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            if not header.startswith("@"):
                continue
            read_id = header[1:].strip().split()[0]
            seq_parts: list[str] = []
            while True:
                line = handle.readline()
                if not line:
                    break
                if line.startswith("+"):
                    break
                seq_parts.append(line.strip())
            sequence = "".join(seq_parts)
            qual_parts: list[str] = []
            qual_len = 0
            while qual_len < len(sequence):
                line = handle.readline()
                if not line:
                    break
                qline = line.strip()
                qual_parts.append(qline)
                qual_len += len(qline)
            records[read_id] = FastqEntry(seq=sequence, qual="".join(qual_parts)[: len(sequence)])
    return records


def mean_qscore(quality: str) -> float:
    if not quality:
        return 0.0
    values = np.fromiter((ord(ch) - 33 for ch in quality), dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.mean(values))
