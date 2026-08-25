import hashlib
import json
import math
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: object, length: int = 16) -> str:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value, ensure_ascii=False, indent=2, allow_nan=False, default=json_default
        ),
        encoding="utf-8",
    )


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def parse_datetime_from_ac_filename(path: Path) -> datetime | None:
    match = re.search(r"(?:^|_)AC_(\d{6}-\d{6})_", path.name, re.IGNORECASE)
    if not match:
        return None
    token = match.group(1)
    try:
        # A naive datetime is interpreted as local time by astimezone().
        return datetime.strptime(token, "%d%m%y-%H%M%S").astimezone()
    except ValueError:
        return None


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return []
    padded = np.r_[False, arr, False].astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


def close_short_false_gaps(mask: np.ndarray, max_gap_samples: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    if max_gap_samples <= 0 or out.size == 0:
        return out
    false_runs = contiguous_true_runs(~out)
    for start, end in false_runs:
        length = end - start + 1
        if start > 0 and end < len(out) - 1 and length <= max_gap_samples:
            out[start : end + 1] = True
    return out


def safe_float(value: object, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except TypeError, ValueError:
        return default
    return result if math.isfinite(result) else default


def nan_or(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def percentile(values: Iterable[float], q: float) -> float | None:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.percentile(arr, q))
