import struct
from pathlib import Path

import numpy as np


def write_ai(
    path: Path, points: list[tuple[float, float, float]], side: float = 8.0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(points, dtype=float)
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(step)]
    data = bytearray(struct.pack("<iiii", 7, len(points), 0, 0))
    for index, ((x, y, z), distance) in enumerate(zip(points, cum, strict=True)):
        data.extend(struct.pack("<ffffi", x, y, z, float(distance), index))
    data.extend(struct.pack("<i", len(points)))
    for _ in points:
        payload = [0.0] * 18
        payload[4] = 1000.0
        payload[5] = side
        payload[6] = side
        data.extend(struct.pack("<18f", *payload))
    path.write_bytes(bytes(data))


def make_track(root: Path) -> Path:
    # Dense-enough square so endpoint proximity identifies it as a closed spline.
    points = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (20.0, 0.0, 0.0),
        (20.0, 0.0, 10.0),
        (20.0, 0.0, 20.0),
        (10.0, 0.0, 20.0),
        (0.0, 0.0, 20.0),
        (0.0, 0.0, 10.0),
    ]
    write_ai(root / "ai" / "fast_lane.ai", points)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / "sections.ini").write_text(
        "[SECTION_0]\nIN=0.0\nOUT=0.25\nTEXT=Straight\n", encoding="utf-8"
    )
    return root
