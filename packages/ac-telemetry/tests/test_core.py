from __future__ import annotations

import numpy as np

from ac_telemetry.util import close_short_false_gaps, contiguous_true_runs, parse_datetime_from_ac_filename


def test_true_runs() -> None:
    mask = np.array([False, True, True, False, True])
    assert contiguous_true_runs(mask) == [(1, 2), (4, 4)]


def test_close_gaps() -> None:
    mask = np.array([True, False, True, False, False, True])
    result = close_short_false_gaps(mask, 1)
    assert result.tolist() == [True, True, True, False, False, True]


def test_datetime_parser() -> None:
    from datetime import datetime
    from pathlib import Path

    parsed = parse_datetime_from_ac_filename(Path("AC_300726-155603_O_car_track.csv"))

    assert parsed == datetime(2026, 7, 30, 15, 56, 3).astimezone()
    assert parsed is not None and parsed.tzinfo is not None
