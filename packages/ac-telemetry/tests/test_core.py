from __future__ import annotations

import numpy as np

from ac_telemetry.util import close_short_false_gaps, contiguous_true_runs, parse_date_from_ac_filename


def test_true_runs() -> None:
    mask = np.array([False, True, True, False, True])
    assert contiguous_true_runs(mask) == [(1, 2), (4, 4)]


def test_close_gaps() -> None:
    mask = np.array([True, False, True, False, False, True])
    result = close_short_false_gaps(mask, 1)
    assert result.tolist() == [True, True, True, False, False, True]


def test_date_parser() -> None:
    from pathlib import Path

    assert parse_date_from_ac_filename(Path("AC_300726-155603_O_car_track.csv")) == "2026-07-30"
