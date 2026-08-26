import numpy as np
import pandas as pd
import pytest
from ac_telemetry.config import ProcessingConfig
from ac_telemetry.segments import segment_passes


def test_segment_boundaries_are_interpolated_on_fixed_track_coordinate() -> None:
    samples = pd.DataFrame(
        {
            "session_id": ["s"] * 5,
            "lap_id": ["l"] * 5,
            "source_lap_number": [1] * 5,
            "sample_index": np.arange(5),
            "lap_time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "dt_s": [1.0] * 5,
            "track_s_m": [0.0, 10.0, 20.0, 30.0, 40.0],
            "track_s_unwrapped_m": [0.0, 10.0, 20.0, 30.0, 40.0],
            "speed_kmh": [100.0, 90.0, 80.0, 90.0, 100.0],
            "is_braking": [False, True, True, False, False],
            "brake_n": [0.0, 0.5, 0.4, 0.0, 0.0],
            "throttle": [1.0, 0.0, 0.0, 1.0, 1.0],
            "is_full_throttle": [True, False, False, True, True],
            "is_partial_throttle": [False] * 5,
            "is_coasting": [False, False, True, False, False],
            "rear_slip_ratio_max": [0.0] * 5,
            "steerAngle": [0.0, 0.1, 0.2, -0.1, -0.1],
            "path_distance_3d_m": [0.0, 10.0, 20.0, 30.0, 40.0],
            "lateral_offset_m": [0.0, 1.0, 2.0, 1.0, 0.0],
            "velocity_cross_track_ms": [0.0] * 5,
            "velocity_heading_error_rad": [0.0] * 5,
            "gear_physical": [3] * 5,
            "rpm": [6000.0, 5500.0, 5000.0, 6000.0, 7000.0],
            "fuel": [20.0] * 5,
            "track_long_g": [0.0, -1.0, -0.5, 0.2, 0.3],
            "track_lat_g": [0.0, 0.5, 1.0, 0.5, 0.0],
            "position.x": [0.0, 10.0, 20.0, 30.0, 40.0],
            "position.y": [0.0] * 5,
            "position.z": [0.0] * 5,
        }
    )
    laps = pd.DataFrame([{"lap_id": "l", "is_complete": True, "is_valid": True}])
    definitions = {
        "coordinate": "track_s_m",
        "segments": [{"id": "corner", "start": 5.0, "end": 35.0}],
    }

    result = segment_passes(
        samples, laps, definitions, ProcessingConfig(), track_length_m=100.0
    ).iloc[0]

    assert result.segment_time_s == pytest.approx(3.0)
    assert result.entry_speed_kmh == pytest.approx(95.0)
    assert result.exit_speed_kmh == pytest.approx(95.0)
    assert result.minimum_speed_track_s_m == pytest.approx(20.0)
    assert result.throttle_pickup_track_s_m == pytest.approx(30.0)
    assert result.full_throttle_commit_track_s_m == pytest.approx(30.0)
    assert result.actual_path_length_m == pytest.approx(30.0)
    assert result.path_excess_m == pytest.approx(0.0)
