from pathlib import Path

import pandas as pd
import pytest
from ac_telemetry.config import ProcessingConfig
from ac_telemetry.track import TrackModel
from track_fixture import make_track


def test_track_model_load_align_and_tables_are_the_test_surface(tmp_path: Path) -> None:
    track = TrackModel.load(make_track(tmp_path / "track"))
    samples = pd.DataFrame(
        {
            "lap_id": ["lap"] * 4,
            "position.x": [2.0, 5.0, 8.0, 12.0],
            "position.y": [0.0] * 4,
            "position.z": [1.0] * 4,
            "velocity.x": [20.0] * 4,
            "velocity.y": [0.0] * 4,
            "velocity.z": [0.0] * 4,
            "accel_world_x_ms2": [0.0] * 4,
            "accel_world_y_ms2": [0.0] * 4,
            "accel_world_z_ms2": [0.0] * 4,
        }
    )

    aligned = track.align(samples, ProcessingConfig())

    assert aligned["track_s_m"].tolist() == pytest.approx([2.0, 5.0, 8.0, 12.0])
    assert aligned["track_projection_distance_3d_m"].tolist() == pytest.approx(
        [1.0] * 4
    )
    assert aligned["lateral_offset_m"].tolist() == pytest.approx([-1.0] * 4)
    assert aligned["velocity_along_track_ms"].tolist() == pytest.approx([20.0] * 4)
    assert aligned["velocity_cross_track_ms"].tolist() == pytest.approx([0.0] * 4)
    assert aligned["track_section_name"].iloc[0] == "Straight"
    assert "track/reference" in track.tables()
    assert len(track.tables()["track/reference"]) == 8
    assert track.reference_id


def test_track_s_is_independent_of_actual_driven_path_length(tmp_path: Path) -> None:
    track = TrackModel.load(make_track(tmp_path / "track"))
    # Both observations are at the same along-track position. One takes a much wider line.
    samples = pd.DataFrame(
        {
            "lap_id": ["a", "b"],
            "position.x": [5.0, 5.0],
            "position.y": [0.0, 0.0],
            "position.z": [0.5, 2.5],
            "velocity.x": [10.0, 10.0],
            "velocity.y": [0.0, 0.0],
            "velocity.z": [0.0, 0.0],
            "accel_world_x_ms2": [0.0, 0.0],
            "accel_world_y_ms2": [0.0, 0.0],
            "accel_world_z_ms2": [0.0, 0.0],
        }
    )

    aligned = track.align(samples, ProcessingConfig())

    assert aligned["track_s_m"].tolist() == pytest.approx([5.0, 5.0])
    assert aligned["lateral_offset_m"].tolist() == pytest.approx([-0.5, -2.5])
