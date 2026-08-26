from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ac_telemetry.config import ProcessingConfig
from ac_telemetry.track import TrackModel
from track_fixture import make_track, write_ai


def _samples(positions: list[tuple[float, float, float]]) -> pd.DataFrame:
    xyz = np.asarray(positions, dtype=float)
    count = len(xyz)
    return pd.DataFrame(
        {
            "lap_id": ["lap"] * count,
            "position.x": xyz[:, 0],
            "position.y": xyz[:, 1],
            "position.z": xyz[:, 2],
            "velocity.x": [20.0] * count,
            "velocity.y": [0.0] * count,
            "velocity.z": [0.0] * count,
            "accel_world_x_ms2": [0.0] * count,
            "accel_world_y_ms2": [0.0] * count,
            "accel_world_z_ms2": [0.0] * count,
        }
    )


def _load_spline(
    tmp_path: Path, points: list[tuple[float, float, float]]
) -> TrackModel:
    root = tmp_path / "track"
    write_ai(root / "ai" / "fast_lane.ai", points)
    return TrackModel.load(root)


def _brute_force_projection(
    points: list[tuple[float, float, float]],
    position: tuple[float, float, float],
    *,
    closed: bool,
) -> tuple[int, float, np.ndarray, float, float]:
    """Independent exact projector used only as a correctness oracle."""
    xyz = np.asarray(points, dtype=float)
    sample = np.asarray(position, dtype=float)
    starts = xyz if closed else xyz[:-1]
    ends = np.roll(xyz, -1, axis=0) if closed else xyz[1:]
    raw_lengths = np.linalg.norm(ends - starts, axis=1)
    lengths = np.where(raw_lengths > 1e-9, raw_lengths, 1e-9)
    best: tuple[float, int, float, np.ndarray] | None = None
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        vector = end - start
        denominator = float(np.dot(vector, vector))
        fraction = (
            0.0
            if denominator <= 1e-12
            else float(np.clip(np.dot(sample - start, vector) / denominator, 0.0, 1.0))
        )
        projection = start + fraction * vector
        distance_squared = float(np.dot(sample - projection, sample - projection))
        candidate = (distance_squared, index, fraction, projection)
        if best is None or distance_squared < best[0] - 1e-12:
            best = candidate

    assert best is not None
    segment_start_s = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
    total_length = float(np.sum(lengths))
    track_s = float(segment_start_s[best[1]] + best[2] * lengths[best[1]])
    if closed:
        track_s %= total_length
    return best[1], best[2], best[3], float(np.sqrt(best[0])), track_s


def _assert_matches_oracle(
    track: TrackModel,
    samples: pd.DataFrame,
    reference_points: list[tuple[float, float, float]],
    query_positions: list[tuple[float, float, float]],
    *,
    assert_segment: bool = False,
) -> pd.DataFrame:
    aligned = track.align(samples, ProcessingConfig())
    canonical_reference_points = [
        tuple(np.asarray(point, dtype=np.float32).astype(float))
        for point in reference_points
    ]
    closed = bool(track.metadata["reference_closed"])
    for row, position in enumerate(query_positions):
        (
            expected_index,
            expected_fraction,
            expected_projection,
            expected_distance,
            expected_s,
        ) = _brute_force_projection(
            canonical_reference_points,
            position,
            closed=closed,
        )
        np.testing.assert_allclose(
            aligned.loc[
                aligned.index[row],
                ["track_projection_x", "track_projection_y", "track_projection_z"],
            ].to_numpy(float),
            expected_projection,
            atol=1e-8,
        )
        assert aligned["track_projection_distance_3d_m"].iloc[row] == pytest.approx(
            expected_distance
        )
        assert aligned["track_s_m"].iloc[row] == pytest.approx(expected_s)
        assert aligned["track_progress"].iloc[row] == pytest.approx(
            expected_s / track.metadata["track_reference_length_m"]
        )
        if assert_segment:
            assert aligned["track_reference_index"].iloc[row] == expected_index
            assert aligned["track_reference_fraction"].iloc[row] == pytest.approx(
                expected_fraction
            )
    return aligned


def _nearest_vertex_counterexample() -> list[tuple[float, float, float]]:
    points = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
    points.extend((500.0 + index * 20.0, 500.0, 0.0) for index in range(18))
    points.extend([(50.0, 0.0, 1.0), (50.0, 100.0, 1.0)])
    points.extend((1000.0 + index * 100.0, 1000.0, 0.0) for index in range(13))
    return points


def _hairpin_points() -> list[tuple[float, float, float]]:
    return [
        (0.0, 0.0, 0.0),
        (100.0, 0.0, 0.0),
        (150.0, 0.0, 50.0),
        (200.0, 0.0, 100.0),
        (200.0, 0.0, 200.0),
        (150.0, 0.0, 250.0),
        (0.0, 0.0, 250.0),
        (-50.0, 0.0, 200.0),
        (-50.0, 0.0, 100.0),
        (50.0, 0.0, 2.0),
        (150.0, 0.0, 2.0),
        (200.0, 0.0, 50.0),
        (250.0, 0.0, 0.0),
        (250.0, 0.0, -100.0),
        (200.0, 0.0, -200.0),
        (100.0, 0.0, -250.0),
        (0.0, 0.0, -250.0),
        (-50.0, 0.0, -200.0),
        (-50.0, 0.0, -100.0),
        (0.0, 0.0, -50.0),
    ]


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


def test_pit_projection_uses_nearby_segments(tmp_path: Path) -> None:
    root = make_track(tmp_path / "track")
    write_ai(
        root / "ai" / "pit_lane.ai",
        [(float(index * 10), 0.0, -10.0) for index in range(13)],
    )
    track = TrackModel.load(root)
    samples = pd.DataFrame(
        {
            "lap_id": ["lap"],
            "position.x": [10.0],
            "position.y": [0.0],
            "position.z": [-10.0],
            "velocity.x": [20.0],
            "velocity.y": [0.0],
            "velocity.z": [0.0],
            "accel_world_x_ms2": [0.0],
            "accel_world_y_ms2": [0.0],
            "accel_world_z_ms2": [0.0],
        }
    )

    aligned = track.align(samples, ProcessingConfig())

    assert aligned["pit_projection_distance_3d_m"].iloc[0] == pytest.approx(0.0)
    assert bool(aligned["is_in_pit"].iloc[0])


def test_projection_is_not_limited_to_single_nearest_vertex(
    tmp_path: Path,
) -> None:
    reference_points = _nearest_vertex_counterexample()
    query_positions = [(50.0, 0.0, 0.0)]
    track = _load_spline(tmp_path, reference_points)

    aligned = _assert_matches_oracle(
        track,
        _samples(query_positions),
        reference_points,
        query_positions,
        assert_segment=True,
    )

    assert aligned["track_reference_index"].iloc[0] == 0
    assert aligned["track_projection_distance_3d_m"].iloc[0] == pytest.approx(0.0)


def test_radius_recovery_finds_segment_outside_knn_seed_candidates(
    tmp_path: Path,
) -> None:
    # Segment 0 passes exactly through the query, but both of its endpoints are
    # farther away than more than eight decoy vertices. A k=8 vertex-only seed
    # therefore cannot include segment 0. The completeness-radius query must
    # recover one of its endpoints before exact segment projection.
    reference_points = [
        (-100.0, 0.0, 0.0),
        (100.0, 0.0, 0.0),
        (-5.0, 0.0, 5.0),
        (-4.0, 0.0, 5.0),
        (-3.0, 0.0, 5.0),
        (-2.0, 0.0, 5.0),
        (-1.0, 0.0, 5.0),
        (0.0, 0.0, 5.0),
        (1.0, 0.0, 5.0),
        (2.0, 0.0, 5.0),
        (3.0, 0.0, 5.0),
        (4.0, 0.0, 5.0),
        (5.0, 0.0, 5.0),
        (200.0, 0.0, 50.0),
    ]
    query_positions = [(0.0, 0.0, 0.0)]
    track = _load_spline(tmp_path, reference_points)

    aligned = _assert_matches_oracle(
        track,
        _samples(query_positions),
        reference_points,
        query_positions,
        assert_segment=True,
    )

    assert aligned["track_reference_index"].iloc[0] == 0
    assert aligned["track_reference_fraction"].iloc[0] == pytest.approx(0.5)
    assert aligned["track_projection_distance_3d_m"].iloc[0] == pytest.approx(0.0)
    assert aligned["track_s_m"].iloc[0] == pytest.approx(100.0)


def test_parallel_branches_use_exact_geometric_distance(tmp_path: Path) -> None:
    reference_points = [
        (0.0, 0.0, 0.0),
        (120.0, 0.0, 0.0),
        (220.0, 0.0, 100.0),
        (220.0, 0.0, 200.0),
        (100.0, 0.0, 300.0),
        (0.0, 0.0, 300.0),
        (120.0, 0.0, 3.0),
        (0.0, 0.0, 3.0),
        (-100.0, 0.0, 100.0),
        (-100.0, 0.0, 200.0),
        (-200.0, 0.0, 300.0),
        (-300.0, 0.0, 300.0),
        (-400.0, 0.0, 300.0),
        (-500.0, 0.0, 300.0),
    ]
    query_positions = [
        (60.0, 0.0, 1.0),
        (60.0, 0.0, 2.5),
        (10.0, 0.0, 2.0),
    ]
    track = _load_spline(tmp_path, reference_points)

    _assert_matches_oracle(
        track,
        _samples(query_positions),
        reference_points,
        query_positions,
    )


def test_hairpin_global_geometry_can_beat_continuity(tmp_path: Path) -> None:
    reference_points = _hairpin_points()
    query_positions = [(100.0, 0.0, 2.0), (50.0, 0.0, 0.0)]
    track = _load_spline(tmp_path, reference_points)

    aligned = _assert_matches_oracle(
        track,
        _samples(query_positions),
        reference_points,
        query_positions,
        assert_segment=True,
    )

    assert aligned["track_reference_index"].tolist() == [9, 0]


def test_closed_loop_seam_projects_and_keeps_canonical_s(tmp_path: Path) -> None:
    reference_points = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (20.0, 0.0, 0.0),
        (20.0, 0.0, 10.0),
        (20.0, 0.0, 20.0),
        (10.0, 0.0, 20.0),
        (0.0, 0.0, 20.0),
        (0.0, 0.0, 10.0),
    ]
    query_positions = [
        (0.0, 0.0, 0.1),
        (0.0, 0.0, 5.0),
        (0.5, 0.0, 5.0),
        (0.1, 0.0, 0.0),
    ]
    track = _load_spline(tmp_path, reference_points)

    aligned = _assert_matches_oracle(
        track,
        _samples(query_positions),
        reference_points,
        query_positions,
    )

    assert aligned["track_s_m"].tolist() == pytest.approx([79.9, 75.0, 75.0, 0.1])
    assert (aligned["track_s_m"] >= 0.0).all()
    assert (aligned["track_s_m"] < track.metadata["track_reference_length_m"]).all()


def test_open_spline_endpoints_do_not_wrap(tmp_path: Path) -> None:
    reference_points = [(float(index * 10), 0.0, 0.0) for index in range(13)]
    query_positions = [
        (-5.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (120.0, 0.0, 0.0),
        (125.0, 0.0, 0.0),
    ]
    track = _load_spline(tmp_path, reference_points)

    aligned = _assert_matches_oracle(
        track,
        _samples(query_positions),
        reference_points,
        query_positions,
    )

    assert track.metadata["reference_closed"] is False
    assert aligned["track_s_m"].tolist() == pytest.approx([0.0, 0.0, 120.0, 120.0])


def test_far_off_track_projection_has_no_distance_threshold(tmp_path: Path) -> None:
    reference_points = _hairpin_points()
    query_positions = [(1000.0, 100.0, -1000.0), (-1000.0, -100.0, 1000.0)]
    track = _load_spline(tmp_path, reference_points)

    aligned = _assert_matches_oracle(
        track,
        _samples(query_positions),
        reference_points,
        query_positions,
    )

    assert (aligned["track_projection_distance_3d_m"] > 25.0).all()


def test_degenerate_reference_segment_does_not_produce_nan(tmp_path: Path) -> None:
    reference_points = [(float(index * 10), 0.0, 0.0) for index in range(13)]
    reference_points.insert(7, reference_points[6])
    track = _load_spline(tmp_path, reference_points)
    samples = _samples([(60.0, 2.0, 0.0)])

    aligned = track.align(samples, ProcessingConfig())

    columns = [
        "track_reference_fraction",
        "track_projection_x",
        "track_projection_y",
        "track_projection_z",
        "track_projection_distance_3d_m",
        "track_s_m",
    ]
    assert np.isfinite(aligned[columns].to_numpy(float)).all()
    assert aligned["track_projection_distance_3d_m"].iloc[0] == pytest.approx(2.0)


def test_randomized_projection_matches_brute_force_oracle(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260826)
    for case in range(12):
        reference_array = np.cumsum(rng.normal(0.0, 8.0, size=(24, 3)), axis=0)
        # Force at least one unusually long segment so the randomized suite also
        # exercises geometries where nearest vertices are a poor proxy for the
        # nearest segment. Keep the spline open by moving the final endpoint far
        # from the start.
        reference_array[-1] += (2000.0, 0.0, 0.0)
        reference_points = [tuple(row) for row in reference_array]
        query_positions: list[tuple[float, float, float]] = []

        # Ordinary near-track telemetry-like samples.
        for _ in range(50):
            index = int(rng.integers(0, len(reference_points) - 1))
            fraction = float(rng.uniform(0.0, 1.0))
            start = reference_array[index]
            vector = reference_array[index + 1] - start
            query = start + fraction * vector + rng.normal(0.0, 4.0, size=3)
            query_positions.append(tuple(query))

        # Far-off-track samples ensure correctness is not coupled to a distance
        # threshold or to telemetry being close to the racing surface.
        for _ in range(25):
            query_positions.append(tuple(rng.uniform(-500.0, 500.0, size=3)))

        # Bias additional queries toward the longest segment. These are useful
        # adversarial cases because a point can be close to the middle of a long
        # segment while both endpoints are relatively far away.
        segment_vectors = np.diff(reference_array, axis=0)
        longest = int(np.argmax(np.linalg.norm(segment_vectors, axis=1)))
        start = reference_array[longest]
        vector = segment_vectors[longest]
        for _ in range(25):
            fraction = float(rng.uniform(0.05, 0.95))
            query = start + fraction * vector + rng.normal(0.0, 6.0, size=3)
            query_positions.append(tuple(query))

        track = _load_spline(tmp_path / f"case-{case}", reference_points)
        _assert_matches_oracle(
            track,
            _samples(query_positions),
            reference_points,
            query_positions,
        )


def test_projection_is_batch_and_order_independent(tmp_path: Path) -> None:
    reference_points = _hairpin_points()
    query_positions = [(50.0, 0.0, 2.0), (50.0, 0.0, 0.0), (400.0, 0.0, 400.0)]
    track = _load_spline(tmp_path, reference_points)
    fields = [
        "track_reference_index",
        "track_reference_fraction",
        "track_projection_x",
        "track_projection_y",
        "track_projection_z",
        "track_projection_distance_3d_m",
        "track_s_m",
    ]

    samples = _samples(query_positions)
    batch = track.align(samples, ProcessingConfig())
    for row in range(len(samples)):
        individual = track.align(samples.iloc[[row]], ProcessingConfig())
        np.testing.assert_allclose(
            batch.iloc[[row]][fields].to_numpy(float),
            individual[fields].to_numpy(float),
            atol=1e-8,
        )

    samples["sample_id"] = np.arange(len(samples))
    permuted = track.align(
        samples.iloc[[2, 0, 1]].reset_index(drop=True),
        ProcessingConfig(),
    )
    restored = permuted.sort_values("sample_id").reset_index(drop=True)
    np.testing.assert_allclose(
        batch[fields].to_numpy(float),
        restored[fields].to_numpy(float),
        atol=1e-8,
    )


def test_projection_matches_oracle_across_radius_query_chunks(
    tmp_path: Path,
) -> None:
    phase = np.linspace(0.0, 2.0 * np.pi, 500, endpoint=False)
    reference_points = [
        (float(100.0 * np.cos(value)), 0.0, float(100.0 * np.sin(value)))
        for value in phase
    ]
    query_positions = [(0.0, 0.0, 0.0)] * 8_193
    track = _load_spline(tmp_path, reference_points)

    aligned = track.align(_samples(query_positions), ProcessingConfig())
    canonical_reference_points = [
        tuple(np.asarray(point, dtype=np.float32).astype(float))
        for point in reference_points
    ]
    (
        expected_index,
        expected_fraction,
        expected_projection,
        expected_distance,
        expected_s,
    ) = _brute_force_projection(
        canonical_reference_points,
        query_positions[0],
        closed=bool(track.metadata["reference_closed"]),
    )
    fields = [
        "track_reference_index",
        "track_reference_fraction",
        "track_projection_x",
        "track_projection_y",
        "track_projection_z",
        "track_projection_distance_3d_m",
        "track_s_m",
    ]
    expected = np.tile(
        [
            expected_index,
            expected_fraction,
            *expected_projection,
            expected_distance,
            expected_s,
        ],
        (len(query_positions), 1),
    )
    np.testing.assert_allclose(aligned[fields].to_numpy(float), expected, atol=1e-8)
