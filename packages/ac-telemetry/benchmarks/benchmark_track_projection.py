"""Benchmark the exact TrackModel projection path on synthetic telemetry.

Example:
    uv run python packages/ac-telemetry/benchmarks/benchmark_track_projection.py --samples 250000

The geometry and samples are generated before timing so repeated runs compare
only projection/alignment work.  The reference uses the same internal geometry
representation as ``TrackModel.load``; this keeps the benchmark independent of
proprietary AC files while still exercising the production projector.
"""

import argparse
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from ac_telemetry.config import ProcessingConfig
from ac_telemetry.track import TrackModel, _project_track_points, _Spline
from scipy.spatial import cKDTree


def _make_spline(
    *, point_count: int, offset: tuple[float, float], uniform_circle: bool = False
) -> _Spline:
    phase = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    radius = (
        np.full(point_count, 900.0)
        if uniform_circle
        else 900.0 + 85.0 * np.sin(3.0 * phase) + 35.0 * np.cos(7.0 * phase)
    )
    points = np.column_stack(
        (
            radius * np.cos(phase) + offset[0],
            np.zeros(point_count) if uniform_circle else 4.0 * np.sin(2.0 * phase),
            radius * np.sin(phase) + offset[1]
            if uniform_circle
            else (radius - 25.0 * np.sin(5.0 * phase)) * np.sin(phase) + offset[1],
        )
    )
    starts = points
    vectors = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(vectors, axis=1)
    length_squared = np.einsum("ij,ij->i", vectors, vectors)
    tangent = vectors / lengths[:, None]
    horizontal = tangent[:, [0, 2]]
    horizontal /= np.linalg.norm(horizontal, axis=1)[:, None]
    left = np.stack((horizontal[:, 1], -horizontal[:, 0]), axis=1)
    heading = np.unwrap(np.arctan2(horizontal[:, 0], horizontal[:, 1]))
    next_heading = np.roll(heading, -1)
    change = (next_heading - heading + np.pi) % (2.0 * np.pi) - np.pi
    curvature = change / lengths
    distance = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
    payload = np.zeros((point_count, 18), dtype=float)
    payload[:, 5:7] = 8.0
    spline_fields = {field.name for field in _Spline.__dataclass_fields__.values()}
    spline_values = dict(
        source_path=Path("<synthetic>"),
        source_hash="synthetic",
        closed=True,
        points=points,
        stored_distance_m=distance.copy(),
        payload=payload,
        distance_m=distance,
        segment_start=starts,
        segment_vector=vectors,
        segment_length_m=lengths,
        tangent=tangent,
        horizontal_tangent=horizontal,
        left_normal=left,
        curvature_1pm=curvature,
        total_length_m=float(lengths.sum()),
        point_tree=cKDTree(points),
    )
    if "segment_length_squared" in spline_fields:
        spline_values["segment_length_squared"] = length_squared
    return _Spline(**spline_values)


def _make_samples(
    reference: _Spline, sample_count: int, *, dense_radius: bool = False
) -> pd.DataFrame:
    rng = np.random.default_rng(20260826)
    if dense_radius:
        positions = np.zeros((sample_count, 3), dtype=float)
        positions[:, [0, 2]] = rng.normal(0.0, 0.01, size=(sample_count, 2))
    else:
        indices = rng.integers(0, reference.segment_count, size=sample_count)
        fraction = rng.random(sample_count)
        on_segment = reference.segment_start[indices] + (
            reference.segment_vector[indices] * fraction[:, None]
        )
        offset = rng.normal(0.0, 3.5, size=(sample_count, 2))
        positions = on_segment.copy()
        positions[:, [0, 2]] += offset

        # Two disjoint 1% groups keep difficult/off-track recovery in the
        # telemetry workload; together they account for 2% of samples.
        hard_count = max(1, sample_count // 100)
        positions[:hard_count, 0] += 1800.0
        positions[hard_count : 2 * hard_count, 2] -= 1600.0

    return pd.DataFrame(
        {
            "lap_id": np.zeros(sample_count, dtype=np.int64),
            "position.x": positions[:, 0],
            "position.y": positions[:, 1],
            "position.z": positions[:, 2],
            "velocity.x": np.full(sample_count, 60.0),
            "velocity.y": np.zeros(sample_count),
            "velocity.z": np.full(sample_count, 10.0),
            "accel_world_x_ms2": np.zeros(sample_count),
            "accel_world_y_ms2": np.zeros(sample_count),
            "accel_world_z_ms2": np.zeros(sample_count),
        }
    )


def _median_seconds(function, iterations: int) -> float:
    measurements = []
    for _ in range(iterations):
        started = time.perf_counter()
        function()
        measurements.append(time.perf_counter() - started)
    return float(median(measurements))


def _peak_traced_megabytes(function) -> float:
    tracemalloc.start()
    tracemalloc.reset_peak()
    function()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=("telemetry", "dense-radius"),
        default="telemetry",
    )
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--reference-points", type=int, default=4_096)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--measure-memory", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.reference_points < 2 or args.iterations < 1:
        parser.error("samples, reference-points, and iterations must be positive")

    dense_radius = args.scenario == "dense-radius"
    reference = _make_spline(
        point_count=args.reference_points,
        offset=(0.0, 0.0),
        uniform_circle=dense_radius,
    )
    pit_reference = (
        None
        if dense_radius
        else _make_spline(
            point_count=args.reference_points,
            offset=(35.0, -35.0),
        )
    )
    track = TrackModel(
        track_dir=Path("<synthetic>"),
        reference=reference,
        pit_reference=pit_reference,
        sections=(),
        drs_zones=(),
        ui_metadata={},
    )
    samples = _make_samples(reference, args.samples, dense_radius=dense_radius)
    points = samples[["position.x", "position.y", "position.z"]].to_numpy(float)
    config = ProcessingConfig()

    # Warm up imports, dispatch, and the spatial index before recording timings.
    _project_track_points(reference, points[: min(1_000, len(points))])

    main_seconds = _median_seconds(
        lambda: _project_track_points(reference, points), args.iterations
    )
    align_seconds = _median_seconds(
        lambda: track.align(samples, config), args.iterations
    )
    if pit_reference is None:
        pit_seconds = None
        main_only_seconds = align_seconds
        align_with_pit_seconds = None
    else:
        pit_seconds = _median_seconds(
            lambda: _project_track_points(pit_reference, points), args.iterations
        )
        # Keep the import of dataclasses.replace useful to profilers that compare
        # the main-only and main-plus-pit variants without changing this benchmark's
        # default workload.
        main_only = replace(track, pit_reference=None)
        main_only_seconds = _median_seconds(
            lambda: main_only.align(samples, config), args.iterations
        )
        align_with_pit_seconds = align_seconds

    print(f"scenario={args.scenario}")
    print(f"samples={args.samples} reference_points={args.reference_points}")
    print(f"iterations={args.iterations}")
    print(f"main_projection_median_s={main_seconds:.6f}")
    print(
        f"pit_projection_median_s={pit_seconds:.6f}"
        if pit_seconds is not None
        else "pit_projection_median_s=n/a"
    )
    print(f"align_main_only_median_s={main_only_seconds:.6f}")
    print(
        f"align_with_pit_median_s={align_with_pit_seconds:.6f}"
        if align_with_pit_seconds is not None
        else "align_with_pit_median_s=n/a"
    )
    if not dense_radius:
        hard_count = max(1, args.samples // 100)
        print(f"hard_sample_fraction={2 * hard_count / args.samples:.6%}")
    if args.measure_memory:
        peak = _peak_traced_megabytes(lambda: _project_track_points(reference, points))
        print(f"main_projection_peak_traced_mb={peak:.2f}")


if __name__ == "__main__":
    main()
