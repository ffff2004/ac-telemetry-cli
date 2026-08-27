from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ProcessingConfig
from .contract_types import (
    ColumnAvailability,
    ColumnSpec,
    ForeignKey,
    MergeMode,
    TableSpec,
)
from .sections import parse_sections_ini
from .util import contiguous_true_runs, json_load

SUPPORTED_COORDINATES = {"track_s_m", "track_progress"}
_LAP_START_SAMPLE_TOLERANCE_M = 50.0

_REQUIRED = ColumnAvailability.REQUIRED
_OPTIONAL = ColumnAvailability.OPTIONAL

_PASS_COLUMN_SPECS = (
    ColumnSpec(
        "session_id",
        _REQUIRED,
        False,
        "Identifier of the session containing this pass.",
    ),
    ColumnSpec(
        "lap_id", _REQUIRED, False, "Identifier of the lap containing this pass."
    ),
    ColumnSpec(
        "source_lap_number",
        _OPTIONAL,
        True,
        "Replay-reported lap number for the containing lap.",
    ),
    ColumnSpec(
        "segment_id",
        _REQUIRED,
        False,
        "Identifier of the configured parent segment or subsegment.",
    ),
    ColumnSpec(
        "segment_name", _REQUIRED, True, "Display name of the configured segment."
    ),
    ColumnSpec(
        "parent_segment_id",
        _OPTIONAL,
        True,
        "Identifier of the parent segment when this row represents a subsegment.",
    ),
    ColumnSpec(
        "coordinate",
        _OPTIONAL,
        True,
        "Coordinate system used by the source segment definitions.",
    ),
    ColumnSpec(
        "segment_start_track_s_m",
        _OPTIONAL,
        True,
        "Segment entry coordinate projected onto the track reference, in metres.",
    ),
    ColumnSpec(
        "segment_end_track_s_m",
        _OPTIONAL,
        True,
        "Segment exit coordinate projected onto the track reference, in metres.",
    ),
    ColumnSpec(
        "sample_count",
        _OPTIONAL,
        True,
        "Number of telemetry samples included in the pass.",
    ),
    ColumnSpec(
        "segment_time_s",
        _REQUIRED,
        True,
        "Elapsed time from the interpolated segment entry to exit, in seconds.",
    ),
    ColumnSpec(
        "entry_speed_kmh",
        _REQUIRED,
        True,
        "Interpolated vehicle speed at segment entry, in kilometres per hour.",
    ),
    ColumnSpec(
        "exit_speed_kmh",
        _REQUIRED,
        True,
        "Interpolated vehicle speed at segment exit, in kilometres per hour.",
    ),
    ColumnSpec(
        "minimum_speed_kmh",
        _REQUIRED,
        True,
        "Lowest sampled speed within the segment, in kilometres per hour.",
    ),
    ColumnSpec(
        "minimum_speed_track_s_m",
        _OPTIONAL,
        True,
        "Track-reference coordinate of the minimum-speed sample, in metres.",
    ),
    ColumnSpec(
        "brake_onset_track_s_m",
        _REQUIRED,
        True,
        "Track-reference coordinate where the selected braking run starts, in metres.",
    ),
    ColumnSpec(
        "brake_release_track_s_m",
        _OPTIONAL,
        True,
        "Track-reference coordinate where the selected braking run ends, in metres.",
    ),
    ColumnSpec(
        "brake_duration_s",
        _OPTIONAL,
        True,
        "Duration of active braking in the selected braking run, in seconds.",
    ),
    ColumnSpec(
        "peak_brake",
        _OPTIONAL,
        True,
        "Maximum brake input in the selected braking run or pass.",
    ),
    ColumnSpec(
        "brake_impulse_proxy_s",
        _OPTIONAL,
        True,
        "Time integral of brake input over the selected braking run.",
    ),
    ColumnSpec(
        "throttle_pickup_track_s_m",
        _OPTIONAL,
        True,
        "First post-minimum-speed track coordinate meeting the throttle pickup threshold, in metres.",
    ),
    ColumnSpec(
        "full_throttle_commit_track_s_m",
        _REQUIRED,
        True,
        "Track coordinate where sustained full-throttle commitment begins, in metres.",
    ),
    ColumnSpec(
        "coasting_time_s",
        _REQUIRED,
        True,
        "Time spent coasting within the segment, in seconds.",
    ),
    ColumnSpec(
        "partial_throttle_time_s",
        _OPTIONAL,
        True,
        "Time spent at partial throttle within the segment, in seconds.",
    ),
    ColumnSpec(
        "rear_slip_integral",
        _OPTIONAL,
        True,
        "Time integral of positive maximum rear-wheel slip ratio.",
    ),
    ColumnSpec(
        "max_abs_steer",
        _OPTIONAL,
        True,
        "Largest absolute steering angle within the segment.",
    ),
    ColumnSpec(
        "steering_reversal_count",
        _OPTIONAL,
        True,
        "Number of significant steering-direction reversals within the segment.",
    ),
    ColumnSpec(
        "actual_path_length_m",
        _OPTIONAL,
        True,
        "Driven three-dimensional path length through the segment, in metres.",
    ),
    ColumnSpec(
        "reference_arc_length_m",
        _OPTIONAL,
        True,
        "Track-reference arc length between segment boundaries, in metres.",
    ),
    ColumnSpec(
        "path_excess_m",
        _OPTIONAL,
        True,
        "Driven path length minus the reference arc length, in metres.",
    ),
    ColumnSpec(
        "entry_lateral_offset_m",
        _OPTIONAL,
        True,
        "Interpolated lateral offset from the track reference at entry, in metres.",
    ),
    ColumnSpec(
        "exit_lateral_offset_m",
        _OPTIONAL,
        True,
        "Interpolated lateral offset from the track reference at exit, in metres.",
    ),
    ColumnSpec(
        "minimum_speed_lateral_offset_m",
        _OPTIONAL,
        True,
        "Lateral offset at the segment minimum-speed sample, in metres.",
    ),
    ColumnSpec(
        "entry_velocity_cross_track_ms",
        _OPTIONAL,
        True,
        "Interpolated cross-track velocity at entry, in metres per second.",
    ),
    ColumnSpec(
        "exit_velocity_cross_track_ms",
        _OPTIONAL,
        True,
        "Interpolated cross-track velocity at exit, in metres per second.",
    ),
    ColumnSpec(
        "entry_heading_error_rad",
        _OPTIONAL,
        True,
        "Interpolated velocity-heading error at entry, in radians.",
    ),
    ColumnSpec(
        "exit_heading_error_rad",
        _OPTIONAL,
        True,
        "Interpolated velocity-heading error at exit, in radians.",
    ),
    ColumnSpec(
        "entry_gear",
        _OPTIONAL,
        True,
        "Discrete physical gear selected at segment entry.",
    ),
    ColumnSpec(
        "exit_gear", _OPTIONAL, True, "Discrete physical gear selected at segment exit."
    ),
    ColumnSpec(
        "entry_rpm", _OPTIONAL, True, "Interpolated engine speed at segment entry."
    ),
    ColumnSpec(
        "exit_rpm", _OPTIONAL, True, "Interpolated engine speed at segment exit."
    ),
    ColumnSpec(
        "entry_throttle",
        _OPTIONAL,
        True,
        "Interpolated throttle input at segment entry.",
    ),
    ColumnSpec(
        "exit_throttle", _OPTIONAL, True, "Interpolated throttle input at segment exit."
    ),
    ColumnSpec(
        "entry_brake", _OPTIONAL, True, "Interpolated brake input at segment entry."
    ),
    ColumnSpec(
        "exit_brake", _OPTIONAL, True, "Interpolated brake input at segment exit."
    ),
    ColumnSpec(
        "entry_steer", _OPTIONAL, True, "Interpolated steering angle at segment entry."
    ),
    ColumnSpec(
        "exit_steer", _OPTIONAL, True, "Interpolated steering angle at segment exit."
    ),
    ColumnSpec(
        "peak_abs_track_lat_g",
        _OPTIONAL,
        True,
        "Largest absolute track-relative lateral acceleration in g.",
    ),
    ColumnSpec(
        "minimum_track_long_g",
        _OPTIONAL,
        True,
        "Lowest track-relative longitudinal acceleration in g.",
    ),
    ColumnSpec(
        "is_complete_lap",
        _OPTIONAL,
        True,
        "Whether the containing lap is confirmed complete.",
    ),
    ColumnSpec(
        "is_valid_lap",
        _OPTIONAL,
        True,
        "Whether the containing lap passes lap-quality validation.",
    ),
    ColumnSpec(
        "valid_for_comparison",
        _REQUIRED,
        True,
        "Whether this pass is eligible for valid-lap segment comparisons.",
    ),
)

SEGMENT_TABLE_SPECS = (
    TableSpec(
        "segments/passes",
        _PASS_COLUMN_SPECS,
        ("session_id", "lap_id", "segment_id"),
        False,
        MergeMode.KEYED,
        (
            ForeignKey(("session_id",), "sessions", ("session_id",)),
            ForeignKey(("lap_id",), "laps", ("lap_id",)),
            ForeignKey(("session_id", "lap_id"), "laps", ("session_id", "lap_id")),
        ),
        allows_untyped_empty_frame=True,
        empty_frame_columns=(),
    ),
)


def generate_segments_from_sections_ini(
    sections_text: str, *, track: str | None = None
) -> dict[str, Any]:
    """Generate continuous analysis segments from ``sections.ini`` text.

    Each segment starts at one section's ``IN`` value and ends at the next
    section's ``IN`` value. The lap prefix and suffix are represented by the
    ``0.0`` and ``1.0`` boundaries. ``OUT`` is intentionally not used because
    the generated segments include each section's exit.
    """
    sections = sorted(
        parse_sections_ini(sections_text), key=lambda section: section.in_progress
    )
    segments: list[dict[str, Any]] = []
    if sections and sections[0].in_progress > 0.0:
        first = sections[0]
        segments.append(
            {
                "id": f"start_to_{first.section_id.lower()}",
                "name": f"Start to {first.name}",
                "start": 0.0,
                "end": first.in_progress,
            }
        )

    for index, current in enumerate(sections):
        following = sections[index + 1] if index + 1 < len(sections) else None
        end = following.in_progress if following is not None else 1.0
        if end <= current.in_progress:
            continue
        segments.append(
            {
                "id": (
                    f"{current.section_id.lower()}_to_{following.section_id.lower()}"
                    if following is not None
                    else f"{current.section_id.lower()}_to_finish"
                ),
                "name": (
                    f"{current.name} + exit to {following.name}"
                    if following is not None
                    else f"{current.name} + exit to finish"
                ),
                "start": current.in_progress,
                "end": end,
            }
        )

    definitions: dict[str, Any] = {
        "coordinate": "track_progress",
        "description": "Automatically generated from sections.ini IN boundaries (IN-to-next-IN) because no explicit segments file is supplied.",
        "segments": segments,
    }
    if track is not None:
        definitions["track"] = track
    _validate_segment_definitions(definitions)
    return definitions


def load_segment_definitions(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = json_load(path)
    coordinate = data.get("coordinate", "track_s_m")
    if coordinate not in SUPPORTED_COORDINATES:
        raise ValueError(
            f"Unsupported segment coordinate {coordinate!r}; expected one of {sorted(SUPPORTED_COORDINATES)}"
        )
    if not isinstance(data.get("segments"), list):
        raise ValueError("Segment definition must contain a segments list")
    _validate_segment_definitions(data)
    return data


def _validate_segment_definitions(definitions: dict[str, Any]) -> None:
    for parent in definitions["segments"]:
        candidates = [parent, *parent.get("subsegments", [])]
        for definition in candidates:
            start = float(definition["start"])
            end = float(definition["end"])
            if end < start:
                identifier = definition.get("id", "<unnamed>")
                raise ValueError(
                    f"Segment {identifier!r} crosses the lap boundary "
                    f"(start={start:g}, end={end:g}); split it at 1.0/0.0"
                )


def _to_m(value: float, coordinate: str, track_length_m: float) -> float:
    return value * track_length_m if coordinate == "track_progress" else value


def _crossing(
    lap: pd.DataFrame,
    target_wrapped_m: float,
    track_length_m: float,
    minimum_unwrapped_m: float | None = None,
) -> tuple[int, int, float] | None:
    s = lap["track_s_unwrapped_m"].to_numpy(float)
    if len(s) < 2 or not np.isfinite(s).any():
        return None
    if (
        minimum_unwrapped_m is None
        and abs(target_wrapped_m) <= 1e-9
        and 0.0 < s[0] <= _LAP_START_SAMPLE_TOLERANCE_M
    ):
        # Replay samples usually begin a few metres after the exact start/finish
        # coordinate. Treat the first sample as the lap-start state instead of
        # incorrectly selecting the finish-line crossing at the end of the lap.
        return 0, 1, 0.0
    lo = (
        int(np.ceil((minimum_unwrapped_m - target_wrapped_m) / track_length_m))
        if minimum_unwrapped_m is not None
        else int(np.floor((np.nanmin(s) - target_wrapped_m) / track_length_m)) - 1
    )
    hi = int(np.ceil((np.nanmax(s) - target_wrapped_m) / track_length_m)) + 1
    for cycle in range(lo, hi + 1):
        target = target_wrapped_m + cycle * track_length_m
        for i in range(len(s) - 1):
            a, b = s[i], s[i + 1]
            if not np.isfinite(a) or not np.isfinite(b):
                continue
            if a <= target <= b and b - a > 1e-9:
                return i, i + 1, float((target - a) / (b - a))
    return None


def _interpolate_numeric(a: Any, b: Any, alpha: float) -> float:
    try:
        av, bv = float(a), float(b)
    except TypeError, ValueError:
        return np.nan
    if not np.isfinite(av) or not np.isfinite(bv):
        return np.nan
    return av + alpha * (bv - av)


def _state_at(
    lap: pd.DataFrame,
    target_wrapped_m: float,
    track_length_m: float,
    minimum_unwrapped_m: float | None = None,
) -> dict[str, Any] | None:
    crossing = _crossing(
        lap,
        target_wrapped_m,
        track_length_m,
        minimum_unwrapped_m=minimum_unwrapped_m,
    )
    if crossing is None:
        return None
    i, j, alpha = crossing
    first, second = lap.iloc[i], lap.iloc[j]
    numeric = [
        "lap_time_s",
        "speed_kmh",
        "throttle",
        "brake_n",
        "rpm",
        "fuel",
        "steerAngle",
        "lateral_offset_m",
        "velocity_cross_track_ms",
        "velocity_heading_error_rad",
        "track_long_g",
        "track_lat_g",
        "path_distance_3d_m",
    ]
    state: dict[str, Any] = {
        name: _interpolate_numeric(first.get(name), second.get(name), alpha)
        for name in numeric
    }
    state["gear_physical"] = int(
        first["gear_physical"] if alpha < 0.5 else second["gear_physical"]
    )
    state["sample_before"] = int(first["sample_index"])
    state["sample_after"] = int(second["sample_index"])
    state["track_s_unwrapped_m"] = float(
        first["track_s_unwrapped_m"]
        + alpha * (second["track_s_unwrapped_m"] - first["track_s_unwrapped_m"])
    )
    return state


def _main_braking_run(
    segment: pd.DataFrame, minimum_index: int
) -> tuple[int, int] | None:
    mask = segment["is_braking"].fillna(False).to_numpy(bool)
    runs = contiguous_true_runs(mask)
    if not runs:
        return None
    candidates = [run for run in runs if run[0] <= minimum_index] or runs
    return max(
        candidates,
        key=lambda run: float(
            (
                segment.iloc[run[0] : run[1] + 1]["brake_n"]
                * segment.iloc[run[0] : run[1] + 1]["dt_s"]
            ).sum()
        ),
    )


def _full_throttle_commit_index(
    segment: pd.DataFrame, start: int, config: ProcessingConfig
) -> int | None:
    mask = segment["is_full_throttle"].fillna(False).to_numpy(bool).copy()
    dt = segment["dt_s"].to_numpy(float)
    for gap_start, gap_end in contiguous_true_runs(~mask):
        if gap_start == 0 or gap_end == len(mask) - 1:
            continue
        if (
            float(dt[gap_start : gap_end + 1].sum())
            <= config.full_throttle_commit_gap_s
        ):
            mask[gap_start : gap_end + 1] = True
    for run_start, run_end in contiguous_true_runs(mask):
        if run_end < start:
            continue
        onset = max(run_start, start)
        if float(dt[onset : run_end + 1].sum()) >= config.full_throttle_commit_min_s:
            return onset
    return None


def _steering_reversals(values: np.ndarray, threshold: float) -> int:
    significant = values[np.abs(values) >= threshold]
    if len(significant) < 2:
        return 0
    signs = np.sign(significant)
    compressed = signs[np.r_[True, signs[1:] != signs[:-1]]]
    return max(0, len(compressed) - 1)


def segment_passes(
    samples: pd.DataFrame,
    laps: pd.DataFrame,
    definitions: dict[str, Any] | None,
    config: ProcessingConfig,
    track_length_m: float,
) -> pd.DataFrame:
    if definitions is None:
        return pd.DataFrame()
    _validate_segment_definitions(definitions)
    coordinate = definitions.get("coordinate", "track_s_m")
    rows: list[dict[str, Any]] = []
    lap_lookup = laps.set_index("lap_id", drop=False)

    expanded: list[dict[str, Any]] = []
    for parent in definitions["segments"]:
        expanded.append(parent)
        for child in parent.get("subsegments", []):
            item = dict(child)
            item["parent_id"] = parent.get("id")
            expanded.append(item)

    for lap_id, original in samples.groupby("lap_id", sort=False):
        lap = original.sort_values("sample_index").reset_index(drop=True)
        if lap_id not in lap_lookup.index:
            continue
        lap_info = lap_lookup.loc[lap_id]
        for definition in expanded:
            start_m = (
                _to_m(float(definition["start"]), coordinate, track_length_m)
                % track_length_m
            )
            end_m = (
                _to_m(float(definition["end"]), coordinate, track_length_m)
                % track_length_m
            )
            entry = _state_at(lap, start_m, track_length_m)
            if entry is None:
                continue
            exit_state = _state_at(
                lap,
                end_m,
                track_length_m,
                minimum_unwrapped_m=entry["track_s_unwrapped_m"],
            )
            if exit_state is None:
                continue
            start_time = float(entry["lap_time_s"])
            end_time = float(exit_state["lap_time_s"])
            if end_time <= start_time:
                continue
            segment = lap[
                (lap["lap_time_s"] >= start_time) & (lap["lap_time_s"] <= end_time)
            ].copy()
            if len(segment) < 2:
                continue
            segment = segment.reset_index(drop=True)
            min_index = int(segment["speed_kmh"].idxmin())
            minimum = segment.iloc[min_index]
            braking = _main_braking_run(segment, min_index)
            throttle_after_min = segment.iloc[min_index:]
            pickup_candidates = throttle_after_min.index[
                throttle_after_min["throttle"] >= config.throttle_event_threshold
            ]
            pickup_index = int(pickup_candidates[0]) if len(pickup_candidates) else None
            commit_index = _full_throttle_commit_index(segment, min_index, config)

            if braking is None:
                brake_start_s = brake_end_s = np.nan
                brake_duration_s = 0.0
                peak_brake = float(segment["brake_n"].max())
                brake_impulse = 0.0
            else:
                b0, b1 = braking
                brake_segment = segment.iloc[b0 : b1 + 1]
                brake_start_s = float(brake_segment["track_s_m"].iloc[0])
                brake_end_s = float(brake_segment["track_s_m"].iloc[-1])
                brake_duration_s = float(
                    brake_segment.loc[brake_segment["is_braking"], "dt_s"].sum()
                )
                peak_brake = float(brake_segment["brake_n"].max())
                brake_impulse = float(
                    (brake_segment["brake_n"] * brake_segment["dt_s"]).sum()
                )

            reference_arc = (end_m - start_m) % track_length_m
            if reference_arc <= 1e-9:
                reference_arc = track_length_m
            path_length = float(
                exit_state["path_distance_3d_m"] - entry["path_distance_3d_m"]
            )
            if path_length < 0:
                path_length = float(
                    segment[["position.x", "position.y", "position.z"]]
                    .diff()
                    .fillna(0.0)
                    .pow(2)
                    .sum(axis=1)
                    .pow(0.5)
                    .sum()
                )

            row = {
                "session_id": segment["session_id"].iloc[0],
                "lap_id": lap_id,
                "source_lap_number": int(segment["source_lap_number"].iloc[0]),
                "segment_id": definition.get("id"),
                "segment_name": definition.get("name", definition.get("id")),
                "parent_segment_id": definition.get("parent_id"),
                "coordinate": coordinate,
                "segment_start_track_s_m": start_m,
                "segment_end_track_s_m": end_m,
                "sample_count": len(segment),
                "segment_time_s": end_time - start_time,
                "entry_speed_kmh": entry["speed_kmh"],
                "exit_speed_kmh": exit_state["speed_kmh"],
                "minimum_speed_kmh": float(minimum["speed_kmh"]),
                "minimum_speed_track_s_m": float(minimum["track_s_m"]),
                "brake_onset_track_s_m": brake_start_s,
                "brake_release_track_s_m": brake_end_s,
                "brake_duration_s": brake_duration_s,
                "peak_brake": peak_brake,
                "brake_impulse_proxy_s": brake_impulse,
                "throttle_pickup_track_s_m": float(
                    segment.loc[pickup_index, "track_s_m"]
                )
                if pickup_index is not None
                else np.nan,
                "full_throttle_commit_track_s_m": float(
                    segment.loc[commit_index, "track_s_m"]
                )
                if commit_index is not None
                else np.nan,
                "coasting_time_s": float(
                    segment.loc[segment["is_coasting"], "dt_s"].sum()
                ),
                "partial_throttle_time_s": float(
                    segment.loc[segment["is_partial_throttle"], "dt_s"].sum()
                ),
                "rear_slip_integral": float(
                    (segment["rear_slip_ratio_max"].clip(0) * segment["dt_s"]).sum()
                ),
                "max_abs_steer": float(segment["steerAngle"].abs().max()),
                "steering_reversal_count": _steering_reversals(
                    segment["steerAngle"].fillna(0).to_numpy(float),
                    config.steering_reversal_threshold,
                ),
                "actual_path_length_m": path_length,
                "reference_arc_length_m": reference_arc,
                "path_excess_m": path_length - reference_arc,
                "entry_lateral_offset_m": entry["lateral_offset_m"],
                "exit_lateral_offset_m": exit_state["lateral_offset_m"],
                "minimum_speed_lateral_offset_m": float(minimum["lateral_offset_m"]),
                "entry_velocity_cross_track_ms": entry["velocity_cross_track_ms"],
                "exit_velocity_cross_track_ms": exit_state["velocity_cross_track_ms"],
                "entry_heading_error_rad": entry["velocity_heading_error_rad"],
                "exit_heading_error_rad": exit_state["velocity_heading_error_rad"],
                "entry_gear": entry["gear_physical"],
                "exit_gear": exit_state["gear_physical"],
                "entry_rpm": entry["rpm"],
                "exit_rpm": exit_state["rpm"],
                "entry_throttle": entry["throttle"],
                "exit_throttle": exit_state["throttle"],
                "entry_brake": entry["brake_n"],
                "exit_brake": exit_state["brake_n"],
                "entry_steer": entry["steerAngle"],
                "exit_steer": exit_state["steerAngle"],
                "peak_abs_track_lat_g": float(segment["track_lat_g"].abs().max()),
                "minimum_track_long_g": float(segment["track_long_g"].min()),
                "is_complete_lap": bool(lap_info["is_complete"]),
                "is_valid_lap": bool(lap_info["is_valid"]),
                "valid_for_comparison": bool(lap_info["is_valid"]),
            }
            rows.append(row)
    return pd.DataFrame(rows)
