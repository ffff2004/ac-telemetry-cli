from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from .assist_activity import detect_abs_activity, detect_tc_activity
from .config import ProcessingConfig
from .contract_types import (
    ColumnAvailability,
    ColumnSpec,
    ForeignKey,
    MergeMode,
    TableSpec,
)
from .util import contiguous_true_runs, stable_id

WHEELS = ("fl", "fr", "rl", "rr")
Wheel = Literal["fl", "fr", "rl", "rr"]

EVENT_INDEX_COLUMNS = [
    "event_id",
    "session_id",
    "lap_id",
    "event_type",
    "start_sample",
    "end_sample_exclusive",
    "start_time_s",
    "end_time_s",
    "span_duration_s",
    "active_duration_s",
    "start_track_s_m",
    "end_track_s_m",
    "start_track_progress",
    "end_track_progress",
]

RELATION_COLUMNS = [
    "relation_id",
    "event_id_a",
    "event_id_b",
    "event_type_a",
    "event_type_b",
    "relation_type",
    "overlap_start_time_s",
    "overlap_end_time_s",
    "overlap_duration_s",
    "coactive_duration_s",
    "a_coverage",
    "b_coverage",
    "gap_s",
]

EVENT_TABLE_SPECS = (
    TableSpec(
        "events/index",
        (
            ColumnSpec(
                "event_id",
                ColumnAvailability.REQUIRED,
                False,
                "Stable identifier for the detected event.",
            ),
            ColumnSpec(
                "session_id",
                ColumnAvailability.OPTIONAL,
                False,
                "Identifier of the session containing the event.",
            ),
            ColumnSpec(
                "lap_id",
                ColumnAvailability.OPTIONAL,
                False,
                "Identifier of the lap containing the event.",
            ),
            ColumnSpec(
                "event_type",
                ColumnAvailability.OPTIONAL,
                False,
                "Detected event category, such as braking, shift, or lockup.",
            ),
            ColumnSpec(
                "start_sample",
                ColumnAvailability.OPTIONAL,
                False,
                "Inclusive source-sample index at the event start.",
            ),
            ColumnSpec(
                "end_sample_exclusive",
                ColumnAvailability.OPTIONAL,
                False,
                "Exclusive source-sample index immediately after the event end.",
            ),
            ColumnSpec(
                "start_time_s",
                ColumnAvailability.OPTIONAL,
                False,
                "Lap-relative time in seconds at the event start.",
            ),
            ColumnSpec(
                "end_time_s",
                ColumnAvailability.OPTIONAL,
                False,
                "Lap-relative time in seconds immediately after the event end.",
            ),
            ColumnSpec(
                "span_duration_s",
                ColumnAvailability.OPTIONAL,
                False,
                "Wall-clock duration in seconds of the full event span.",
            ),
            ColumnSpec(
                "active_duration_s",
                ColumnAvailability.OPTIONAL,
                False,
                "Duration in seconds of samples satisfying the event predicate.",
            ),
            ColumnSpec(
                "start_track_s_m",
                ColumnAvailability.OPTIONAL,
                True,
                "Track distance in metres at the event start.",
            ),
            ColumnSpec(
                "end_track_s_m",
                ColumnAvailability.OPTIONAL,
                True,
                "Track distance in metres at the event end.",
            ),
            ColumnSpec(
                "start_track_progress",
                ColumnAvailability.OPTIONAL,
                True,
                "Normalized track progress at the event start.",
            ),
            ColumnSpec(
                "end_track_progress",
                ColumnAvailability.OPTIONAL,
                True,
                "Normalized track progress at the event end.",
            ),
        ),
        ("event_id",),
        False,
        MergeMode.KEYED,
        (
            ForeignKey(("session_id",), "sessions", ("session_id",)),
            ForeignKey(("lap_id",), "laps", ("lap_id",)),
            ForeignKey(("session_id", "lap_id"), "laps", ("session_id", "lap_id")),
        ),
    ),
    TableSpec(
        "events/braking",
        (
            ColumnSpec(
                "event_id",
                ColumnAvailability.REQUIRED,
                False,
                "Identifier of the corresponding braking event in events/index.",
            ),
            ColumnSpec(
                "entry_speed_kmh",
                ColumnAvailability.OPTIONAL,
                True,
                "Vehicle speed in km/h at brake-event entry.",
            ),
            ColumnSpec(
                "release_speed_kmh",
                ColumnAvailability.OPTIONAL,
                True,
                "Vehicle speed in km/h at brake-event release.",
            ),
            ColumnSpec(
                "minimum_speed_kmh",
                ColumnAvailability.OPTIONAL,
                True,
                "Lowest vehicle speed in km/h during the event.",
            ),
            ColumnSpec(
                "peak_brake",
                ColumnAvailability.OPTIONAL,
                True,
                "Maximum normalized brake input during the event.",
            ),
            ColumnSpec(
                "mean_brake",
                ColumnAvailability.OPTIONAL,
                True,
                "Mean normalized brake input during the event.",
            ),
            ColumnSpec(
                "brake_impulse_proxy_s",
                ColumnAvailability.OPTIONAL,
                False,
                "Time-integral proxy of normalized brake input, in seconds.",
            ),
            ColumnSpec(
                "time_to_90pct_peak_s",
                ColumnAvailability.OPTIONAL,
                True,
                "Seconds from event start until brake input reaches 90% of its peak.",
            ),
            ColumnSpec(
                "release_duration_s",
                ColumnAvailability.OPTIONAL,
                False,
                "Duration in seconds from peak brake input through event release.",
            ),
            ColumnSpec(
                "release_slope_per_s",
                ColumnAvailability.OPTIONAL,
                True,
                "Brake-input slope per second from peak through release.",
            ),
            ColumnSpec(
                "release_monotonicity",
                ColumnAvailability.OPTIONAL,
                False,
                "Fraction of release transitions that do not increase materially.",
            ),
            ColumnSpec(
                "steer_at_start",
                ColumnAvailability.OPTIONAL,
                True,
                "Steering angle at brake-event entry.",
            ),
            ColumnSpec(
                "steer_at_release",
                ColumnAvailability.OPTIONAL,
                True,
                "Steering angle at brake-event release.",
            ),
        ),
        ("event_id",),
        False,
        MergeMode.KEYED,
        (ForeignKey(("event_id",), "events/index", ("event_id",)),),
        empty_frame_columns=("event_id",),
    ),
    TableSpec(
        "events/throttle",
        (
            ColumnSpec(
                "event_id",
                ColumnAvailability.REQUIRED,
                False,
                "Identifier of the corresponding throttle event in events/index.",
            ),
            ColumnSpec(
                "start_speed_kmh",
                ColumnAvailability.OPTIONAL,
                True,
                "Vehicle speed in km/h at throttle-event start.",
            ),
            ColumnSpec(
                "initial_throttle",
                ColumnAvailability.OPTIONAL,
                True,
                "Normalized throttle input at the event start.",
            ),
            ColumnSpec(
                "peak_throttle",
                ColumnAvailability.OPTIONAL,
                False,
                "Maximum normalized throttle input during the event.",
            ),
            ColumnSpec(
                "time_to_50pct_s",
                ColumnAvailability.OPTIONAL,
                True,
                "Seconds from event start until throttle reaches 50%.",
            ),
            ColumnSpec(
                "time_to_full_throttle_s",
                ColumnAvailability.OPTIONAL,
                True,
                "Seconds from event start until throttle reaches the full-throttle threshold.",
            ),
            ColumnSpec(
                "ramp_rate_per_s",
                ColumnAvailability.OPTIONAL,
                False,
                "Peak normalized throttle divided by the pickup duration, per second.",
            ),
            ColumnSpec(
                "steer_at_pickup",
                ColumnAvailability.OPTIONAL,
                True,
                "Steering angle at throttle pickup.",
            ),
            ColumnSpec(
                "steer_at_full_throttle",
                ColumnAvailability.OPTIONAL,
                True,
                "Steering angle when the full-throttle threshold is first reached.",
            ),
            ColumnSpec(
                "velocity_heading_rate_at_pickup_rad_s",
                ColumnAvailability.OPTIONAL,
                True,
                "Velocity-heading angular rate in rad/s at throttle pickup.",
            ),
            ColumnSpec(
                "countersteer_proxy",
                ColumnAvailability.OPTIONAL,
                False,
                "Whether steering angle changes sign within the event.",
            ),
            ColumnSpec(
                "throttle_lift_within_event",
                ColumnAvailability.OPTIONAL,
                False,
                "Whether throttle drops by more than 0.20 within the event.",
            ),
        ),
        ("event_id",),
        False,
        MergeMode.KEYED,
        (ForeignKey(("event_id",), "events/index", ("event_id",)),),
        empty_frame_columns=("event_id",),
    ),
    TableSpec(
        "events/shifts",
        (
            ColumnSpec(
                "event_id",
                ColumnAvailability.REQUIRED,
                False,
                "Identifier of the corresponding shift event in events/index.",
            ),
            ColumnSpec(
                "from_gear",
                ColumnAvailability.OPTIONAL,
                False,
                "Stable physical gear before the shift transition.",
            ),
            ColumnSpec(
                "to_gear",
                ColumnAvailability.OPTIONAL,
                False,
                "Stable physical gear after the shift transition.",
            ),
            ColumnSpec(
                "direction",
                ColumnAvailability.OPTIONAL,
                False,
                "Shift direction: up or down.",
            ),
            ColumnSpec(
                "speed_before_kmh",
                ColumnAvailability.OPTIONAL,
                True,
                "Vehicle speed in km/h immediately before the shift.",
            ),
            ColumnSpec(
                "speed_after_kmh",
                ColumnAvailability.OPTIONAL,
                True,
                "Vehicle speed in km/h when the destination gear is confirmed.",
            ),
            ColumnSpec(
                "rpm_before",
                ColumnAvailability.OPTIONAL,
                True,
                "Engine speed immediately before the shift.",
            ),
            ColumnSpec(
                "rpm_after",
                ColumnAvailability.OPTIONAL,
                True,
                "Engine speed when the destination gear is confirmed.",
            ),
            ColumnSpec(
                "rpm_delta",
                ColumnAvailability.OPTIONAL,
                True,
                "Engine-speed change across the shift.",
            ),
            ColumnSpec(
                "track_long_g_before",
                ColumnAvailability.OPTIONAL,
                True,
                "Longitudinal acceleration in g immediately before the shift.",
            ),
            ColumnSpec(
                "track_long_g_after",
                ColumnAvailability.OPTIONAL,
                True,
                "Longitudinal acceleration in g when the destination gear is confirmed.",
            ),
            ColumnSpec(
                "neutral_duration_s",
                ColumnAvailability.OPTIONAL,
                False,
                "Time in seconds spent in neutral during the shift transition.",
            ),
        ),
        ("event_id",),
        False,
        MergeMode.KEYED,
        (ForeignKey(("event_id",), "events/index", ("event_id",)),),
        empty_frame_columns=("event_id",),
    ),
    TableSpec(
        "events/wheel_slip",
        (
            ColumnSpec(
                "event_id",
                ColumnAvailability.REQUIRED,
                False,
                "Identifier of the corresponding wheel-slip event in events/index.",
            ),
            ColumnSpec(
                "slip_kind",
                ColumnAvailability.OPTIONAL,
                False,
                "Longitudinal slip classification: lockup or wheelspin.",
            ),
            ColumnSpec(
                "wheel",
                ColumnAvailability.OPTIONAL,
                False,
                "Wheel where longitudinal slip was detected: fl, fr, rl, or rr.",
            ),
            ColumnSpec(
                "driven_status",
                ColumnAvailability.OPTIONAL,
                False,
                "Whether the wheel is driven, not driven, or unknown for the vehicle.",
            ),
            ColumnSpec(
                "extreme_slip_ratio",
                ColumnAvailability.OPTIONAL,
                False,
                "Most negative lockup ratio or most positive wheelspin ratio.",
            ),
            ColumnSpec(
                "onset_slip_ratio",
                ColumnAvailability.OPTIONAL,
                False,
                "Slip ratio at the first active sample.",
            ),
            ColumnSpec(
                "recovery_slip_ratio",
                ColumnAvailability.OPTIONAL,
                False,
                "Slip ratio at the last active sample.",
            ),
            ColumnSpec(
                "mean_slip_ratio",
                ColumnAvailability.OPTIONAL,
                False,
                "Mean slip ratio over active samples.",
            ),
            ColumnSpec(
                "slip_integral_s",
                ColumnAvailability.OPTIONAL,
                False,
                "Time integral of absolute slip ratio over active samples, in seconds.",
            ),
            ColumnSpec(
                "entry_speed_kmh",
                ColumnAvailability.OPTIONAL,
                False,
                "Vehicle speed in km/h at the first active sample.",
            ),
            ColumnSpec(
                "exit_speed_kmh",
                ColumnAvailability.OPTIONAL,
                False,
                "Vehicle speed in km/h at the last active sample.",
            ),
            ColumnSpec(
                "mean_brake",
                ColumnAvailability.OPTIONAL,
                True,
                "Mean normalized brake input over active samples.",
            ),
            ColumnSpec(
                "mean_throttle",
                ColumnAvailability.OPTIONAL,
                True,
                "Mean normalized throttle input over active samples.",
            ),
            ColumnSpec(
                "mean_wheel_load",
                ColumnAvailability.OPTIONAL,
                True,
                "Mean load on the affected wheel over active samples.",
            ),
            ColumnSpec(
                "mean_abs_steer",
                ColumnAvailability.OPTIONAL,
                True,
                "Mean absolute steering angle over active samples.",
            ),
        ),
        ("event_id",),
        False,
        MergeMode.KEYED,
        (ForeignKey(("event_id",), "events/index", ("event_id",)),),
        empty_frame_columns=("event_id",),
    ),
    TableSpec(
        "events/relations",
        (
            ColumnSpec(
                "relation_id",
                ColumnAvailability.REQUIRED,
                False,
                "Stable identifier for the relationship between two events.",
            ),
            ColumnSpec(
                "event_id_a",
                ColumnAvailability.REQUIRED,
                False,
                "Identifier of the earlier event in the relation.",
            ),
            ColumnSpec(
                "event_id_b",
                ColumnAvailability.REQUIRED,
                False,
                "Identifier of the later event in the relation.",
            ),
            ColumnSpec(
                "event_type_a",
                ColumnAvailability.OPTIONAL,
                False,
                "Event category of event_id_a.",
            ),
            ColumnSpec(
                "event_type_b",
                ColumnAvailability.OPTIONAL,
                False,
                "Event category of event_id_b.",
            ),
            ColumnSpec(
                "relation_type",
                ColumnAvailability.OPTIONAL,
                False,
                "Relationship classification: overlap or near.",
            ),
            ColumnSpec(
                "overlap_start_time_s",
                ColumnAvailability.OPTIONAL,
                True,
                "Lap-relative overlap start time in seconds, when events overlap.",
            ),
            ColumnSpec(
                "overlap_end_time_s",
                ColumnAvailability.OPTIONAL,
                True,
                "Lap-relative overlap end time in seconds, when events overlap.",
            ),
            ColumnSpec(
                "overlap_duration_s",
                ColumnAvailability.OPTIONAL,
                False,
                "Duration in seconds shared by the two event spans.",
            ),
            ColumnSpec(
                "coactive_duration_s",
                ColumnAvailability.OPTIONAL,
                False,
                "Duration in seconds when both event predicates are active.",
            ),
            ColumnSpec(
                "a_coverage",
                ColumnAvailability.OPTIONAL,
                False,
                "Fraction of event_id_a's span covered by the overlap.",
            ),
            ColumnSpec(
                "b_coverage",
                ColumnAvailability.OPTIONAL,
                False,
                "Fraction of event_id_b's span covered by the overlap.",
            ),
            ColumnSpec(
                "gap_s",
                ColumnAvailability.OPTIONAL,
                False,
                "Non-overlap gap in seconds between the event spans.",
            ),
        ),
        ("relation_id",),
        False,
        MergeMode.KEYED,
        (
            ForeignKey(("event_id_a",), "events/index", ("event_id",)),
            ForeignKey(("event_id_b",), "events/index", ("event_id",)),
        ),
    ),
)

EVENT_TABLE_SPECS_BY_NAME = {table.name: table for table in EVENT_TABLE_SPECS}


def _event_columns(logical_name: str) -> tuple[str, ...]:
    return tuple(
        column.name for column in EVENT_TABLE_SPECS_BY_NAME[logical_name].columns
    )


def _event_empty_columns(logical_name: str) -> tuple[str, ...]:
    columns = EVENT_TABLE_SPECS_BY_NAME[logical_name].empty_frame_columns
    if columns is None:
        raise AssertionError(f"Event table {logical_name!r} has no empty layout")
    return columns


_COMMON_INTERNAL_COLUMNS = {
    "event_id",
    "session_id",
    "lap_id",
    "event_type",
    "start_sample",
    "end_sample",
    "end_sample_exclusive",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "span_duration_s",
    "active_duration_s",
    "start_track_s_m",
    "end_track_s_m",
    "start_track_progress",
    "end_track_progress",
    "_active_sample_durations",
}

_REQUIRED_SAMPLE_COLUMNS = {
    "session_id",
    "lap_id",
    "sample_index",
    "lap_time_s",
    "dt_s",
    "track_s_m",
    "track_progress",
    "speed_kmh",
    "brake_n",
    "throttle",
    "is_braking",
    "gear_physical",
    "rpm",
    "track_long_g",
    "steerAngle",
    "velocity_heading_rate_rad_s",
    *(f"wheel_{wheel}_slip_ratio" for wheel in WHEELS),
    *(f"wheel_{wheel}_load" for wheel in WHEELS),
}


class EventInputError(ValueError):
    """Normalized samples do not satisfy the event detector contract."""


class EventConfigError(ValueError):
    """Event thresholds or interval policies are internally inconsistent."""


def _validate_inputs(samples: pd.DataFrame, config: ProcessingConfig) -> None:
    missing = sorted(_REQUIRED_SAMPLE_COLUMNS - set(samples.columns))
    if missing:
        raise EventInputError(f"Missing event sample columns: {missing}")
    if config.shift_stable_samples < 1:
        raise EventConfigError("shift_stable_samples must be at least 1")
    if config.lockup_slip_ratio_threshold >= 0:
        raise EventConfigError("lockup_slip_ratio_threshold must be negative")
    if config.wheelspin_slip_ratio_threshold <= 0:
        raise EventConfigError("wheelspin_slip_ratio_threshold must be positive")
    durations = (
        config.brake_gap_close_s,
        config.throttle_gap_close_s,
        config.wheel_event_gap_close_s,
        config.minimum_brake_event_s,
        config.minimum_throttle_event_s,
        config.minimum_wheel_event_s,
        config.event_near_shift_s,
    )
    if any(value < 0 for value in durations):
        raise EventConfigError("event gap and duration values must be non-negative")
    if samples.empty:
        return
    key_columns = ["session_id", "lap_id", "sample_index"]
    if samples.duplicated(key_columns).any():
        raise EventInputError(f"Duplicate event sample key in {key_columns}")
    required_values = [
        "session_id",
        "lap_id",
        "sample_index",
        "lap_time_s",
        "dt_s",
    ]
    if samples[required_values].isna().any().any():
        raise EventInputError("Event sample keys and time fields cannot be missing")
    if (samples["dt_s"] <= 0).any():
        raise EventInputError("dt_s must be positive")
    for _, lap in samples.groupby(["session_id", "lap_id"], sort=False):
        ordered = lap.sort_values("sample_index")
        if (ordered["lap_time_s"].diff().dropna() < 0).any():
            raise EventInputError("lap_time_s cannot move backwards within a lap")


@dataclass(frozen=True, slots=True)
class VehicleProfile:
    """Vehicle facts known independently of replay telemetry."""

    driven_wheels: frozenset[Wheel] | None = None

    def __post_init__(self) -> None:
        if self.driven_wheels is not None and not self.driven_wheels <= set(WHEELS):
            raise ValueError(f"Unknown driven wheel in {sorted(self.driven_wheels)!r}")


@dataclass(frozen=True, slots=True)
class EventDataset:
    """Detected events and their temporal relationships."""

    events: pd.DataFrame
    braking: pd.DataFrame
    throttle: pd.DataFrame
    shifts: pd.DataFrame
    wheel_slip: pd.DataFrame
    relations: pd.DataFrame
    abs_activity: pd.DataFrame
    tc_activity: pd.DataFrame

    def to_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "events/index": self.events,
            "events/braking": self.braking,
            "events/throttle": self.throttle,
            "events/shifts": self.shifts,
            "events/wheel_slip": self.wheel_slip,
            "events/relations": self.relations,
            "events/abs_activity": self.abs_activity,
            "events/tc_activity": self.tc_activity,
        }


def _close_short_false_gaps_by_time(
    mask: np.ndarray, dt_s: np.ndarray, maximum_gap_s: float
) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    if maximum_gap_s <= 0 or out.size == 0:
        return out
    for start, end in contiguous_true_runs(~out):
        gap_duration = float(dt_s[start : end + 1].sum())
        if start > 0 and end < len(out) - 1 and gap_duration <= maximum_gap_s:
            out[start : end + 1] = True
    return out


def _event_base(
    g: pd.DataFrame,
    start: int,
    end: int,
    event_type: str,
    active_mask: np.ndarray | None = None,
    identity: object | None = None,
) -> dict[str, Any]:
    segment = g.iloc[start : end + 1]
    if active_mask is None:
        active_mask = np.ones(len(g), dtype=bool)
    active = np.asarray(active_mask[start : end + 1], dtype=bool)
    active_sample_durations = tuple(
        (int(sample), float(duration))
        for sample, duration in zip(
            segment.loc[active, "sample_index"],
            segment.loc[active, "dt_s"],
            strict=True,
        )
    )
    start_sample = int(g["sample_index"].iloc[start])
    end_sample = int(g["sample_index"].iloc[end])
    span_duration = float(segment["dt_s"].sum())
    return {
        "event_id": stable_id(
            g["session_id"].iloc[0],
            g["lap_id"].iloc[0],
            event_type,
            identity,
            start_sample,
            end_sample,
        ),
        "session_id": g["session_id"].iloc[0],
        "lap_id": g["lap_id"].iloc[0],
        "event_type": event_type,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "end_sample_exclusive": end_sample + 1,
        "start_time_s": float(g["lap_time_s"].iloc[start]),
        "end_time_s": float(g["lap_time_s"].iloc[end] + g["dt_s"].iloc[end]),
        # Internal aliases used by the spectral assist detectors.
        "duration_s": span_duration,
        "span_duration_s": span_duration,
        "active_duration_s": float(segment.loc[active, "dt_s"].sum()),
        "start_track_s_m": float(g["track_s_m"].iloc[start]),
        "end_track_s_m": float(g["track_s_m"].iloc[end]),
        "start_track_progress": float(g["track_progress"].iloc[start]),
        "end_track_progress": float(g["track_progress"].iloc[end]),
        "_active_sample_durations": active_sample_durations,
    }


def _lap_groups(samples: pd.DataFrame) -> list[pd.DataFrame]:
    return [
        group.sort_values("sample_index").reset_index(drop=True)
        for _, group in samples.groupby(["session_id", "lap_id"], sort=False)
    ]


def detect_braking(samples: pd.DataFrame, config: ProcessingConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for g in _lap_groups(samples):
        raw_mask = g["is_braking"].fillna(False).to_numpy(dtype=bool)
        mask = _close_short_false_gaps_by_time(
            raw_mask, g["dt_s"].to_numpy(float), config.brake_gap_close_s
        )
        for start, end in contiguous_true_runs(mask):
            segment = g.iloc[start : end + 1]
            active = raw_mask[start : end + 1]
            active_duration = float(segment.loc[active, "dt_s"].sum())
            if active_duration < config.minimum_brake_event_s:
                continue
            if float(g["speed_kmh"].iloc[start]) < config.minimum_brake_entry_speed_kmh:
                continue
            peak = float(segment["brake_n"].max())
            peak_idx = int(segment["brake_n"].idxmax())
            target_indices = segment.index[segment["brake_n"] >= peak * 0.90]
            ramp_to_90 = (
                float(
                    g["lap_time_s"].iloc[int(target_indices[0])]
                    - g["lap_time_s"].iloc[start]
                )
                if len(target_indices)
                else np.nan
            )
            release = segment.loc[peak_idx:]
            release_duration = float(release["dt_s"].sum())
            pressure_diff = release["brake_n"].diff().dropna()
            row = _event_base(g, start, end, "braking", raw_mask)
            row.update(
                {
                    "entry_speed_kmh": float(g["speed_kmh"].iloc[start]),
                    "release_speed_kmh": float(g["speed_kmh"].iloc[end]),
                    "minimum_speed_kmh": float(segment["speed_kmh"].min()),
                    "peak_brake": peak,
                    "mean_brake": float(segment["brake_n"].mean()),
                    "brake_impulse_proxy_s": float(
                        (segment["brake_n"] * segment["dt_s"]).sum()
                    ),
                    "time_to_90pct_peak_s": ramp_to_90,
                    "release_duration_s": release_duration,
                    "release_slope_per_s": (
                        float(
                            (release["brake_n"].iloc[-1] - release["brake_n"].iloc[0])
                            / release_duration
                        )
                        if release_duration > 0
                        else np.nan
                    ),
                    "release_monotonicity": (
                        float((pressure_diff <= 0.01).mean())
                        if len(pressure_diff)
                        else 1.0
                    ),
                    "steer_at_start": float(g["steerAngle"].iloc[start]),
                    "steer_at_release": float(g["steerAngle"].iloc[end]),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def detect_throttle(samples: pd.DataFrame, config: ProcessingConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for g in _lap_groups(samples):
        raw_mask = (
            (g["throttle"] >= config.throttle_event_threshold)
            .fillna(False)
            .to_numpy(dtype=bool)
        )
        mask = _close_short_false_gaps_by_time(
            raw_mask, g["dt_s"].to_numpy(float), config.throttle_gap_close_s
        )
        for start, end in contiguous_true_runs(mask):
            segment = g.iloc[start : end + 1]
            active = raw_mask[start : end + 1]
            active_duration = float(segment.loc[active, "dt_s"].sum())
            if active_duration < config.minimum_throttle_event_s:
                continue
            peak = float(segment["throttle"].max())
            indices_50 = segment.index[segment["throttle"] >= 0.50]
            indices_95 = segment.index[
                segment["throttle"] >= config.full_throttle_threshold
            ]
            start_time = float(g["lap_time_s"].iloc[start])
            time_50 = (
                float(g.loc[int(indices_50[0]), "lap_time_s"] - start_time)
                if len(indices_50)
                else np.nan
            )
            time_95 = (
                float(g.loc[int(indices_95[0]), "lap_time_s"] - start_time)
                if len(indices_95)
                else np.nan
            )
            ramp_rate = peak / max(
                time_95 if np.isfinite(time_95) and time_95 > 0 else active_duration,
                1e-6,
            )
            row = _event_base(g, start, end, "throttle", raw_mask)
            row.update(
                {
                    "start_speed_kmh": float(g["speed_kmh"].iloc[start]),
                    "initial_throttle": float(g["throttle"].iloc[start]),
                    "peak_throttle": peak,
                    "time_to_50pct_s": time_50,
                    "time_to_full_throttle_s": time_95,
                    "ramp_rate_per_s": float(ramp_rate),
                    "steer_at_pickup": float(g["steerAngle"].iloc[start]),
                    "steer_at_full_throttle": (
                        float(g.loc[int(indices_95[0]), "steerAngle"])
                        if len(indices_95)
                        else np.nan
                    ),
                    "velocity_heading_rate_at_pickup_rad_s": float(
                        g["velocity_heading_rate_rad_s"].iloc[start]
                    ),
                    "countersteer_proxy": bool(
                        np.sign(segment["steerAngle"]).nunique(dropna=True) > 1
                    ),
                    "throttle_lift_within_event": bool(
                        (segment["throttle"].diff() < -0.20).any()
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _confirmed_gear(raw_gear: np.ndarray, start: int, samples: int) -> int | None:
    gear = int(raw_gear[start])
    if gear <= 0 or start + samples > len(raw_gear):
        return None
    return gear if np.all(raw_gear[start : start + samples] == gear) else None


def detect_shifts(samples: pd.DataFrame, config: ProcessingConfig) -> pd.DataFrame:
    """Detect the observable transition from one stable physical gear to another."""

    rows: list[dict[str, Any]] = []
    confirmation = config.shift_stable_samples
    for g in _lap_groups(samples):
        raw_gear = g["gear_physical"].fillna(0).astype(int).to_numpy()
        current_gear: int | None = None
        index = 0
        while index < len(g):
            if current_gear is None:
                current_gear = _confirmed_gear(raw_gear, index, confirmation)
                index += confirmation if current_gear is not None else 1
                continue
            if int(raw_gear[index]) == current_gear:
                index += 1
                continue
            transition_start = index
            candidate_index = index
            next_gear: int | None = None
            while candidate_index < len(g):
                next_gear = _confirmed_gear(raw_gear, candidate_index, confirmation)
                if next_gear is not None:
                    break
                candidate_index += 1
            if next_gear is None:
                break
            if next_gear == current_gear:
                index = candidate_index + confirmation
                continue
            before_index = max(0, transition_start - 1)
            row = _event_base(
                g,
                transition_start,
                candidate_index,
                "shift",
                identity=(current_gear, next_gear),
            )
            transition = g.iloc[transition_start : candidate_index + 1]
            neutral = raw_gear[transition_start : candidate_index + 1] <= 0
            row.update(
                {
                    "from_gear": current_gear,
                    "to_gear": next_gear,
                    "direction": "up" if next_gear > current_gear else "down",
                    "speed_before_kmh": float(g["speed_kmh"].iloc[before_index]),
                    "speed_after_kmh": float(g["speed_kmh"].iloc[candidate_index]),
                    "rpm_before": float(g["rpm"].iloc[before_index]),
                    "rpm_after": float(g["rpm"].iloc[candidate_index]),
                    "rpm_delta": float(
                        g["rpm"].iloc[candidate_index] - g["rpm"].iloc[before_index]
                    ),
                    "track_long_g_before": float(g["track_long_g"].iloc[before_index]),
                    "track_long_g_after": float(
                        g["track_long_g"].iloc[candidate_index]
                    ),
                    "neutral_duration_s": float(transition.loc[neutral, "dt_s"].sum()),
                }
            )
            rows.append(row)
            current_gear = next_gear
            index = candidate_index + confirmation
    return pd.DataFrame(rows)


def _driven_status(
    session_id: object, wheel: Wheel, profiles: dict[str, VehicleProfile]
) -> str:
    profile = profiles.get(str(session_id))
    if profile is None or profile.driven_wheels is None:
        return "unknown"
    return "driven" if wheel in profile.driven_wheels else "not_driven"


def detect_wheel_slip(
    samples: pd.DataFrame,
    config: ProcessingConfig,
    vehicle_profiles: dict[str, VehicleProfile] | None = None,
) -> pd.DataFrame:
    """Detect independent per-wheel negative and positive longitudinal slip."""

    profiles = vehicle_profiles or {}
    rows: list[dict[str, Any]] = []
    for g in _lap_groups(samples):
        lockup_speed_mask = (
            g["speed_kmh"] >= config.lockup_minimum_speed_kmh
        ).to_numpy(dtype=bool)
        wheelspin_speed_mask = (
            g["speed_kmh"] >= config.wheelspin_minimum_speed_kmh
        ).to_numpy(dtype=bool)
        dt_s = g["dt_s"].to_numpy(float)
        for wheel_value in WHEELS:
            wheel = cast(Wheel, wheel_value)
            slip_column = f"wheel_{wheel}_slip_ratio"
            slip = g[slip_column].to_numpy(float)
            predicates = (
                (
                    "lockup",
                    lockup_speed_mask & (slip <= config.lockup_slip_ratio_threshold),
                ),
                (
                    "wheelspin",
                    wheelspin_speed_mask
                    & (slip >= config.wheelspin_slip_ratio_threshold),
                ),
            )
            for slip_kind, raw_mask in predicates:
                mask = _close_short_false_gaps_by_time(
                    raw_mask, dt_s, config.wheel_event_gap_close_s
                )
                for start, end in contiguous_true_runs(mask):
                    segment = g.iloc[start : end + 1]
                    active = raw_mask[start : end + 1]
                    active_segment = segment.loc[active]
                    active_duration = float(active_segment["dt_s"].sum())
                    if active_duration < config.minimum_wheel_event_s:
                        continue
                    row = _event_base(
                        g,
                        start,
                        end,
                        slip_kind,
                        raw_mask,
                        identity=wheel,
                    )
                    extreme = (
                        float(active_segment[slip_column].min())
                        if slip_kind == "lockup"
                        else float(active_segment[slip_column].max())
                    )
                    row.update(
                        {
                            "slip_kind": slip_kind,
                            "wheel": wheel,
                            "driven_status": _driven_status(
                                g["session_id"].iloc[0], wheel, profiles
                            ),
                            "extreme_slip_ratio": extreme,
                            "onset_slip_ratio": float(
                                active_segment[slip_column].iloc[0]
                            ),
                            "recovery_slip_ratio": float(
                                active_segment[slip_column].iloc[-1]
                            ),
                            "mean_slip_ratio": float(
                                active_segment[slip_column].mean()
                            ),
                            "slip_integral_s": float(
                                (
                                    active_segment[slip_column].abs()
                                    * active_segment["dt_s"]
                                ).sum()
                            ),
                            "entry_speed_kmh": float(segment["speed_kmh"].iloc[0]),
                            "exit_speed_kmh": float(segment["speed_kmh"].iloc[-1]),
                            "mean_brake": float(active_segment["brake_n"].mean()),
                            "mean_throttle": float(active_segment["throttle"].mean()),
                            "mean_wheel_load": float(
                                active_segment[f"wheel_{wheel}_load"].mean()
                            ),
                            "mean_abs_steer": float(
                                active_segment["steerAngle"].abs().mean()
                            ),
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def _event_index(frames: list[pd.DataFrame]) -> pd.DataFrame:
    available = [frame for frame in frames if not frame.empty]
    if not available:
        return pd.DataFrame(columns=EVENT_INDEX_COLUMNS)
    projected_frames: list[pd.DataFrame] = []
    for frame in available:
        projected = frame.copy()
        if "end_sample_exclusive" not in projected:
            projected["end_sample_exclusive"] = projected["end_sample"] + 1
        if "span_duration_s" not in projected:
            projected["span_duration_s"] = projected["duration_s"]
            projected["end_time_s"] = (
                projected["start_time_s"] + projected["duration_s"]
            )
        if "active_duration_s" not in projected:
            projected["active_duration_s"] = projected["span_duration_s"]
        projected_frames.append(projected[EVENT_INDEX_COLUMNS])
    return (
        pd.concat(projected_frames, ignore_index=True)
        .sort_values(
            [
                "session_id",
                "lap_id",
                "start_sample",
                "end_sample_exclusive",
                "event_type",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _detail(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    empty_columns: tuple[str, ...],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=empty_columns)
    return cast(pd.DataFrame, frame[list(columns)].copy())


def _active_sample_durations(row: pd.Series) -> dict[int, float]:
    return {
        int(sample): float(duration)
        for sample, duration in row["_active_sample_durations"]
    }


def _build_relations(frames: list[pd.DataFrame], near_shift_s: float) -> pd.DataFrame:
    available = [frame for frame in frames if not frame.empty]
    if not available:
        return pd.DataFrame(columns=RELATION_COLUMNS)
    enriched = pd.concat(available, ignore_index=True)
    records: list[dict[str, Any]] = []
    meaningful_pairs = {
        frozenset(("braking", "throttle")),
        frozenset(("braking", "lockup")),
        frozenset(("braking", "wheelspin")),
        frozenset(("throttle", "lockup")),
        frozenset(("throttle", "wheelspin")),
        frozenset(("shift", "braking")),
        frozenset(("shift", "throttle")),
        frozenset(("shift", "lockup")),
        frozenset(("shift", "wheelspin")),
        frozenset(("lockup",)),
        frozenset(("wheelspin",)),
    }
    for _, lap in enriched.groupby(["session_id", "lap_id"], sort=False):
        ordered = lap.sort_values(["start_time_s", "end_time_s", "event_id"])
        rows = [row for _, row in ordered.iterrows()]
        for index, a in enumerate(rows):
            for b in rows[index + 1 :]:
                pair = frozenset((str(a["event_type"]), str(b["event_type"])))
                if pair not in meaningful_pairs:
                    continue
                overlap_start = max(float(a["start_time_s"]), float(b["start_time_s"]))
                overlap_end = min(float(a["end_time_s"]), float(b["end_time_s"]))
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > 0:
                    relation_type = "overlap"
                    gap = 0.0
                else:
                    contains_shift = "shift" in pair
                    contains_wheel_slip = bool(pair & {"lockup", "wheelspin"})
                    gap = max(float(b["start_time_s"]) - float(a["end_time_s"]), 0.0)
                    if not (
                        contains_shift and contains_wheel_slip and gap <= near_shift_s
                    ):
                        continue
                    relation_type = "near"
                    overlap_start = np.nan
                    overlap_end = np.nan
                active_a = _active_sample_durations(a)
                active_b = _active_sample_durations(b)
                coactive = sum(
                    min(active_a[sample], active_b[sample])
                    for sample in active_a.keys() & active_b.keys()
                )
                duration_a = max(float(a["span_duration_s"]), 1e-12)
                duration_b = max(float(b["span_duration_s"]), 1e-12)
                records.append(
                    {
                        "relation_id": stable_id(
                            a["event_id"], b["event_id"], relation_type
                        ),
                        "event_id_a": a["event_id"],
                        "event_id_b": b["event_id"],
                        "event_type_a": a["event_type"],
                        "event_type_b": b["event_type"],
                        "relation_type": relation_type,
                        "overlap_start_time_s": overlap_start,
                        "overlap_end_time_s": overlap_end,
                        "overlap_duration_s": overlap,
                        "coactive_duration_s": coactive,
                        "a_coverage": overlap / duration_a,
                        "b_coverage": overlap / duration_b,
                        "gap_s": gap,
                    }
                )
    return pd.DataFrame(records, columns=RELATION_COLUMNS)


def detect_events(
    samples: pd.DataFrame,
    config: ProcessingConfig | None = None,
    vehicle_profiles: dict[str, VehicleProfile] | None = None,
) -> EventDataset:
    """Detect all driving events through one deterministic in-process seam."""

    config = config or ProcessingConfig()
    _validate_inputs(samples, config)
    braking = detect_braking(samples, config)
    throttle = detect_throttle(samples, config)
    shifts = detect_shifts(samples, config)
    wheel_slip = detect_wheel_slip(samples, config, vehicle_profiles)
    abs_activity = detect_abs_activity(samples, braking, config)
    driven_wheels_by_session = {
        session_id: profile.driven_wheels
        for session_id, profile in (vehicle_profiles or {}).items()
    }
    tc_activity = detect_tc_activity(
        samples, throttle, config, driven_wheels_by_session
    )
    core_frames = [braking, throttle, shifts, wheel_slip]
    return EventDataset(
        events=_event_index(core_frames + [abs_activity, tc_activity]),
        braking=_detail(
            braking,
            _event_columns("events/braking"),
            _event_empty_columns("events/braking"),
        ),
        throttle=_detail(
            throttle,
            _event_columns("events/throttle"),
            _event_empty_columns("events/throttle"),
        ),
        shifts=_detail(
            shifts,
            _event_columns("events/shifts"),
            _event_empty_columns("events/shifts"),
        ),
        wheel_slip=_detail(
            wheel_slip,
            _event_columns("events/wheel_slip"),
            _event_empty_columns("events/wheel_slip"),
        ),
        relations=_build_relations(core_frames, config.event_near_shift_s),
        abs_activity=abs_activity,
        tc_activity=tc_activity,
    )
