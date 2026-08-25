from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .abs_activity import detect_abs_activity
from .config import ProcessingConfig
from .tc_activity import detect_tc_activity
from .util import close_short_false_gaps, contiguous_true_runs, stable_id


def _event_base(g: pd.DataFrame, start: int, end: int, event_type: str) -> dict[str, Any]:
    segment = g.iloc[start : end + 1]
    return {
        "event_id": stable_id(g["lap_id"].iloc[0], event_type, int(g["sample_index"].iloc[start])),
        "session_id": g["session_id"].iloc[0],
        "lap_id": g["lap_id"].iloc[0],
        "event_type": event_type,
        "start_sample": int(g["sample_index"].iloc[start]),
        "end_sample": int(g["sample_index"].iloc[end]),
        "start_time_s": float(g["lap_time_s"].iloc[start]),
        "end_time_s": float(g["lap_time_s"].iloc[end]),
        "duration_s": float(segment["dt_s"].sum()),
        "start_distance_m": float(g["actual_distance_m"].iloc[start]),
        "end_distance_m": float(g["actual_distance_m"].iloc[end]),
        "start_progress": float(g["progress"].iloc[start]),
        "end_progress": float(g["progress"].iloc[end]),
    }


def detect_braking(samples: pd.DataFrame, config: ProcessingConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, original in samples.groupby("lap_id", sort=False):
        g = original.sort_values("sample_index").reset_index(drop=True)
        median_dt = float(g["dt_s"].median()) if len(g) else 0.015
        gap_samples = max(1, round(config.brake_gap_close_s / max(median_dt, 1e-6)))
        mask = close_short_false_gaps(g["is_braking"].to_numpy(), gap_samples)
        for start, end in contiguous_true_runs(mask):
            segment = g.iloc[start : end + 1]
            duration = float(segment["dt_s"].sum())
            if duration < config.minimum_brake_event_s:
                continue
            if float(g["speed_kmh"].iloc[start]) < config.minimum_brake_entry_speed_kmh:
                continue
            peak = float(segment["brake_n"].max())
            peak_idx = int(segment["brake_n"].idxmax())
            target = peak * 0.90
            target_indices = segment.index[segment["brake_n"] >= target]
            ramp_to_90 = (
                float(g["lap_time_s"].iloc[int(target_indices[0])] - g["lap_time_s"].iloc[start])
                if len(target_indices)
                else np.nan
            )
            release = segment.loc[peak_idx:]
            release_duration = float(release["dt_s"].sum())
            release_slope = (
                float((release["brake_n"].iloc[-1] - release["brake_n"].iloc[0]) / release_duration)
                if release_duration > 0
                else np.nan
            )
            pressure_diff = release["brake_n"].diff().dropna()
            monotonicity = float((pressure_diff <= 0.01).mean()) if len(pressure_diff) else 1.0
            row = _event_base(g, start, end, "braking")
            row.update(
                {
                    "entry_speed_kmh": float(g["speed_kmh"].iloc[start]),
                    "release_speed_kmh": float(g["speed_kmh"].iloc[end]),
                    "minimum_speed_kmh": float(segment["speed_kmh"].min()),
                    "peak_brake": peak,
                    "mean_brake": float(segment["brake_n"].mean()),
                    "brake_impulse_proxy_s": float((segment["brake_n"] * segment["dt_s"]).sum()),
                    "time_to_90pct_peak_s": ramp_to_90,
                    "release_duration_s": release_duration,
                    "release_slope_per_s": release_slope,
                    "release_monotonicity": monotonicity,
                    "steer_at_start": float(g["steerAngle"].iloc[start]),
                    "steer_at_release": float(g["steerAngle"].iloc[end]),
                    "front_lock_time_s": float(segment.loc[segment["is_front_lock_candidate"], "dt_s"].sum()),
                    "rear_wheelspin_time_s": float(segment.loc[segment["is_rear_wheelspin_candidate"], "dt_s"].sum()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def detect_throttle(samples: pd.DataFrame, config: ProcessingConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, original in samples.groupby("lap_id", sort=False):
        g = original.sort_values("sample_index").reset_index(drop=True)
        mask = (g["throttle"] >= config.throttle_event_threshold).to_numpy()
        for start, end in contiguous_true_runs(mask):
            segment = g.iloc[start : end + 1]
            duration = float(segment["dt_s"].sum())
            if duration < config.minimum_throttle_event_s:
                continue
            peak = float(segment["throttle"].max())
            indices_50 = segment.index[segment["throttle"] >= 0.50]
            indices_95 = segment.index[segment["throttle"] >= config.full_throttle_threshold]
            start_time = float(g["lap_time_s"].iloc[start])
            time_50 = float(g.loc[int(indices_50[0]), "lap_time_s"] - start_time) if len(indices_50) else np.nan
            time_95 = float(g.loc[int(indices_95[0]), "lap_time_s"] - start_time) if len(indices_95) else np.nan
            ramp_rate = peak / max(time_95 if np.isfinite(time_95) and time_95 > 0 else duration, 1e-6)
            row = _event_base(g, start, end, "throttle_application")
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
                        float(g.loc[int(indices_95[0]), "steerAngle"]) if len(indices_95) else np.nan
                    ),
                    "yaw_rate_at_pickup_rad_s": float(g["yaw_rate_rad_s"].iloc[start]),
                    "rear_slip_at_pickup": float(g["rear_slip_ratio_max"].iloc[start]),
                    "rear_slip_peak": float(segment["rear_slip_ratio_max"].max()),
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


def detect_shifts(samples: pd.DataFrame) -> pd.DataFrame:
    """Combine AC's neutral transition samples into one physical gear-change event."""
    rows: list[dict[str, Any]] = []
    for _, original in samples.groupby("lap_id", sort=False):
        g = original.sort_values("sample_index").reset_index(drop=True)
        raw_gear = g["gear_physical"].fillna(0).astype(int).to_numpy()
        last_stable: int | None = None
        for idx, current in enumerate(raw_gear):
            if current <= 0:
                continue
            if last_stable is None:
                last_stable = int(current)
                continue
            if int(current) == last_stable:
                continue
            before = last_stable
            after = int(current)
            lookback = idx - 1
            while lookback > 0 and raw_gear[lookback] <= 0:
                lookback -= 1
            row = _event_base(g, idx, idx, "gear_shift")
            row.update(
                {
                    "from_gear": before,
                    "to_gear": after,
                    "direction": "up" if after > before else "down",
                    "speed_before_kmh": float(g["speed_kmh"].iloc[lookback]),
                    "speed_after_kmh": float(g["speed_kmh"].iloc[idx]),
                    "rpm_before": float(g["rpm"].iloc[lookback]),
                    "rpm_after": float(g["rpm"].iloc[idx]),
                    "rpm_drop": float(g["rpm"].iloc[lookback] - g["rpm"].iloc[idx]),
                    "long_g_before": float(g["long_g"].iloc[lookback]),
                    "long_g_after": float(g["long_g"].iloc[idx]),
                    "neutral_samples_between": int(np.count_nonzero(raw_gear[lookback + 1 : idx] <= 0)),
                }
            )
            rows.append(row)
            last_stable = after
    return pd.DataFrame(rows)

def _detect_wheel_event(
    samples: pd.DataFrame,
    event_type: str,
    mask_column: str,
    wheels: tuple[str, ...],
    slip_mode: str,
    config: ProcessingConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, original in samples.groupby("lap_id", sort=False):
        g = original.sort_values("sample_index").reset_index(drop=True)
        median_dt = float(g["dt_s"].median()) if len(g) else 0.015
        gap_samples = max(1, round(config.wheel_event_gap_close_s / max(median_dt, 1e-6)))
        mask = close_short_false_gaps(g[mask_column].to_numpy(), gap_samples)
        for start, end in contiguous_true_runs(mask):
            segment = g.iloc[start : end + 1]
            if float(segment["dt_s"].sum()) < config.minimum_wheel_event_s:
                continue
            row = _event_base(g, start, end, event_type)
            slip_columns = [f"wheel_{wheel}_slip_ratio" for wheel in wheels]
            if slip_mode == "min":
                active_wheel = segment[slip_columns].min().idxmin().replace("wheel_", "").replace("_slip_ratio", "")
                extreme = float(segment[slip_columns].min().min())
            else:
                active_wheel = segment[slip_columns].max().idxmax().replace("wheel_", "").replace("_slip_ratio", "")
                extreme = float(segment[slip_columns].max().max())
            row.update(
                {
                    "primary_wheel": active_wheel,
                    "extreme_slip_ratio": extreme,
                    "entry_speed_kmh": float(segment["speed_kmh"].iloc[0]),
                    "mean_brake": float(segment["brake_n"].mean()),
                    "mean_throttle": float(segment["throttle"].mean()),
                    "mean_abs_steer": float(segment["steerAngle"].abs().mean()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def detect_all_events(samples: pd.DataFrame, config: ProcessingConfig) -> dict[str, pd.DataFrame]:
    braking = detect_braking(samples, config)
    throttle = detect_throttle(samples, config)
    return {
        "events/braking": braking,
        "events/abs_activity": detect_abs_activity(samples, braking, config),
        "events/throttle": throttle,
        "events/tc_activity": detect_tc_activity(samples, throttle, config),
        "events/shifts": detect_shifts(samples),
        "events/lockups": _detect_wheel_event(
            samples, "front_lockup_candidate", "is_front_lock_candidate", ("fl", "fr"), "min", config
        ),
        "events/wheelspin": _detect_wheel_event(
            samples, "rear_wheelspin_candidate", "is_rear_wheelspin_candidate", ("rl", "rr"), "max", config
        ),
    }
