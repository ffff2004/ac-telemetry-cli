from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ProcessingConfig
from .util import parse_date_from_ac_filename, sha256_file, stable_id


REQUIRED_COLUMNS = {
    "frame",
    "position.x",
    "position.y",
    "position.z",
    "velocity.x",
    "velocity.y",
    "velocity.z",
    "currentLap",
    "currentLapTime",
    "lastLapTime",
    "bestLapTime",
    "gas",
    "brake",
    "clutch",
    "steerAngle",
    "gear",
    "rpm",
    "fuel",
}

WHEELS = {"fl": "wheelFL", "fr": "wheelFR", "rl": "wheelRL", "rr": "wheelRR"}


@dataclass(slots=True)
class ReplayResult:
    metadata: dict[str, Any]
    samples: pd.DataFrame
    laps: pd.DataFrame
    quality_flags: pd.DataFrame


def read_replay_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            text = line[1:].strip()
            if not text:
                continue
            key, _, raw = text.partition(" ")
            raw = raw.strip()
            if re.fullmatch(r"[-+]?\d+", raw):
                value: Any = int(raw)
            elif re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", raw):
                value = float(raw)
            else:
                value = raw
            metadata[key] = value
    return metadata


def inspect_replay(path: Path) -> dict[str, Any]:
    metadata = read_replay_metadata(path)
    header = pd.read_csv(path, comment="#", nrows=5)
    missing = sorted(REQUIRED_COLUMNS - set(header.columns))
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "date_from_filename": parse_date_from_ac_filename(path),
        "metadata": metadata,
        "column_count": len(header.columns),
        "columns": list(header.columns),
        "missing_required_columns": missing,
    }


def _numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [column for column in columns if column in df.columns]
    return df[existing].apply(pd.to_numeric, errors="coerce")


def _lap_segments(df: pd.DataFrame, tolerance_ms: float) -> pd.Series:
    source_lap = pd.to_numeric(df["currentLap"], errors="coerce").ffill().fillna(-1)
    lap_time = pd.to_numeric(df["currentLapTime"], errors="coerce")
    source_changed = source_lap.ne(source_lap.shift())
    time_reset = lap_time.diff().lt(-abs(tolerance_ms))
    boundary = source_changed | time_reset | df.index.to_series().eq(df.index[0])
    return boundary.cumsum().astype(int) - 1


def _derive_sample_channels(
    raw: pd.DataFrame,
    metadata: dict[str, Any],
    session_id: str,
    config: ProcessingConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    flags: list[dict[str, Any]] = []
    df = raw.copy()
    interval_s = float(metadata.get("recordingInterval", 15.0)) / 1000.0

    numeric_columns = [
        "frame", "position.x", "position.y", "position.z",
        "rotation.x", "rotation.y", "rotation.z",
        "velocity.x", "velocity.y", "velocity.z",
        "currentLap", "currentLapTime", "lastLapTime", "bestLapTime",
        "gas", "brake", "clutch", "steerAngle", "gear", "rpm", "fuel",
        "drivetrainSpeed", "fuelPerLap",
    ]
    for prefix in WHEELS.values():
        numeric_columns.extend(
            [
                f"{prefix}.angularVelocity", f"{prefix}.slipAngle",
                f"{prefix}.slipRatio", f"{prefix}.load",
                f"{prefix}.position.x", f"{prefix}.position.y", f"{prefix}.position.z",
            ]
        )
    converted = _numeric(df, numeric_columns)
    for column in converted.columns:
        df[column] = converted[column]

    df["session_id"] = session_id
    df["sample_index"] = np.arange(len(df), dtype=np.int64)
    df["source_frame"] = df["frame"].astype("Int64")
    df["source_lap_number"] = df["currentLap"].astype("Int64")
    df["lap_segment_index"] = _lap_segments(df, config.time_reset_tolerance_ms)
    df["lap_id"] = [stable_id(session_id, segment) for segment in df["lap_segment_index"]]
    df["lap_time_s"] = df["currentLapTime"] / 1000.0

    source_time = df["frame"] * interval_s
    dt = source_time.diff()
    invalid_dt = dt.le(0) | dt.isna()
    df["timestamp_s"] = source_time - source_time.iloc[0]
    df["dt_s"] = dt.where(~invalid_dt, interval_s)

    df["throttle_raw"] = df["gas"]
    df["brake_raw"] = df["brake"]
    df["clutch_raw"] = df["clutch"]
    df["throttle"] = (df["gas"] / 255.0).clip(0, 1).fillna(0)
    df["brake_n"] = (df["brake"] / 255.0).clip(0, 1).fillna(0)
    # Most replay exports use 0..255 clutch; tolerate an already-normalized channel.
    clutch_max = df["clutch"].max(skipna=True)
    df["clutch_n"] = (
        (df["clutch"] / 255.0) if pd.notna(clutch_max) and clutch_max > 1.5 else df["clutch"]
    ).clip(0, 1).fillna(0)

    velocity = df[["velocity.x", "velocity.y", "velocity.z"]].fillna(0.0).to_numpy(float)
    speed_ms = np.linalg.norm(velocity, axis=1)
    df["speed_ms"] = speed_ms
    df["speed_kmh"] = speed_ms * 3.6

    # Derivatives. The longitudinal axis follows the horizontal velocity vector,
    # which is stable even when replay body rotations are encoded unexpectedly.
    vx = df["velocity.x"].to_numpy(float)
    vz = df["velocity.z"].to_numpy(float)
    dt_arr = df["dt_s"].to_numpy(float)
    ax = np.r_[0.0, np.diff(vx) / np.maximum(dt_arr[1:], 1e-6)]
    az = np.r_[0.0, np.diff(vz) / np.maximum(dt_arr[1:], 1e-6)]
    horiz_speed = np.hypot(vx, vz)
    ux = np.divide(vx, horiz_speed, out=np.zeros_like(vx), where=horiz_speed > 1.0)
    uz = np.divide(vz, horiz_speed, out=np.zeros_like(vz), where=horiz_speed > 1.0)
    long_accel = ax * ux + az * uz
    lat_accel = ux * az - uz * ax
    df["accel_world_x_ms2"] = ax
    df["accel_world_z_ms2"] = az
    df["long_accel_ms2"] = long_accel
    df["lat_accel_ms2"] = lat_accel
    df["long_g"] = long_accel / 9.80665
    df["lat_g"] = lat_accel / 9.80665

    if "rotation.y" in df:
        yaw = np.unwrap(df["rotation.y"].ffill().fillna(0).to_numpy(float))
        yaw_rate = np.r_[0.0, np.diff(yaw) / np.maximum(dt_arr[1:], 1e-6)]
        df["yaw_rad"] = yaw
        df["yaw_rate_rad_s"] = yaw_rate
    else:
        df["yaw_rad"] = np.nan
        df["yaw_rate_rad_s"] = np.nan

    df["actual_distance_m"] = 0.0
    df["normalized_actual_distance"] = np.nan
    for lap_id, indices in df.groupby("lap_id", sort=False).groups.items():
        g = df.loc[indices]
        dx = g["position.x"].diff().fillna(0.0)
        dz = g["position.z"].diff().fillna(0.0)
        step = np.hypot(dx, dz)
        jump_mask = step > config.position_jump_threshold_m
        if jump_mask.any():
            for idx in g.index[jump_mask]:
                flags.append(
                    {
                        "severity": "warning",
                        "code": "POSITION_JUMP",
                        "session_id": session_id,
                        "lap_id": lap_id,
                        "sample_start": int(df.at[idx, "sample_index"]),
                        "sample_end": int(df.at[idx, "sample_index"]),
                        "message": f"Position step exceeded {config.position_jump_threshold_m:g} m",
                        "affected_channels": "position.x,position.z",
                    }
                )
        step = step.where(~jump_mask, 0.0)
        distance = step.cumsum().to_numpy(float)
        df.loc[indices, "actual_distance_m"] = distance
        total = float(distance[-1]) if len(distance) else 0.0
        if total > 0:
            df.loc[indices, "normalized_actual_distance"] = distance / total

    df["progress"] = df["normalized_actual_distance"]
    df["progress_source"] = "cumulative_distance_proxy"
    df["progress_confidence"] = np.where(df["normalized_actual_distance"].notna(), 0.55, 0.0)

    # Gear coding observed in these exports: 1 is neutral, physical gear is raw-1.
    df["gear_raw"] = df["gear"].astype("Int64")
    df["gear_physical"] = (df["gear"] - 1).where(df["gear"] >= 2, 0).astype("Int64")

    df["is_moving"] = df["speed_kmh"] >= config.moving_speed_threshold_kmh
    df["is_full_throttle"] = (
        (df["throttle"] >= config.full_throttle_threshold)
        & (df["brake_n"] < config.pedal_zero_threshold)
    )
    df["is_partial_throttle"] = (
        (df["throttle"] >= config.pedal_zero_threshold)
        & (df["throttle"] < config.full_throttle_threshold)
        & (df["brake_n"] < config.pedal_zero_threshold)
    )
    df["is_braking"] = df["brake_n"] >= config.brake_active_threshold
    df["is_coasting"] = (
        (df["throttle"] < config.pedal_zero_threshold)
        & (df["brake_n"] < config.pedal_zero_threshold)
    )
    df["is_brake_throttle_overlap"] = (
        (df["throttle"] >= config.pedal_zero_threshold)
        & (df["brake_n"] >= config.brake_active_threshold)
    )

    for short, prefix in WHEELS.items():
        rename = {
            f"{prefix}.angularVelocity": f"wheel_{short}_angular_velocity",
            f"{prefix}.slipAngle": f"wheel_{short}_slip_angle",
            f"{prefix}.slipRatio": f"wheel_{short}_slip_ratio",
            f"{prefix}.load": f"wheel_{short}_load",
            f"{prefix}.position.x": f"wheel_{short}_position_x",
            f"{prefix}.position.y": f"wheel_{short}_position_y",
            f"{prefix}.position.z": f"wheel_{short}_position_z",
        }
        for source, target in rename.items():
            df[target] = df[source] if source in df else np.nan

    fl = df["wheel_fl_slip_ratio"]
    fr = df["wheel_fr_slip_ratio"]
    rl = df["wheel_rl_slip_ratio"]
    rr = df["wheel_rr_slip_ratio"]
    df["front_mean_slip_ratio"] = pd.concat([fl, fr], axis=1).mean(axis=1)
    df["rear_mean_slip_ratio"] = pd.concat([rl, rr], axis=1).mean(axis=1)
    df["front_slip_ratio_min"] = pd.concat([fl, fr], axis=1).min(axis=1)
    df["rear_slip_ratio_max"] = pd.concat([rl, rr], axis=1).max(axis=1)
    df["front_total_load"] = df[["wheel_fl_load", "wheel_fr_load"]].sum(axis=1, min_count=1)
    df["rear_total_load"] = df[["wheel_rl_load", "wheel_rr_load"]].sum(axis=1, min_count=1)
    df["left_total_load"] = df[["wheel_fl_load", "wheel_rl_load"]].sum(axis=1, min_count=1)
    df["right_total_load"] = df[["wheel_fr_load", "wheel_rr_load"]].sum(axis=1, min_count=1)

    df["is_front_lock_candidate"] = (
        df["is_braking"]
        & (df["speed_kmh"] >= config.lockup_minimum_speed_kmh)
        & (df["front_slip_ratio_min"] <= config.lockup_slip_ratio_threshold)
    )
    df["is_rear_wheelspin_candidate"] = (
        (df["throttle"] >= config.wheelspin_minimum_throttle)
        & (df["speed_kmh"] >= config.wheelspin_minimum_speed_kmh)
        & (df["rear_slip_ratio_max"] >= config.wheelspin_slip_ratio_threshold)
    )

    # No pit-lane or track-valid channel exists in this replay export. Unknown is not False.
    df["is_in_pit"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df["is_off_track_candidate"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df["is_valid_sample"] = df["lap_time_s"].notna() & df["speed_kmh"].notna()

    standard_columns = [
        "session_id", "sample_index", "source_frame", "timestamp_s", "dt_s",
        "lap_id", "lap_segment_index", "source_lap_number", "lap_time_s",
        "position.x", "position.y", "position.z",
        "actual_distance_m", "normalized_actual_distance", "progress",
        "progress_source", "progress_confidence",
        "velocity.x", "velocity.y", "velocity.z", "speed_ms", "speed_kmh",
        "accel_world_x_ms2", "accel_world_z_ms2", "long_accel_ms2", "lat_accel_ms2",
        "long_g", "lat_g", "yaw_rad", "yaw_rate_rad_s",
        "throttle_raw", "brake_raw", "clutch_raw", "throttle", "brake_n", "clutch_n",
        "steerAngle", "gear_raw", "gear_physical", "rpm", "drivetrainSpeed",
        "fuel", "fuelPerLap",
    ]
    for short in WHEELS:
        standard_columns.extend(
            [
                f"wheel_{short}_angular_velocity", f"wheel_{short}_slip_angle",
                f"wheel_{short}_slip_ratio", f"wheel_{short}_load",
                f"wheel_{short}_position_x", f"wheel_{short}_position_y",
                f"wheel_{short}_position_z",
            ]
        )
    standard_columns.extend(
        [
            "front_mean_slip_ratio", "rear_mean_slip_ratio", "front_slip_ratio_min",
            "rear_slip_ratio_max", "front_total_load", "rear_total_load",
            "left_total_load", "right_total_load",
            "is_moving", "is_full_throttle", "is_partial_throttle", "is_braking",
            "is_coasting", "is_brake_throttle_overlap", "is_front_lock_candidate",
            "is_rear_wheelspin_candidate", "is_in_pit", "is_off_track_candidate",
            "is_valid_sample",
        ]
    )
    available = [column for column in standard_columns if column in df.columns]
    return df[available].copy(), flags


def _build_laps(
    samples: pd.DataFrame,
    raw: pd.DataFrame,
    session_id: str,
    config: ProcessingConfig,
) -> pd.DataFrame:
    next_segment_first: dict[int, pd.Series] = {}
    grouped_raw = raw.groupby("_lap_segment_index", sort=True)
    segment_keys = list(grouped_raw.groups)
    for current, nxt in zip(segment_keys[:-1], segment_keys[1:], strict=False):
        next_segment_first[current] = grouped_raw.get_group(nxt).iloc[0]

    rows: list[dict[str, Any]] = []
    sample_groups = samples.groupby("lap_segment_index", sort=True)
    for segment_index, g in sample_groups:
        g = g.sort_values("sample_index")
        source_lap_number = int(g["source_lap_number"].iloc[0])
        lap_id = str(g["lap_id"].iloc[0])
        next_first = next_segment_first.get(int(segment_index))
        complete = False
        official_lap_time_s: float | None = None
        if next_first is not None:
            next_source_lap = int(next_first["currentLap"])
            last_lap_ms = float(next_first.get("lastLapTime", 0) or 0)
            if next_source_lap == source_lap_number + 1 and last_lap_ms > 0:
                complete = True
                official_lap_time_s = last_lap_ms / 1000.0

        repeated_source_lap = (
            int(segment_index) > 0
            and (samples["lap_segment_index"] == int(segment_index) - 1).any()
            and source_lap_number
            == int(
                samples.loc[
                    samples["lap_segment_index"] == int(segment_index) - 1,
                    "source_lap_number",
                ].iloc[0]
            )
        )
        fragment = bool(not complete and (segment_index == 0 or repeated_source_lap))
        moving = g[g["is_moving"]]
        denominator = max(len(moving), 1)
        lap_time_s = official_lap_time_s or float(g["lap_time_s"].max())
        start_fuel = float(g["fuel"].iloc[0]) if g["fuel"].notna().any() else np.nan
        end_fuel = float(g["fuel"].iloc[-1]) if g["fuel"].notna().any() else np.nan
        row: dict[str, Any] = {
            "lap_id": lap_id,
            "session_id": session_id,
            "lap_segment_index": int(segment_index),
            "source_lap_number": source_lap_number,
            "lap_time_s": lap_time_s,
            "official_lap_time_s": official_lap_time_s,
            "is_complete": complete,
            "is_valid": bool(complete and not fragment),
            "is_out_lap": pd.NA,
            "is_in_lap": pd.NA,
            "is_pit_lap": pd.NA,
            "is_replay_fragment": fragment,
            "start_sample": int(g["sample_index"].iloc[0]),
            "end_sample": int(g["sample_index"].iloc[-1]),
            "sample_count": len(g),
            "start_fuel": start_fuel,
            "end_fuel": end_fuel,
            "fuel_used": start_fuel - end_fuel if np.isfinite(start_fuel) and np.isfinite(end_fuel) else np.nan,
            "actual_distance_m": float(g["actual_distance_m"].max()),
            "max_speed_kmh": float(g["speed_kmh"].max()),
            "mean_speed_kmh": float(moving["speed_kmh"].mean()) if len(moving) else np.nan,
            "min_speed_kmh": float(moving["speed_kmh"].min()) if len(moving) else np.nan,
            "max_rpm": float(g["rpm"].max()),
            "full_throttle_time_s": float(g.loc[g["is_full_throttle"], "dt_s"].sum()),
            "full_throttle_pct": 100.0 * float(moving["is_full_throttle"].sum()) / denominator,
            "partial_throttle_time_s": float(g.loc[g["is_partial_throttle"], "dt_s"].sum()),
            "partial_throttle_pct": 100.0 * float(moving["is_partial_throttle"].sum()) / denominator,
            "braking_time_s": float(g.loc[g["is_braking"], "dt_s"].sum()),
            "braking_pct": 100.0 * float(moving["is_braking"].sum()) / denominator,
            "coasting_time_s": float(g.loc[g["is_coasting"], "dt_s"].sum()),
            "coasting_pct": 100.0 * float(moving["is_coasting"].sum()) / denominator,
            "overlap_time_s": float(g.loc[g["is_brake_throttle_overlap"], "dt_s"].sum()),
            "front_lock_time_s": float(g.loc[g["is_front_lock_candidate"], "dt_s"].sum()),
            "rear_wheelspin_time_s": float(g.loc[g["is_rear_wheelspin_candidate"], "dt_s"].sum()),
            "front_slip_integral": float((g["front_slip_ratio_min"].clip(upper=0).abs() * g["dt_s"]).sum()),
            "rear_slip_integral": float((g["rear_slip_ratio_max"].clip(lower=0) * g["dt_s"]).sum()),
            "rear_tire_stress_proxy": float(
                (
                    g["rear_slip_ratio_max"].clip(lower=0)
                    * g["rear_total_load"].fillna(0)
                    * g["dt_s"]
                ).sum()
            ),
        }
        # Mechanical classification with explicit reasons.
        reasons: list[str] = []
        lap_class = "unknown"
        if fragment:
            lap_class = "fragment"
            reasons.append("replay_started_mid_lap_or_time_nonzero")
        elif not complete:
            lap_class = "incomplete"
            reasons.append("no_confirmed_next_lap_transition")
        elif row["actual_distance_m"] < 0.9 * float(samples.groupby("lap_id")["actual_distance_m"].max().median()):
            lap_class = "pit_or_error"
            reasons.append("path_length_short_relative_to_session")
        else:
            lap_class = "unclassified_complete"
            reasons.append("complete_lap_without_semantic_pit_channel")
        row["lap_class"] = lap_class
        row["classification_confidence"] = 0.95 if lap_class in {"fragment", "incomplete"} else 0.55
        row["classification_reasons"] = ";".join(reasons)
        rows.append(row)
    return pd.DataFrame(rows)


def load_replay(
    path: Path,
    config: ProcessingConfig,
    setup_id: str | None = None,
    session_label: str | None = None,
) -> ReplayResult:
    metadata = read_replay_metadata(path)
    source_hash = sha256_file(path)
    session_id = stable_id(path.name, source_hash)
    raw = pd.read_csv(path, comment="#", low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(raw.columns))
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {missing}")

    raw["_lap_segment_index"] = _lap_segments(raw, config.time_reset_tolerance_ms)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        samples, flags = _derive_sample_channels(raw, metadata, session_id, config)
    laps = _build_laps(samples, raw, session_id, config)

    session_metadata = {
        "session_id": session_id,
        "session_label": session_label or path.stem,
        "source_file": str(path),
        "source_name": path.name,
        "source_hash": source_hash,
        "date": parse_date_from_ac_filename(path),
        "car_id": metadata.get("carID"),
        "track_id": metadata.get("track"),
        "track_config": metadata.get("trackConfig"),
        "driver_name": metadata.get("driverName"),
        "weather": metadata.get("weather"),
        "sample_interval_ms": metadata.get("recordingInterval"),
        "frame_count": len(samples),
        "lap_count_total": len(laps),
        "lap_count_complete": int(laps["is_complete"].sum()),
        "session_duration_s": float(samples["timestamp_s"].max()) if len(samples) else 0.0,
        "setup_id": setup_id,
        "replay_metadata": metadata,
    }
    quality = pd.DataFrame(flags)
    return ReplayResult(session_metadata, samples, laps, quality)
