import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from ac_replay_parser import ParsedCar, ParsedReplay, parse_replay_data

from .config import ProcessingConfig
from .contract_types import ForeignKey, MergeMode, TableSpec, column_specs
from .track import TrackModel
from .util import parse_datetime_from_ac_filename, sha256_file, stable_id

WHEELS = {"fl": "wheelFL", "fr": "wheelFR", "rl": "wheelRL", "rr": "wheelRR"}

_SAMPLE_COLUMNS = (
    "session_id",
    "sample_index",
    "source_frame",
    "timestamp_s",
    "dt_s",
    "lap_id",
    "lap_segment_index",
    "source_lap_number",
    "lap_time_s",
    "position.x",
    "position.y",
    "position.z",
    "rotation.x",
    "rotation.y",
    "rotation.z",
    "velocity.x",
    "velocity.y",
    "velocity.z",
    "speed_ms",
    "speed_kmh",
    "accel_world_x_ms2",
    "accel_world_y_ms2",
    "accel_world_z_ms2",
    "path_tangent_accel_ms2",
    "path_normal_accel_ms2",
    "path_tangent_g",
    "path_normal_g",
    "velocity_heading_raw_rad",
    "velocity_heading_rate_rad_s",
    "path_distance_2d_m",
    "path_distance_3d_m",
    "throttle_raw",
    "brake_raw",
    "clutch_raw",
    "throttle",
    "brake_n",
    "clutch_n",
    "steerAngle",
    "gear_raw",
    "gear_physical",
    "rpm",
    "drivetrainSpeed",
    "fuel",
    "fuelPerLap",
    "boost",
    "bodyworkNoise",
    "damageFrontDeformation",
    "damageFront",
    "damageRear",
    "damageLeft",
    "damageRight",
    "carDirt",
    "engineHealth",
    "statusRaw",
    "statusLights",
    "statusHorn",
    "statusCameraDirection",
    "statusGearboxBeingDamaged",
    "statusUnknown",
    "statusUnknown2",
    "handbrake",
    "wipers",
    "turnSignals",
    "lowBeams",
    "extraStatusRaw",
    *(f"extraOption{index}" for index in range(10)),
    *(
        item
        for wheel in WHEELS
        for item in (
            f"wheel_{wheel}_angular_velocity",
            f"wheel_{wheel}_slip_angle",
            f"wheel_{wheel}_slip_ratio",
            f"wheel_{wheel}_nd_slip",
            f"wheel_{wheel}_load",
            f"wheel_{wheel}_dirt",
            *(f"wheel_{wheel}_static_position_{axis}" for axis in "xyz"),
            *(f"wheel_{wheel}_static_rotation_{axis}" for axis in "xyz"),
            *(f"wheel_{wheel}_position_{axis}" for axis in "xyz"),
            *(f"wheel_{wheel}_rotation_{axis}" for axis in "xyz"),
        )
    ),
    "front_mean_slip_ratio",
    "rear_mean_slip_ratio",
    "front_slip_ratio_min",
    "rear_slip_ratio_max",
    "front_total_load",
    "rear_total_load",
    "left_total_load",
    "right_total_load",
    "is_moving",
    "is_full_throttle",
    "is_partial_throttle",
    "is_braking",
    "is_coasting",
    "is_brake_throttle_overlap",
    "is_valid_sample",
    "track_reference_index",
    "track_reference_fraction",
    "track_projection_x",
    "track_projection_y",
    "track_projection_z",
    "track_projection_distance_3d_m",
    "track_s_m",
    "track_progress",
    "lateral_offset_m",
    "track_heading_rad",
    "track_curvature_1pm",
    "track_side_left_m",
    "track_side_right_m",
    "distance_to_left_boundary_m",
    "distance_to_right_boundary_m",
    "lateral_position_normalized",
    "is_off_track_candidate",
    "velocity_along_track_ms",
    "velocity_cross_track_ms",
    "vertical_velocity_ms",
    "velocity_heading_rad",
    "velocity_heading_error_rad",
    "accel_along_track_ms2",
    "accel_cross_track_ms2",
    "track_long_g",
    "track_lat_g",
    "track_section_id",
    "track_section_name",
    "drs_detection_zone_id",
    "drs_activation_zone_id",
    "is_in_drs_detection_window",
    "is_in_drs_activation_zone",
    "pit_projection_distance_3d_m",
    "pit_s_m",
    "pit_progress",
    "is_in_pit",
    "track_s_unwrapped_m",
)

_LAP_COLUMNS = (
    "lap_id",
    "session_id",
    "lap_segment_index",
    "source_lap_number",
    "lap_time_s",
    "official_lap_time_s",
    "is_complete",
    "is_valid",
    "is_out_lap",
    "is_in_lap",
    "is_pit_lap",
    "is_replay_fragment",
    "start_sample",
    "end_sample",
    "sample_count",
    "start_fuel",
    "end_fuel",
    "fuel_used",
    "path_distance_2d_m",
    "path_distance_3d_m",
    "reference_length_m",
    "path_excess_vs_ai_line_m",
    "max_speed_kmh",
    "mean_speed_kmh",
    "min_speed_kmh",
    "max_rpm",
    "pit_time_s",
    "off_track_candidate_time_s",
    "median_track_projection_error_m",
    "max_track_projection_error_m",
    "mean_abs_lateral_offset_m",
    "full_throttle_time_s",
    "full_throttle_pct",
    "partial_throttle_time_s",
    "partial_throttle_pct",
    "braking_time_s",
    "braking_pct",
    "coasting_time_s",
    "coasting_pct",
    "overlap_time_s",
    "overlap_pct",
    "front_slip_integral",
    "rear_slip_integral",
    "rear_tire_stress_proxy",
    "lap_class",
    "classification_confidence",
    "classification_reasons",
    "braking_event_count",
    "abs_wheel_episode_count",
    "tc_wheel_episode_count",
    "shift_count",
    "lockup_event_count",
    "wheelspin_event_count",
    "lockup_wheel_time_s",
    "wheelspin_wheel_time_s",
    "abs_active_time_s_union",
    "abs_active_braking_pct",
    "max_abs_activity_score",
    "tc_active_time_s_union",
    "tc_active_throttle_pct",
    "max_tc_activity_score",
)

_SESSION_COLUMNS = (
    "session_id",
    "session_label",
    "source_file",
    "source_name",
    "source_hash",
    "source_format",
    "datetime",
    "car_index",
    "car_id",
    "track_id",
    "track_config",
    "driver_name",
    "nation_code",
    "driver_team",
    "car_skin_id",
    "weather",
    "sample_interval_ms",
    "frame_count",
    "lap_count_total",
    "lap_count_complete",
    "session_duration_s",
    "setup_id",
    "replay_metadata",
    "track_reference_id",
    "driven_wheels",
)

REPLAY_TABLE_SPECS = (
    TableSpec(
        "sessions",
        column_specs(
            _SESSION_COLUMNS,
            required=frozenset({"session_id", "setup_id"}),
            non_nullable=frozenset({"session_id"}),
        ),
        ("session_id",),
        True,
        MergeMode.KEYED,
        ignored_identity_columns=frozenset({"source_file", "source_name"}),
    ),
    TableSpec(
        "laps",
        column_specs(
            _LAP_COLUMNS,
            required=frozenset(
                {
                    "session_id",
                    "lap_id",
                    "is_complete",
                    "is_valid",
                    "lap_time_s",
                    "source_lap_number",
                }
            ),
            non_nullable=frozenset(
                {
                    "session_id",
                    "lap_id",
                    "is_complete",
                    "is_valid",
                    "lap_time_s",
                    "source_lap_number",
                }
            ),
        ),
        ("lap_id",),
        True,
        MergeMode.KEYED,
        (ForeignKey(("session_id",), "sessions", ("session_id",)),),
    ),
    TableSpec(
        "samples",
        column_specs(
            _SAMPLE_COLUMNS,
            required=frozenset({"session_id", "lap_id", "sample_index"}),
            non_nullable=frozenset({"session_id", "lap_id", "sample_index"}),
        ),
        ("session_id", "lap_id", "sample_index"),
        True,
        MergeMode.KEYED,
        (
            ForeignKey(("session_id",), "sessions", ("session_id",)),
            ForeignKey(("lap_id",), "laps", ("lap_id",)),
            ForeignKey(("session_id", "lap_id"), "laps", ("session_id", "lap_id")),
        ),
    ),
    TableSpec(
        "quality/flags",
        column_specs(
            (
                "severity",
                "code",
                "session_id",
                "lap_id",
                "sample_start",
                "sample_end",
                "message",
                "affected_channels",
            ),
            required=frozenset(
                {"session_id", "lap_id", "code", "sample_start", "sample_end"}
            ),
            non_nullable=frozenset(
                {"session_id", "lap_id", "code", "sample_start", "sample_end"}
            ),
        ),
        ("session_id", "lap_id", "code", "sample_start", "sample_end"),
        False,
        MergeMode.KEYED,
        (
            ForeignKey(("session_id",), "sessions", ("session_id",)),
            ForeignKey(("lap_id",), "laps", ("lap_id",)),
            ForeignKey(("session_id", "lap_id"), "laps", ("session_id", "lap_id")),
        ),
    ),
)


@dataclass(slots=True)
class ReplayResult:
    metadata: dict[str, Any]
    samples: pd.DataFrame
    laps: pd.DataFrame
    quality_flags: pd.DataFrame


def _replay_header_metadata(replay: ParsedReplay) -> dict[str, Any]:
    header = replay.header
    return {
        "replayVersion": header.version,
        "recordingInterval": header.recording_interval,
        "weather": header.weather,
        "track": header.track,
        "trackConfig": header.track_config,
        "numCars": header.num_cars,
        "currentRecordingIndex": header.current_recording_index,
        "replayNumFrames": header.num_frames,
        "numTrackObjects": header.num_track_objects,
    }


def inspect_replay(path: Path) -> dict[str, Any]:
    replay = parse_replay_data(path.read_bytes())
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "datetime_from_filename": parse_datetime_from_ac_filename(path),
        "metadata": _replay_header_metadata(replay),
        "driver_names": list(replay.driver_names),
        "car_count": len(replay.cars),
        "cars": [
            {
                "car_index": index,
                "car_id": car.header.car_id,
                "driver_name": car.header.driver_name,
                "nation_code": car.header.nation_code,
                "driver_team": car.header.driver_team,
                "car_skin_id": car.header.car_skin_id,
                "frame_count": len(car.frames),
                "extra_version": car.extra_version,
                "extra_frame_count": len(car.extra_frames),
            }
            for index, car in enumerate(replay.cars)
        ],
        "csp_data_offset": replay.csp_data_offset,
    }


def _car_to_raw(car: ParsedCar) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(car.frames):
        row: dict[str, Any] = {
            "frame": frame_index,
            "position.x": frame.position.x,
            "position.y": frame.position.y,
            "position.z": frame.position.z,
            "rotation.x": frame.rotation.x,
            "rotation.y": frame.rotation.y,
            "rotation.z": frame.rotation.z,
            "velocity.x": frame.velocity.x,
            "velocity.y": frame.velocity.y,
            "velocity.z": frame.velocity.z,
            "steerAngle": frame.steer_angle,
            "bodyworkNoise": frame.bodywork_noise,
            "drivetrainSpeed": frame.drivetrain_speed,
            "currentLap": frame.current_lap,
            "currentLapTime": frame.current_lap_time,
            "lastLapTime": frame.last_lap_time,
            "bestLapTime": frame.best_lap_time,
            "fuel": frame.fuel,
            "fuelPerLap": frame.fuel_per_lap,
            "rpm": frame.rpm,
            "gear": frame.gear,
            "gas": frame.gas,
            "brake": frame.brake,
            "boost": frame.boost,
            "damageFrontDeformation": frame.damage_front_deformation,
            "damageFront": frame.damage_front,
            "damageRear": frame.damage_rear,
            "damageLeft": frame.damage_left,
            "damageRight": frame.damage_right,
            "carDirt": frame.dirt,
            "engineHealth": frame.engine_health,
            "statusRaw": frame.status.raw,
            "statusLights": frame.status.lights,
            "statusHorn": frame.status.horn,
            "statusCameraDirection": frame.status.camera_direction,
            "statusGearboxBeingDamaged": frame.status.gearbox_being_damaged,
            "statusUnknown": frame.unknown,
            "statusUnknown2": frame.unknown2,
        }
        if frame_index < len(car.extra_frames):
            extra = car.extra_frames[frame_index]
            row.update(
                {
                    "clutch": extra.clutch,
                    "handbrake": extra.handbrake,
                    "wipers": extra.wipers,
                    "turnSignals": extra.turn_signals,
                    "lowBeams": extra.low_beams,
                    "extraStatusRaw": extra.raw_status,
                }
            )
            for index, value in enumerate(extra.extra_options):
                row[f"extraOption{index}"] = value
        else:
            row.update(
                {
                    "clutch": np.nan,
                    "handbrake": np.nan,
                    "wipers": np.nan,
                    "turnSignals": np.nan,
                    "lowBeams": pd.NA,
                    "extraStatusRaw": np.nan,
                }
            )
            for index in range(10):
                row[f"extraOption{index}"] = pd.NA

        for short, wheel in zip(WHEELS, frame.wheels, strict=True):
            prefix = WHEELS[short]
            row.update(
                {
                    f"{prefix}.angularVelocity": wheel.angular_velocity,
                    f"{prefix}.slipAngle": wheel.slip_angle,
                    f"{prefix}.slipRatio": wheel.slip_ratio,
                    f"{prefix}.ndSlip": wheel.nd_slip,
                    f"{prefix}.load": wheel.load,
                    f"{prefix}.dirt": wheel.dirt,
                    f"{prefix}.staticPosition.x": wheel.static_position.x,
                    f"{prefix}.staticPosition.y": wheel.static_position.y,
                    f"{prefix}.staticPosition.z": wheel.static_position.z,
                    f"{prefix}.staticRotation.x": wheel.static_rotation.x,
                    f"{prefix}.staticRotation.y": wheel.static_rotation.y,
                    f"{prefix}.staticRotation.z": wheel.static_rotation.z,
                    f"{prefix}.position.x": wheel.position.x,
                    f"{prefix}.position.y": wheel.position.y,
                    f"{prefix}.position.z": wheel.position.z,
                    f"{prefix}.rotation.x": wheel.rotation.x,
                    f"{prefix}.rotation.y": wheel.rotation.y,
                    f"{prefix}.rotation.z": wheel.rotation.z,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [column for column in columns if column in df.columns]
    return cast(pd.DataFrame, df[existing].apply(pd.to_numeric, errors="coerce"))


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
        "frame",
        "position.x",
        "position.y",
        "position.z",
        "rotation.x",
        "rotation.y",
        "rotation.z",
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
        "handbrake",
        "wipers",
        "turnSignals",
        "extraStatusRaw",
        "steerAngle",
        "gear",
        "rpm",
        "fuel",
        "drivetrainSpeed",
        "fuelPerLap",
        "boost",
        "bodyworkNoise",
        "damageFrontDeformation",
        "damageFront",
        "damageRear",
        "damageLeft",
        "damageRight",
        "carDirt",
        "engineHealth",
        "statusRaw",
        "statusCameraDirection",
        "statusUnknown",
        "statusUnknown2",
    ]
    for prefix in WHEELS.values():
        numeric_columns.extend(
            [
                f"{prefix}.angularVelocity",
                f"{prefix}.slipAngle",
                f"{prefix}.slipRatio",
                f"{prefix}.ndSlip",
                f"{prefix}.load",
                f"{prefix}.dirt",
                *(f"{prefix}.staticPosition.{axis}" for axis in "xyz"),
                *(f"{prefix}.staticRotation.{axis}" for axis in "xyz"),
                *(f"{prefix}.position.{axis}" for axis in "xyz"),
                *(f"{prefix}.rotation.{axis}" for axis in "xyz"),
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
    df["lap_id"] = [
        stable_id(session_id, segment) for segment in df["lap_segment_index"]
    ]
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
    clutch_max = float(df["clutch"].max(skipna=True))
    df["clutch_n"] = (
        (
            (df["clutch"] / 255.0)
            if not np.isnan(clutch_max) and clutch_max > 1.5
            else df["clutch"]
        )
        .clip(0, 1)
        .fillna(0)
    )

    velocity = (
        df[["velocity.x", "velocity.y", "velocity.z"]].fillna(0.0).to_numpy(float)
    )
    speed_ms = np.linalg.norm(velocity, axis=1)
    df["speed_ms"] = speed_ms
    df["speed_kmh"] = speed_ms * 3.6

    dt_arr = df["dt_s"].to_numpy(float)
    acceleration = np.zeros_like(velocity)
    if len(velocity) > 1:
        acceleration[1:] = np.diff(velocity, axis=0) / np.maximum(
            dt_arr[1:, None], 1e-6
        )
    df["accel_world_x_ms2"] = acceleration[:, 0]
    df["accel_world_y_ms2"] = acceleration[:, 1]
    df["accel_world_z_ms2"] = acceleration[:, 2]

    # These are path-relative, not chassis-relative. Naming the coordinate system is
    # deliberate: replay rotation semantics are not sufficiently proven to infer body yaw.
    vx = velocity[:, 0]
    vz = velocity[:, 2]
    horizontal_speed = np.hypot(vx, vz)
    ux = np.divide(
        vx, horizontal_speed, out=np.zeros_like(vx), where=horizontal_speed > 1.0
    )
    uz = np.divide(
        vz, horizontal_speed, out=np.zeros_like(vz), where=horizontal_speed > 1.0
    )
    path_tangent_accel = acceleration[:, 0] * ux + acceleration[:, 2] * uz
    path_normal_accel = ux * acceleration[:, 2] - uz * acceleration[:, 0]
    df["path_tangent_accel_ms2"] = path_tangent_accel
    df["path_normal_accel_ms2"] = path_normal_accel
    df["path_tangent_g"] = path_tangent_accel / 9.80665
    df["path_normal_g"] = path_normal_accel / 9.80665
    velocity_heading = np.arctan2(vx, vz)
    df["velocity_heading_raw_rad"] = velocity_heading
    df["velocity_heading_rate_rad_s"] = np.r_[
        0.0,
        np.diff(np.unwrap(velocity_heading)) / np.maximum(dt_arr[1:], 1e-6),
    ]

    df["path_distance_2d_m"] = 0.0
    df["path_distance_3d_m"] = 0.0
    for lap_id, indices in df.groupby("lap_id", sort=False).groups.items():
        g = df.loc[indices]
        delta = (
            g[["position.x", "position.y", "position.z"]]
            .diff()
            .fillna(0.0)
            .to_numpy(float)
        )
        step_2d = np.hypot(delta[:, 0], delta[:, 2])
        step_3d = np.linalg.norm(delta, axis=1)
        jump_mask = step_3d > config.position_jump_threshold_m
        if bool(jump_mask.any()):
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
                        "affected_channels": "position.x,position.y,position.z",
                    }
                )
        step_2d = np.where(jump_mask, 0.0, step_2d)
        step_3d = np.where(jump_mask, 0.0, step_3d)
        df.loc[indices, "path_distance_2d_m"] = np.cumsum(step_2d)
        df.loc[indices, "path_distance_3d_m"] = np.cumsum(step_3d)

    # Gear coding observed in these exports: 1 is neutral, physical gear is raw-1.
    df["gear_raw"] = df["gear"].astype("Int64")
    df["gear_physical"] = (df["gear"] - 1).where(df["gear"] >= 2, 0).astype("Int64")

    df["is_moving"] = df["speed_kmh"] >= config.moving_speed_threshold_kmh
    df["is_full_throttle"] = (df["throttle"] >= config.full_throttle_threshold) & (
        df["brake_n"] < config.pedal_zero_threshold
    )
    df["is_partial_throttle"] = (
        (df["throttle"] >= config.pedal_zero_threshold)
        & (df["throttle"] < config.full_throttle_threshold)
        & (df["brake_n"] < config.pedal_zero_threshold)
    )
    df["is_braking"] = df["brake_n"] >= config.brake_active_threshold
    df["is_coasting"] = (df["throttle"] < config.pedal_zero_threshold) & (
        df["brake_n"] < config.pedal_zero_threshold
    )
    df["is_brake_throttle_overlap"] = (
        df["throttle"] >= config.pedal_zero_threshold
    ) & (df["brake_n"] >= config.brake_active_threshold)

    for short, prefix in WHEELS.items():
        rename = {
            f"{prefix}.angularVelocity": f"wheel_{short}_angular_velocity",
            f"{prefix}.slipAngle": f"wheel_{short}_slip_angle",
            f"{prefix}.slipRatio": f"wheel_{short}_slip_ratio",
            f"{prefix}.ndSlip": f"wheel_{short}_nd_slip",
            f"{prefix}.load": f"wheel_{short}_load",
            f"{prefix}.dirt": f"wheel_{short}_dirt",
        }
        for family in ("staticPosition", "staticRotation", "position", "rotation"):
            snake = {
                "staticPosition": "static_position",
                "staticRotation": "static_rotation",
                "position": "position",
                "rotation": "rotation",
            }[family]
            for axis in "xyz":
                rename[f"{prefix}.{family}.{axis}"] = f"wheel_{short}_{snake}_{axis}"
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
    df["front_total_load"] = df[["wheel_fl_load", "wheel_fr_load"]].sum(
        axis=1, min_count=1
    )
    df["rear_total_load"] = df[["wheel_rl_load", "wheel_rr_load"]].sum(
        axis=1, min_count=1
    )
    df["left_total_load"] = df[["wheel_fl_load", "wheel_rl_load"]].sum(
        axis=1, min_count=1
    )
    df["right_total_load"] = df[["wheel_fr_load", "wheel_rr_load"]].sum(
        axis=1, min_count=1
    )
    df["is_valid_sample"] = df["lap_time_s"].notna() & df["speed_kmh"].notna()

    standard_columns = [
        "session_id",
        "sample_index",
        "source_frame",
        "timestamp_s",
        "dt_s",
        "lap_id",
        "lap_segment_index",
        "source_lap_number",
        "lap_time_s",
        "position.x",
        "position.y",
        "position.z",
        "rotation.x",
        "rotation.y",
        "rotation.z",
        "velocity.x",
        "velocity.y",
        "velocity.z",
        "speed_ms",
        "speed_kmh",
        "accel_world_x_ms2",
        "accel_world_y_ms2",
        "accel_world_z_ms2",
        "path_tangent_accel_ms2",
        "path_normal_accel_ms2",
        "path_tangent_g",
        "path_normal_g",
        "velocity_heading_raw_rad",
        "velocity_heading_rate_rad_s",
        "path_distance_2d_m",
        "path_distance_3d_m",
        "throttle_raw",
        "brake_raw",
        "clutch_raw",
        "throttle",
        "brake_n",
        "clutch_n",
        "steerAngle",
        "gear_raw",
        "gear_physical",
        "rpm",
        "drivetrainSpeed",
        "fuel",
        "fuelPerLap",
        "boost",
        "bodyworkNoise",
        "damageFrontDeformation",
        "damageFront",
        "damageRear",
        "damageLeft",
        "damageRight",
        "carDirt",
        "engineHealth",
        "statusRaw",
        "statusLights",
        "statusHorn",
        "statusCameraDirection",
        "statusGearboxBeingDamaged",
        "statusUnknown",
        "statusUnknown2",
        "handbrake",
        "wipers",
        "turnSignals",
        "lowBeams",
        "extraStatusRaw",
        *(f"extraOption{index}" for index in range(10)),
    ]
    for short in WHEELS:
        standard_columns.extend(
            [
                f"wheel_{short}_angular_velocity",
                f"wheel_{short}_slip_angle",
                f"wheel_{short}_slip_ratio",
                f"wheel_{short}_nd_slip",
                f"wheel_{short}_load",
                f"wheel_{short}_dirt",
                *(f"wheel_{short}_static_position_{axis}" for axis in "xyz"),
                *(f"wheel_{short}_static_rotation_{axis}" for axis in "xyz"),
                *(f"wheel_{short}_position_{axis}" for axis in "xyz"),
                *(f"wheel_{short}_rotation_{axis}" for axis in "xyz"),
            ]
        )
    standard_columns.extend(
        [
            "front_mean_slip_ratio",
            "rear_mean_slip_ratio",
            "front_slip_ratio_min",
            "rear_slip_ratio_max",
            "front_total_load",
            "rear_total_load",
            "left_total_load",
            "right_total_load",
            "is_moving",
            "is_full_throttle",
            "is_partial_throttle",
            "is_braking",
            "is_coasting",
            "is_brake_throttle_overlap",
            "is_valid_sample",
        ]
    )
    available = [column for column in standard_columns if column in df.columns]
    return cast(pd.DataFrame, df[available].copy()), flags


def _build_laps(
    samples: pd.DataFrame,
    raw: pd.DataFrame,
    session_id: str,
    config: ProcessingConfig,
    track_model: TrackModel,
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
        moving_time_s = float(moving["dt_s"].sum())
        lap_time_s = official_lap_time_s or float(g["lap_time_s"].max())
        fuel = g["fuel"]
        start_fuel = float(fuel.iloc[0]) if bool(fuel.notna().any()) else np.nan
        end_fuel = float(fuel.iloc[-1]) if bool(fuel.notna().any()) else np.nan

        pit_mask = g["is_in_pit"].fillna(False).astype(bool)
        pit_time_s = float(g.loc[pit_mask, "dt_s"].sum())
        first_window = g["lap_time_s"] <= min(2.0, max(lap_time_s, 0.0))
        last_window = g["lap_time_s"] >= max(0.0, lap_time_s - 2.0)
        starts_in_pit = bool((pit_mask & first_window).any())
        ends_in_pit = bool((pit_mask & last_window).any())
        is_pit_lap = bool(pit_time_s >= 0.30)
        median_projection_error = float(g["track_projection_distance_3d_m"].median())
        alignment_good = (
            median_projection_error <= config.track_projection_quality_threshold_m
        )
        is_valid = bool(complete and not fragment and not is_pit_lap and alignment_good)

        path_2d = float(g["path_distance_2d_m"].max())
        path_3d = float(g["path_distance_3d_m"].max())
        reference_length = track_model.reference.total_length_m
        row: dict[str, Any] = {
            "lap_id": lap_id,
            "session_id": session_id,
            "lap_segment_index": int(segment_index),
            "source_lap_number": source_lap_number,
            "lap_time_s": lap_time_s,
            "official_lap_time_s": official_lap_time_s,
            "is_complete": complete,
            "is_valid": is_valid,
            "is_out_lap": bool(complete and starts_in_pit),
            "is_in_lap": bool(complete and ends_in_pit),
            "is_pit_lap": is_pit_lap,
            "is_replay_fragment": fragment,
            "start_sample": int(g["sample_index"].iloc[0]),
            "end_sample": int(g["sample_index"].iloc[-1]),
            "sample_count": len(g),
            "start_fuel": start_fuel,
            "end_fuel": end_fuel,
            "fuel_used": start_fuel - end_fuel
            if np.isfinite(start_fuel) and np.isfinite(end_fuel)
            else np.nan,
            "path_distance_2d_m": path_2d,
            "path_distance_3d_m": path_3d,
            "reference_length_m": reference_length,
            "path_excess_vs_ai_line_m": path_3d - reference_length,
            "max_speed_kmh": float(g["speed_kmh"].max()),
            "mean_speed_kmh": float(moving["speed_kmh"].mean())
            if len(moving)
            else np.nan,
            "min_speed_kmh": float(moving["speed_kmh"].min())
            if len(moving)
            else np.nan,
            "max_rpm": float(g["rpm"].max()),
            "pit_time_s": pit_time_s,
            "off_track_candidate_time_s": float(
                g.loc[g["is_off_track_candidate"].fillna(False), "dt_s"].sum()
            ),
            "median_track_projection_error_m": median_projection_error,
            "max_track_projection_error_m": float(
                g["track_projection_distance_3d_m"].max()
            ),
            "mean_abs_lateral_offset_m": float(g["lateral_offset_m"].abs().mean()),
        }
        for state, prefix in [
            ("is_full_throttle", "full_throttle"),
            ("is_partial_throttle", "partial_throttle"),
            ("is_braking", "braking"),
            ("is_coasting", "coasting"),
            ("is_brake_throttle_overlap", "overlap"),
        ]:
            state_time = float(g.loc[g[state].fillna(False), "dt_s"].sum())
            row[f"{prefix}_time_s"] = state_time
            row[f"{prefix}_pct"] = (
                100.0 * state_time / moving_time_s if moving_time_s > 0 else 0.0
            )
        row["front_slip_integral"] = float(
            (g["front_slip_ratio_min"].clip(upper=0).abs() * g["dt_s"]).sum()
        )
        row["rear_slip_integral"] = float(
            (g["rear_slip_ratio_max"].clip(lower=0) * g["dt_s"]).sum()
        )
        row["rear_tire_stress_proxy"] = float(
            (
                g["rear_slip_ratio_max"].clip(lower=0)
                * g["rear_total_load"].fillna(0)
                * g["dt_s"]
            ).sum()
        )

        reasons: list[str] = []
        if fragment:
            lap_class = "fragment"
            reasons.append("replay_started_mid_lap_or_time_nonzero")
        elif not complete:
            lap_class = "incomplete"
            reasons.append("no_confirmed_next_lap_transition")
        elif not alignment_good:
            lap_class = "alignment_error"
            reasons.append("median_track_projection_error_high")
        elif starts_in_pit:
            lap_class = "out_lap"
            reasons.append("pit_lane_detected_near_lap_start")
        elif ends_in_pit:
            lap_class = "in_lap"
            reasons.append("pit_lane_detected_near_lap_end")
        elif is_pit_lap:
            lap_class = "pit_lap"
            reasons.append("pit_lane_detected_during_lap")
        else:
            lap_class = "valid_complete"
            reasons.append("complete_non_pit_lap_with_track_alignment")
        row["lap_class"] = lap_class
        row["classification_confidence"] = 0.95 if complete or fragment else 0.8
        row["classification_reasons"] = ";".join(reasons)
        rows.append(row)
    return pd.DataFrame(rows)


def load_replay(
    path: Path,
    config: ProcessingConfig,
    track_model: TrackModel,
    setup_id: str | None = None,
    session_label: str | None = None,
    driver_name: str | None = None,
) -> list[ReplayResult]:
    replay = parse_replay_data(path.read_bytes())
    if not replay.cars:
        raise ValueError(f"{path.name}: replay contains no cars")
    source_hash = sha256_file(path)
    metadata = _replay_header_metadata(replay)
    results: list[ReplayResult] = []
    selected_cars = [
        (index, car)
        for index, car in enumerate(replay.cars)
        if driver_name is None or car.header.driver_name == driver_name
    ]
    if driver_name is not None and not selected_cars:
        raise ValueError(f'Driver "{driver_name}" was not found in {path.name}')

    for car_index, car in selected_cars:
        car_metadata = {
            **metadata,
            "carID": car.header.car_id,
            "driverName": car.header.driver_name,
            "nationCode": car.header.nation_code,
            "driverTeam": car.header.driver_team,
            "carSkinID": car.header.car_skin_id,
        }
        session_id = stable_id(
            path.name,
            source_hash,
            car_index,
            car.header.car_id,
            car.header.driver_name,
        )
        raw = _car_to_raw(car)
        raw["_lap_segment_index"] = _lap_segments(raw, config.time_reset_tolerance_ms)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            samples, flags = _derive_sample_channels(
                raw, car_metadata, session_id, config
            )
        samples = track_model.align(samples, config)
        laps = _build_laps(samples, raw, session_id, config, track_model)

        label = session_label or path.stem
        if len(replay.cars) > 1:
            label = f"{label} ({car.header.driver_name or car.header.car_id})"
        session_metadata = {
            "session_id": session_id,
            "session_label": label,
            "source_file": str(path),
            "source_name": path.name,
            "source_hash": source_hash,
            "source_format": "acreplay",
            "datetime": parse_datetime_from_ac_filename(path),
            "car_index": car_index,
            "car_id": car.header.car_id,
            "track_id": replay.header.track,
            "track_config": replay.header.track_config,
            "driver_name": car.header.driver_name,
            "nation_code": car.header.nation_code,
            "driver_team": car.header.driver_team,
            "car_skin_id": car.header.car_skin_id,
            "weather": replay.header.weather,
            "sample_interval_ms": replay.header.recording_interval,
            "frame_count": len(samples),
            "lap_count_total": len(laps),
            "lap_count_complete": int(laps["is_complete"].sum()),
            "session_duration_s": float(samples["timestamp_s"].max())
            if len(samples)
            else 0.0,
            "setup_id": setup_id,
            "replay_metadata": car_metadata,
        }
        results.append(
            ReplayResult(session_metadata, samples, laps, pd.DataFrame(flags))
        )
    return results
