import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from ac_replay_parser import ParsedCar, ParsedReplay, parse_replay_data

from .config import ProcessingConfig
from .contract_types import (
    ColumnAvailability,
    ColumnSpec,
    ForeignKey,
    MergeMode,
    TableSpec,
)
from .track import TrackModel
from .util import parse_datetime_from_ac_filename, sha256_file, stable_id

WHEELS = {"fl": "wheelFL", "fr": "wheelFR", "rl": "wheelRL", "rr": "wheelRR"}

_REQUIRED = ColumnAvailability.REQUIRED
_OPTIONAL = ColumnAvailability.OPTIONAL

_SESSION_COLUMN_SPECS = (
    ColumnSpec(
        "session_id", _REQUIRED, False, "Stable identifier for this replay car session."
    ),
    ColumnSpec("session_label", _OPTIONAL, True, "Human-readable session label."),
    ColumnSpec(
        "source_file",
        _OPTIONAL,
        True,
        "Original replay path, excluded from identity comparison.",
    ),
    ColumnSpec(
        "source_name",
        _OPTIONAL,
        True,
        "Original replay filename, excluded from identity comparison.",
    ),
    ColumnSpec("source_hash", _OPTIONAL, True, "SHA-256 digest of the replay file."),
    ColumnSpec("source_format", _OPTIONAL, True, "Replay container format identifier."),
    ColumnSpec(
        "datetime",
        _OPTIONAL,
        True,
        "Timestamp parsed from the Assetto Corsa replay filename.",
    ),
    ColumnSpec("car_index", _OPTIONAL, True, "Zero-based car index within the replay."),
    ColumnSpec("car_id", _OPTIONAL, True, "Assetto Corsa car identifier."),
    ColumnSpec(
        "track_id",
        _OPTIONAL,
        True,
        "Assetto Corsa track identifier from the replay header.",
    ),
    ColumnSpec(
        "track_config",
        _OPTIONAL,
        True,
        "Track layout/configuration from the replay header.",
    ),
    ColumnSpec(
        "driver_name", _OPTIONAL, True, "Driver name recorded for the replay car."
    ),
    ColumnSpec(
        "nation_code", _OPTIONAL, True, "Driver nation code recorded by the replay."
    ),
    ColumnSpec("driver_team", _OPTIONAL, True, "Driver team recorded by the replay."),
    ColumnSpec(
        "car_skin_id", _OPTIONAL, True, "Car skin identifier recorded by the replay."
    ),
    ColumnSpec(
        "weather", _OPTIONAL, True, "Weather value recorded by the replay header."
    ),
    ColumnSpec(
        "sample_interval_ms",
        _OPTIONAL,
        True,
        "Nominal replay recording interval in milliseconds.",
    ),
    ColumnSpec(
        "frame_count", _OPTIONAL, True, "Number of persisted telemetry samples."
    ),
    ColumnSpec(
        "lap_count_total",
        _OPTIONAL,
        True,
        "Number of lap segments derived from the replay.",
    ),
    ColumnSpec(
        "lap_count_complete",
        _OPTIONAL,
        True,
        "Number of derived lap segments confirmed complete.",
    ),
    ColumnSpec(
        "session_duration_s",
        _OPTIONAL,
        True,
        "Elapsed duration of persisted samples in seconds.",
    ),
    ColumnSpec(
        "setup_id",
        _REQUIRED,
        True,
        "Stable identifier of the linked setup, when supplied.",
    ),
    ColumnSpec(
        "replay_metadata", _OPTIONAL, True, "Serialized replay and car header metadata."
    ),
    ColumnSpec(
        "track_reference_id",
        _OPTIONAL,
        True,
        "Identifier of the track reference used for alignment.",
    ),
    ColumnSpec(
        "driven_wheels",
        _OPTIONAL,
        True,
        "Configured driven wheel positions for vehicle analysis.",
    ),
)

_LAP_COLUMN_SPECS = (
    ColumnSpec(
        "lap_id", _REQUIRED, False, "Stable identifier for the derived lap segment."
    ),
    ColumnSpec(
        "session_id", _REQUIRED, False, "Identifier of the session containing the lap."
    ),
    ColumnSpec(
        "lap_segment_index",
        _OPTIONAL,
        True,
        "Zero-based segment index after lap/time reset detection.",
    ),
    ColumnSpec(
        "source_lap_number",
        _REQUIRED,
        False,
        "Lap number reported by the replay for this segment.",
    ),
    ColumnSpec(
        "lap_time_s", _REQUIRED, False, "Derived or official lap duration in seconds."
    ),
    ColumnSpec(
        "official_lap_time_s",
        _OPTIONAL,
        True,
        "Official completed-lap time reported by the following replay frame, in seconds.",
    ),
    ColumnSpec(
        "is_complete",
        _REQUIRED,
        False,
        "Whether a subsequent frame confirms the lap completed.",
    ),
    ColumnSpec(
        "is_valid",
        _REQUIRED,
        False,
        "Whether the lap is complete, non-fragmentary, non-pit, and well aligned.",
    ),
    ColumnSpec(
        "is_out_lap",
        _OPTIONAL,
        True,
        "Whether the completed lap begins in the pit lane.",
    ),
    ColumnSpec(
        "is_in_lap", _OPTIONAL, True, "Whether the completed lap ends in the pit lane."
    ),
    ColumnSpec(
        "is_pit_lap",
        _OPTIONAL,
        True,
        "Whether the lap contains at least the configured pit-lane duration.",
    ),
    ColumnSpec(
        "is_replay_fragment",
        _OPTIONAL,
        True,
        "Whether the replay begins mid-lap or repeats a source lap.",
    ),
    ColumnSpec(
        "start_sample", _OPTIONAL, True, "First sample index in the lap segment."
    ),
    ColumnSpec("end_sample", _OPTIONAL, True, "Last sample index in the lap segment."),
    ColumnSpec(
        "sample_count", _OPTIONAL, True, "Number of samples in the lap segment."
    ),
    ColumnSpec("start_fuel", _OPTIONAL, True, "Fuel value at the first lap sample."),
    ColumnSpec("end_fuel", _OPTIONAL, True, "Fuel value at the last lap sample."),
    ColumnSpec("fuel_used", _OPTIONAL, True, "Start fuel minus end fuel for the lap."),
    ColumnSpec(
        "path_distance_2d_m",
        _OPTIONAL,
        True,
        "Accumulated horizontal driven distance in metres.",
    ),
    ColumnSpec(
        "path_distance_3d_m",
        _OPTIONAL,
        True,
        "Accumulated three-dimensional driven distance in metres.",
    ),
    ColumnSpec(
        "reference_length_m",
        _OPTIONAL,
        True,
        "Length of the aligned track reference in metres.",
    ),
    ColumnSpec(
        "path_excess_vs_ai_line_m",
        _OPTIONAL,
        True,
        "Driven 3D distance minus track-reference length in metres.",
    ),
    ColumnSpec(
        "max_speed_kmh", _OPTIONAL, True, "Maximum sample speed in kilometres per hour."
    ),
    ColumnSpec(
        "mean_speed_kmh",
        _OPTIONAL,
        True,
        "Mean speed of moving samples in kilometres per hour.",
    ),
    ColumnSpec(
        "min_speed_kmh",
        _OPTIONAL,
        True,
        "Minimum speed of moving samples in kilometres per hour.",
    ),
    ColumnSpec(
        "max_rpm", _OPTIONAL, True, "Maximum engine speed recorded during the lap."
    ),
    ColumnSpec(
        "pit_time_s", _OPTIONAL, True, "Time projected into the pit lane in seconds."
    ),
    ColumnSpec(
        "off_track_candidate_time_s",
        _OPTIONAL,
        True,
        "Time marked as a possible off-track excursion in seconds.",
    ),
    ColumnSpec(
        "median_track_projection_error_m",
        _OPTIONAL,
        True,
        "Median 3D distance between samples and track projection in metres.",
    ),
    ColumnSpec(
        "max_track_projection_error_m",
        _OPTIONAL,
        True,
        "Maximum 3D distance between samples and track projection in metres.",
    ),
    ColumnSpec(
        "mean_abs_lateral_offset_m",
        _OPTIONAL,
        True,
        "Mean absolute lateral displacement from the track reference in metres.",
    ),
    ColumnSpec(
        "full_throttle_time_s", _OPTIONAL, True, "Time at full throttle in seconds."
    ),
    ColumnSpec(
        "full_throttle_pct",
        _OPTIONAL,
        True,
        "Full-throttle time as a percentage of moving time.",
    ),
    ColumnSpec(
        "partial_throttle_time_s",
        _OPTIONAL,
        True,
        "Time at partial throttle in seconds.",
    ),
    ColumnSpec(
        "partial_throttle_pct",
        _OPTIONAL,
        True,
        "Partial-throttle time as a percentage of moving time.",
    ),
    ColumnSpec(
        "braking_time_s", _OPTIONAL, True, "Time with active braking in seconds."
    ),
    ColumnSpec(
        "braking_pct", _OPTIONAL, True, "Braking time as a percentage of moving time."
    ),
    ColumnSpec("coasting_time_s", _OPTIONAL, True, "Time coasting in seconds."),
    ColumnSpec(
        "coasting_pct", _OPTIONAL, True, "Coasting time as a percentage of moving time."
    ),
    ColumnSpec(
        "overlap_time_s",
        _OPTIONAL,
        True,
        "Time with throttle and brake active together in seconds.",
    ),
    ColumnSpec(
        "overlap_pct",
        _OPTIONAL,
        True,
        "Throttle/brake overlap as a percentage of moving time.",
    ),
    ColumnSpec(
        "front_slip_integral",
        _OPTIONAL,
        True,
        "Time integral of front-wheel braking slip magnitude.",
    ),
    ColumnSpec(
        "rear_slip_integral",
        _OPTIONAL,
        True,
        "Time integral of rear-wheel acceleration slip.",
    ),
    ColumnSpec(
        "rear_tire_stress_proxy",
        _OPTIONAL,
        True,
        "Load-weighted integral of rear acceleration slip.",
    ),
    ColumnSpec(
        "lap_class", _OPTIONAL, True, "Derived classification explaining lap usability."
    ),
    ColumnSpec(
        "classification_confidence",
        _OPTIONAL,
        True,
        "Heuristic confidence in the lap classification.",
    ),
    ColumnSpec(
        "classification_reasons",
        _OPTIONAL,
        True,
        "Semicolon-separated reasons for the lap classification.",
    ),
    ColumnSpec(
        "braking_event_count",
        _OPTIONAL,
        True,
        "Number of detected braking events in the lap.",
    ),
    ColumnSpec(
        "abs_wheel_episode_count",
        _OPTIONAL,
        True,
        "Number of ABS wheel-activity episodes in the lap.",
    ),
    ColumnSpec(
        "tc_wheel_episode_count",
        _OPTIONAL,
        True,
        "Number of traction-control wheel-activity episodes in the lap.",
    ),
    ColumnSpec(
        "shift_count",
        _OPTIONAL,
        True,
        "Number of detected gear-shift events in the lap.",
    ),
    ColumnSpec(
        "lockup_event_count",
        _OPTIONAL,
        True,
        "Number of detected lockup events in the lap.",
    ),
    ColumnSpec(
        "wheelspin_event_count",
        _OPTIONAL,
        True,
        "Number of detected wheelspin events in the lap.",
    ),
    ColumnSpec(
        "lockup_wheel_time_s",
        _OPTIONAL,
        True,
        "Union duration of lockup wheel activity in seconds.",
    ),
    ColumnSpec(
        "wheelspin_wheel_time_s",
        _OPTIONAL,
        True,
        "Union duration of wheelspin wheel activity in seconds.",
    ),
    ColumnSpec(
        "abs_active_time_s_union",
        _OPTIONAL,
        True,
        "Union duration of ABS activity in seconds.",
    ),
    ColumnSpec(
        "abs_active_braking_pct",
        _OPTIONAL,
        True,
        "ABS-active time as a percentage of braking time.",
    ),
    ColumnSpec(
        "max_abs_activity_score",
        _OPTIONAL,
        True,
        "Maximum ABS activity score within the lap.",
    ),
    ColumnSpec(
        "tc_active_time_s_union",
        _OPTIONAL,
        True,
        "Union duration of traction-control activity in seconds.",
    ),
    ColumnSpec(
        "tc_active_throttle_pct",
        _OPTIONAL,
        True,
        "Traction-control-active time as a percentage of throttle time.",
    ),
    ColumnSpec(
        "max_tc_activity_score",
        _OPTIONAL,
        True,
        "Maximum traction-control activity score within the lap.",
    ),
)

_SAMPLE_COLUMN_SPECS = (
    ColumnSpec("session_id", _REQUIRED, False, "Identifier of the source session."),
    ColumnSpec(
        "sample_index", _REQUIRED, False, "Zero-based index of this persisted sample."
    ),
    ColumnSpec(
        "source_frame",
        _OPTIONAL,
        True,
        "Zero-based frame index in the replay car stream.",
    ),
    ColumnSpec("timestamp_s", _OPTIONAL, True, "Elapsed replay time in seconds."),
    ColumnSpec(
        "dt_s",
        _OPTIONAL,
        True,
        "Sample interval used for time integration, in seconds.",
    ),
    ColumnSpec("lap_id", _REQUIRED, False, "Identifier of the derived lap segment."),
    ColumnSpec(
        "lap_segment_index",
        _OPTIONAL,
        True,
        "Zero-based segment index after lap/time reset detection.",
    ),
    ColumnSpec(
        "source_lap_number", _OPTIONAL, True, "Lap number reported by the replay frame."
    ),
    ColumnSpec("lap_time_s", _OPTIONAL, True, "Current replay lap time in seconds."),
    ColumnSpec("position.x", _OPTIONAL, True, "Replay world position X coordinate."),
    ColumnSpec("position.y", _OPTIONAL, True, "Replay world position Y coordinate."),
    ColumnSpec("position.z", _OPTIONAL, True, "Replay world position Z coordinate."),
    ColumnSpec(
        "rotation.x",
        _OPTIONAL,
        True,
        "Replay rotation X component; body yaw is not inferred from it.",
    ),
    ColumnSpec(
        "rotation.y",
        _OPTIONAL,
        True,
        "Replay rotation Y component; body yaw is not inferred from it.",
    ),
    ColumnSpec(
        "rotation.z",
        _OPTIONAL,
        True,
        "Replay rotation Z component; body yaw is not inferred from it.",
    ),
    ColumnSpec(
        "velocity.x",
        _OPTIONAL,
        True,
        "Replay world velocity X component in metres per second.",
    ),
    ColumnSpec(
        "velocity.y",
        _OPTIONAL,
        True,
        "Replay world velocity Y component in metres per second.",
    ),
    ColumnSpec(
        "velocity.z",
        _OPTIONAL,
        True,
        "Replay world velocity Z component in metres per second.",
    ),
    ColumnSpec(
        "speed_ms", _OPTIONAL, True, "Magnitude of world velocity in metres per second."
    ),
    ColumnSpec(
        "speed_kmh",
        _OPTIONAL,
        True,
        "Magnitude of world velocity in kilometres per hour.",
    ),
    ColumnSpec(
        "accel_world_x_ms2",
        _OPTIONAL,
        True,
        "Finite-difference world acceleration X component in metres per second squared.",
    ),
    ColumnSpec(
        "accel_world_y_ms2",
        _OPTIONAL,
        True,
        "Finite-difference world acceleration Y component in metres per second squared.",
    ),
    ColumnSpec(
        "accel_world_z_ms2",
        _OPTIONAL,
        True,
        "Finite-difference world acceleration Z component in metres per second squared.",
    ),
    ColumnSpec(
        "path_tangent_accel_ms2",
        _OPTIONAL,
        True,
        "Acceleration along horizontal velocity direction in metres per second squared.",
    ),
    ColumnSpec(
        "path_normal_accel_ms2",
        _OPTIONAL,
        True,
        "Horizontal acceleration normal to velocity direction in metres per second squared.",
    ),
    ColumnSpec(
        "path_tangent_g",
        _OPTIONAL,
        True,
        "Path-tangent acceleration expressed in standard gravity.",
    ),
    ColumnSpec(
        "path_normal_g",
        _OPTIONAL,
        True,
        "Path-normal acceleration expressed in standard gravity.",
    ),
    ColumnSpec(
        "velocity_heading_raw_rad",
        _OPTIONAL,
        True,
        "Heading of horizontal world velocity in radians.",
    ),
    ColumnSpec(
        "velocity_heading_rate_rad_s",
        _OPTIONAL,
        True,
        "Time derivative of unwrapped velocity heading in radians per second.",
    ),
    ColumnSpec(
        "path_distance_2d_m",
        _OPTIONAL,
        True,
        "Cumulative horizontal distance within the lap segment in metres.",
    ),
    ColumnSpec(
        "path_distance_3d_m",
        _OPTIONAL,
        True,
        "Cumulative three-dimensional distance within the lap segment in metres.",
    ),
    ColumnSpec("throttle_raw", _OPTIONAL, True, "Raw replay throttle channel."),
    ColumnSpec("brake_raw", _OPTIONAL, True, "Raw replay brake channel."),
    ColumnSpec("clutch_raw", _OPTIONAL, True, "Raw CSP clutch channel, if available."),
    ColumnSpec(
        "throttle",
        _OPTIONAL,
        True,
        "Throttle normalized to the range zero through one.",
    ),
    ColumnSpec(
        "brake_n", _OPTIONAL, True, "Brake normalized to the range zero through one."
    ),
    ColumnSpec(
        "clutch_n", _OPTIONAL, True, "Clutch normalized to the range zero through one."
    ),
    ColumnSpec("steerAngle", _OPTIONAL, True, "Replay steering-angle channel."),
    ColumnSpec("gear_raw", _OPTIONAL, True, "Raw replay gear code."),
    ColumnSpec(
        "gear_physical",
        _OPTIONAL,
        True,
        "Physical forward gear inferred from the replay gear code.",
    ),
    ColumnSpec("rpm", _OPTIONAL, True, "Engine speed reported by the replay."),
    ColumnSpec("drivetrainSpeed", _OPTIONAL, True, "Replay drivetrain-speed channel."),
    ColumnSpec("fuel", _OPTIONAL, True, "Fuel value reported by the replay."),
    ColumnSpec("fuelPerLap", _OPTIONAL, True, "Replay fuel-per-lap estimate."),
    ColumnSpec("boost", _OPTIONAL, True, "Replay boost channel."),
    ColumnSpec("bodyworkNoise", _OPTIONAL, True, "Replay bodywork-noise channel."),
    ColumnSpec(
        "damageFrontDeformation",
        _OPTIONAL,
        True,
        "Replay front-deformation damage channel.",
    ),
    ColumnSpec("damageFront", _OPTIONAL, True, "Replay front damage channel."),
    ColumnSpec("damageRear", _OPTIONAL, True, "Replay rear damage channel."),
    ColumnSpec("damageLeft", _OPTIONAL, True, "Replay left-side damage channel."),
    ColumnSpec("damageRight", _OPTIONAL, True, "Replay right-side damage channel."),
    ColumnSpec("carDirt", _OPTIONAL, True, "Replay car-dirt channel."),
    ColumnSpec("engineHealth", _OPTIONAL, True, "Replay engine-health channel."),
    ColumnSpec("statusRaw", _OPTIONAL, True, "Raw replay status bit field."),
    ColumnSpec(
        "statusLights",
        _OPTIONAL,
        True,
        "Whether replay status reports vehicle lights on.",
    ),
    ColumnSpec(
        "statusHorn", _OPTIONAL, True, "Whether replay status reports horn active."
    ),
    ColumnSpec(
        "statusCameraDirection",
        _OPTIONAL,
        True,
        "Camera direction decoded from the replay status.",
    ),
    ColumnSpec(
        "statusGearboxBeingDamaged",
        _OPTIONAL,
        True,
        "Whether replay status reports gearbox damage.",
    ),
    ColumnSpec("statusUnknown", _OPTIONAL, True, "Undocumented replay status channel."),
    ColumnSpec(
        "statusUnknown2", _OPTIONAL, True, "Second undocumented replay status channel."
    ),
    ColumnSpec("handbrake", _OPTIONAL, True, "CSP handbrake channel, if available."),
    ColumnSpec("wipers", _OPTIONAL, True, "CSP wiper-state channel, if available."),
    ColumnSpec(
        "turnSignals", _OPTIONAL, True, "CSP turn-signal channel, if available."
    ),
    ColumnSpec("lowBeams", _OPTIONAL, True, "CSP low-beam state, if available."),
    ColumnSpec(
        "extraStatusRaw",
        _OPTIONAL,
        True,
        "Raw CSP extra status bit field, if available.",
    ),
    ColumnSpec(
        "extraOption0",
        _OPTIONAL,
        True,
        "First undocumented CSP extra option, if available.",
    ),
    ColumnSpec(
        "extraOption1",
        _OPTIONAL,
        True,
        "Second undocumented CSP extra option, if available.",
    ),
    ColumnSpec(
        "extraOption2",
        _OPTIONAL,
        True,
        "Third undocumented CSP extra option, if available.",
    ),
    ColumnSpec(
        "extraOption3",
        _OPTIONAL,
        True,
        "Fourth undocumented CSP extra option, if available.",
    ),
    ColumnSpec(
        "extraOption4",
        _OPTIONAL,
        True,
        "Fifth undocumented CSP extra option, if available.",
    ),
    ColumnSpec(
        "extraOption5",
        _OPTIONAL,
        True,
        "Sixth undocumented CSP extra option, if available.",
    ),
    ColumnSpec(
        "extraOption6",
        _OPTIONAL,
        True,
        "Seventh undocumented CSP extra option, if available.",
    ),
    ColumnSpec(
        "extraOption7",
        _OPTIONAL,
        True,
        "Eighth undocumented CSP extra option, if available.",
    ),
    ColumnSpec(
        "extraOption8",
        _OPTIONAL,
        True,
        "Ninth undocumented CSP extra option, if available.",
    ),
    ColumnSpec(
        "extraOption9",
        _OPTIONAL,
        True,
        "Tenth undocumented CSP extra option, if available.",
    ),
    ColumnSpec(
        "wheel_fl_angular_velocity",
        _OPTIONAL,
        True,
        "Front-left wheel angular velocity.",
    ),
    ColumnSpec("wheel_fl_slip_angle", _OPTIONAL, True, "Front-left wheel slip angle."),
    ColumnSpec("wheel_fl_slip_ratio", _OPTIONAL, True, "Front-left wheel slip ratio."),
    ColumnSpec(
        "wheel_fl_nd_slip", _OPTIONAL, True, "Front-left wheel normalized slip channel."
    ),
    ColumnSpec("wheel_fl_load", _OPTIONAL, True, "Front-left wheel load."),
    ColumnSpec("wheel_fl_dirt", _OPTIONAL, True, "Front-left wheel dirt channel."),
    ColumnSpec(
        "wheel_fl_static_position_x",
        _OPTIONAL,
        True,
        "Front-left wheel static-position X component.",
    ),
    ColumnSpec(
        "wheel_fl_static_position_y",
        _OPTIONAL,
        True,
        "Front-left wheel static-position Y component.",
    ),
    ColumnSpec(
        "wheel_fl_static_position_z",
        _OPTIONAL,
        True,
        "Front-left wheel static-position Z component.",
    ),
    ColumnSpec(
        "wheel_fl_static_rotation_x",
        _OPTIONAL,
        True,
        "Front-left wheel static-rotation X component.",
    ),
    ColumnSpec(
        "wheel_fl_static_rotation_y",
        _OPTIONAL,
        True,
        "Front-left wheel static-rotation Y component.",
    ),
    ColumnSpec(
        "wheel_fl_static_rotation_z",
        _OPTIONAL,
        True,
        "Front-left wheel static-rotation Z component.",
    ),
    ColumnSpec(
        "wheel_fl_position_x", _OPTIONAL, True, "Front-left wheel position X component."
    ),
    ColumnSpec(
        "wheel_fl_position_y", _OPTIONAL, True, "Front-left wheel position Y component."
    ),
    ColumnSpec(
        "wheel_fl_position_z", _OPTIONAL, True, "Front-left wheel position Z component."
    ),
    ColumnSpec(
        "wheel_fl_rotation_x", _OPTIONAL, True, "Front-left wheel rotation X component."
    ),
    ColumnSpec(
        "wheel_fl_rotation_y", _OPTIONAL, True, "Front-left wheel rotation Y component."
    ),
    ColumnSpec(
        "wheel_fl_rotation_z", _OPTIONAL, True, "Front-left wheel rotation Z component."
    ),
    ColumnSpec(
        "wheel_fr_angular_velocity",
        _OPTIONAL,
        True,
        "Front-right wheel angular velocity.",
    ),
    ColumnSpec("wheel_fr_slip_angle", _OPTIONAL, True, "Front-right wheel slip angle."),
    ColumnSpec("wheel_fr_slip_ratio", _OPTIONAL, True, "Front-right wheel slip ratio."),
    ColumnSpec(
        "wheel_fr_nd_slip",
        _OPTIONAL,
        True,
        "Front-right wheel normalized slip channel.",
    ),
    ColumnSpec("wheel_fr_load", _OPTIONAL, True, "Front-right wheel load."),
    ColumnSpec("wheel_fr_dirt", _OPTIONAL, True, "Front-right wheel dirt channel."),
    ColumnSpec(
        "wheel_fr_static_position_x",
        _OPTIONAL,
        True,
        "Front-right wheel static-position X component.",
    ),
    ColumnSpec(
        "wheel_fr_static_position_y",
        _OPTIONAL,
        True,
        "Front-right wheel static-position Y component.",
    ),
    ColumnSpec(
        "wheel_fr_static_position_z",
        _OPTIONAL,
        True,
        "Front-right wheel static-position Z component.",
    ),
    ColumnSpec(
        "wheel_fr_static_rotation_x",
        _OPTIONAL,
        True,
        "Front-right wheel static-rotation X component.",
    ),
    ColumnSpec(
        "wheel_fr_static_rotation_y",
        _OPTIONAL,
        True,
        "Front-right wheel static-rotation Y component.",
    ),
    ColumnSpec(
        "wheel_fr_static_rotation_z",
        _OPTIONAL,
        True,
        "Front-right wheel static-rotation Z component.",
    ),
    ColumnSpec(
        "wheel_fr_position_x",
        _OPTIONAL,
        True,
        "Front-right wheel position X component.",
    ),
    ColumnSpec(
        "wheel_fr_position_y",
        _OPTIONAL,
        True,
        "Front-right wheel position Y component.",
    ),
    ColumnSpec(
        "wheel_fr_position_z",
        _OPTIONAL,
        True,
        "Front-right wheel position Z component.",
    ),
    ColumnSpec(
        "wheel_fr_rotation_x",
        _OPTIONAL,
        True,
        "Front-right wheel rotation X component.",
    ),
    ColumnSpec(
        "wheel_fr_rotation_y",
        _OPTIONAL,
        True,
        "Front-right wheel rotation Y component.",
    ),
    ColumnSpec(
        "wheel_fr_rotation_z",
        _OPTIONAL,
        True,
        "Front-right wheel rotation Z component.",
    ),
    ColumnSpec(
        "wheel_rl_angular_velocity",
        _OPTIONAL,
        True,
        "Rear-left wheel angular velocity.",
    ),
    ColumnSpec("wheel_rl_slip_angle", _OPTIONAL, True, "Rear-left wheel slip angle."),
    ColumnSpec("wheel_rl_slip_ratio", _OPTIONAL, True, "Rear-left wheel slip ratio."),
    ColumnSpec(
        "wheel_rl_nd_slip", _OPTIONAL, True, "Rear-left wheel normalized slip channel."
    ),
    ColumnSpec("wheel_rl_load", _OPTIONAL, True, "Rear-left wheel load."),
    ColumnSpec("wheel_rl_dirt", _OPTIONAL, True, "Rear-left wheel dirt channel."),
    ColumnSpec(
        "wheel_rl_static_position_x",
        _OPTIONAL,
        True,
        "Rear-left wheel static-position X component.",
    ),
    ColumnSpec(
        "wheel_rl_static_position_y",
        _OPTIONAL,
        True,
        "Rear-left wheel static-position Y component.",
    ),
    ColumnSpec(
        "wheel_rl_static_position_z",
        _OPTIONAL,
        True,
        "Rear-left wheel static-position Z component.",
    ),
    ColumnSpec(
        "wheel_rl_static_rotation_x",
        _OPTIONAL,
        True,
        "Rear-left wheel static-rotation X component.",
    ),
    ColumnSpec(
        "wheel_rl_static_rotation_y",
        _OPTIONAL,
        True,
        "Rear-left wheel static-rotation Y component.",
    ),
    ColumnSpec(
        "wheel_rl_static_rotation_z",
        _OPTIONAL,
        True,
        "Rear-left wheel static-rotation Z component.",
    ),
    ColumnSpec(
        "wheel_rl_position_x", _OPTIONAL, True, "Rear-left wheel position X component."
    ),
    ColumnSpec(
        "wheel_rl_position_y", _OPTIONAL, True, "Rear-left wheel position Y component."
    ),
    ColumnSpec(
        "wheel_rl_position_z", _OPTIONAL, True, "Rear-left wheel position Z component."
    ),
    ColumnSpec(
        "wheel_rl_rotation_x", _OPTIONAL, True, "Rear-left wheel rotation X component."
    ),
    ColumnSpec(
        "wheel_rl_rotation_y", _OPTIONAL, True, "Rear-left wheel rotation Y component."
    ),
    ColumnSpec(
        "wheel_rl_rotation_z", _OPTIONAL, True, "Rear-left wheel rotation Z component."
    ),
    ColumnSpec(
        "wheel_rr_angular_velocity",
        _OPTIONAL,
        True,
        "Rear-right wheel angular velocity.",
    ),
    ColumnSpec("wheel_rr_slip_angle", _OPTIONAL, True, "Rear-right wheel slip angle."),
    ColumnSpec("wheel_rr_slip_ratio", _OPTIONAL, True, "Rear-right wheel slip ratio."),
    ColumnSpec(
        "wheel_rr_nd_slip", _OPTIONAL, True, "Rear-right wheel normalized slip channel."
    ),
    ColumnSpec("wheel_rr_load", _OPTIONAL, True, "Rear-right wheel load."),
    ColumnSpec("wheel_rr_dirt", _OPTIONAL, True, "Rear-right wheel dirt channel."),
    ColumnSpec(
        "wheel_rr_static_position_x",
        _OPTIONAL,
        True,
        "Rear-right wheel static-position X component.",
    ),
    ColumnSpec(
        "wheel_rr_static_position_y",
        _OPTIONAL,
        True,
        "Rear-right wheel static-position Y component.",
    ),
    ColumnSpec(
        "wheel_rr_static_position_z",
        _OPTIONAL,
        True,
        "Rear-right wheel static-position Z component.",
    ),
    ColumnSpec(
        "wheel_rr_static_rotation_x",
        _OPTIONAL,
        True,
        "Rear-right wheel static-rotation X component.",
    ),
    ColumnSpec(
        "wheel_rr_static_rotation_y",
        _OPTIONAL,
        True,
        "Rear-right wheel static-rotation Y component.",
    ),
    ColumnSpec(
        "wheel_rr_static_rotation_z",
        _OPTIONAL,
        True,
        "Rear-right wheel static-rotation Z component.",
    ),
    ColumnSpec(
        "wheel_rr_position_x", _OPTIONAL, True, "Rear-right wheel position X component."
    ),
    ColumnSpec(
        "wheel_rr_position_y", _OPTIONAL, True, "Rear-right wheel position Y component."
    ),
    ColumnSpec(
        "wheel_rr_position_z", _OPTIONAL, True, "Rear-right wheel position Z component."
    ),
    ColumnSpec(
        "wheel_rr_rotation_x", _OPTIONAL, True, "Rear-right wheel rotation X component."
    ),
    ColumnSpec(
        "wheel_rr_rotation_y", _OPTIONAL, True, "Rear-right wheel rotation Y component."
    ),
    ColumnSpec(
        "wheel_rr_rotation_z", _OPTIONAL, True, "Rear-right wheel rotation Z component."
    ),
    ColumnSpec(
        "front_mean_slip_ratio", _OPTIONAL, True, "Mean front-wheel slip ratio."
    ),
    ColumnSpec("rear_mean_slip_ratio", _OPTIONAL, True, "Mean rear-wheel slip ratio."),
    ColumnSpec(
        "front_slip_ratio_min", _OPTIONAL, True, "Minimum front-wheel slip ratio."
    ),
    ColumnSpec(
        "rear_slip_ratio_max", _OPTIONAL, True, "Maximum rear-wheel slip ratio."
    ),
    ColumnSpec("front_total_load", _OPTIONAL, True, "Sum of front-wheel loads."),
    ColumnSpec("rear_total_load", _OPTIONAL, True, "Sum of rear-wheel loads."),
    ColumnSpec("left_total_load", _OPTIONAL, True, "Sum of left-wheel loads."),
    ColumnSpec("right_total_load", _OPTIONAL, True, "Sum of right-wheel loads."),
    ColumnSpec(
        "is_moving",
        _OPTIONAL,
        True,
        "Whether speed meets the configured moving threshold.",
    ),
    ColumnSpec(
        "is_full_throttle",
        _OPTIONAL,
        True,
        "Whether throttle is full while brake is inactive.",
    ),
    ColumnSpec(
        "is_partial_throttle",
        _OPTIONAL,
        True,
        "Whether throttle is partial while brake is inactive.",
    ),
    ColumnSpec(
        "is_braking",
        _OPTIONAL,
        True,
        "Whether normalized brake meets the active threshold.",
    ),
    ColumnSpec(
        "is_coasting", _OPTIONAL, True, "Whether throttle and brake are both inactive."
    ),
    ColumnSpec(
        "is_brake_throttle_overlap",
        _OPTIONAL,
        True,
        "Whether throttle and brake are active together.",
    ),
    ColumnSpec(
        "is_valid_sample",
        _OPTIONAL,
        True,
        "Whether lap time and speed are available for the sample.",
    ),
    ColumnSpec(
        "track_reference_index",
        _OPTIONAL,
        True,
        "Index of the nearest track-reference sample.",
    ),
    ColumnSpec(
        "track_reference_fraction",
        _OPTIONAL,
        True,
        "Fraction along the selected track-reference segment.",
    ),
    ColumnSpec(
        "track_projection_x",
        _OPTIONAL,
        True,
        "World X coordinate of the aligned track projection.",
    ),
    ColumnSpec(
        "track_projection_y",
        _OPTIONAL,
        True,
        "World Y coordinate of the aligned track projection.",
    ),
    ColumnSpec(
        "track_projection_z",
        _OPTIONAL,
        True,
        "World Z coordinate of the aligned track projection.",
    ),
    ColumnSpec(
        "track_projection_distance_3d_m",
        _OPTIONAL,
        True,
        "3D distance from sample to track projection in metres.",
    ),
    ColumnSpec(
        "track_s_m",
        _OPTIONAL,
        True,
        "Wrapped distance along the track reference in metres.",
    ),
    ColumnSpec(
        "track_progress",
        _OPTIONAL,
        True,
        "Wrapped fractional progress along the track reference.",
    ),
    ColumnSpec(
        "lateral_offset_m",
        _OPTIONAL,
        True,
        "Signed lateral offset from the track reference in metres.",
    ),
    ColumnSpec(
        "track_heading_rad",
        _OPTIONAL,
        True,
        "Track-reference heading at the projection in radians.",
    ),
    ColumnSpec(
        "track_curvature_1pm",
        _OPTIONAL,
        True,
        "Track-reference curvature at the projection in reciprocal metres.",
    ),
    ColumnSpec(
        "track_side_left_m",
        _OPTIONAL,
        True,
        "Available track-reference distance to the left boundary in metres.",
    ),
    ColumnSpec(
        "track_side_right_m",
        _OPTIONAL,
        True,
        "Available track-reference distance to the right boundary in metres.",
    ),
    ColumnSpec(
        "distance_to_left_boundary_m",
        _OPTIONAL,
        True,
        "Distance from sample to the left track boundary in metres.",
    ),
    ColumnSpec(
        "distance_to_right_boundary_m",
        _OPTIONAL,
        True,
        "Distance from sample to the right track boundary in metres.",
    ),
    ColumnSpec(
        "lateral_position_normalized",
        _OPTIONAL,
        True,
        "Lateral position normalized by available track width.",
    ),
    ColumnSpec(
        "is_off_track_candidate",
        _OPTIONAL,
        True,
        "Whether the sample is a candidate off-track position.",
    ),
    ColumnSpec(
        "velocity_along_track_ms",
        _OPTIONAL,
        True,
        "Velocity along the aligned track direction in metres per second.",
    ),
    ColumnSpec(
        "velocity_cross_track_ms",
        _OPTIONAL,
        True,
        "Velocity across the aligned track direction in metres per second.",
    ),
    ColumnSpec(
        "vertical_velocity_ms",
        _OPTIONAL,
        True,
        "World vertical velocity in metres per second.",
    ),
    ColumnSpec(
        "velocity_heading_rad",
        _OPTIONAL,
        True,
        "Velocity heading after track alignment in radians.",
    ),
    ColumnSpec(
        "velocity_heading_error_rad",
        _OPTIONAL,
        True,
        "Velocity heading minus track heading in radians.",
    ),
    ColumnSpec(
        "accel_along_track_ms2",
        _OPTIONAL,
        True,
        "Acceleration along the aligned track direction in metres per second squared.",
    ),
    ColumnSpec(
        "accel_cross_track_ms2",
        _OPTIONAL,
        True,
        "Acceleration across the aligned track direction in metres per second squared.",
    ),
    ColumnSpec(
        "track_long_g",
        _OPTIONAL,
        True,
        "Along-track acceleration expressed in standard gravity.",
    ),
    ColumnSpec(
        "track_lat_g",
        _OPTIONAL,
        True,
        "Cross-track acceleration expressed in standard gravity.",
    ),
    ColumnSpec(
        "track_section_id",
        _OPTIONAL,
        True,
        "Identifier of the containing configured track section.",
    ),
    ColumnSpec(
        "track_section_name",
        _OPTIONAL,
        True,
        "Name of the containing configured track section.",
    ),
    ColumnSpec(
        "drs_detection_zone_id",
        _OPTIONAL,
        True,
        "Identifier of the matching DRS detection zone.",
    ),
    ColumnSpec(
        "drs_activation_zone_id",
        _OPTIONAL,
        True,
        "Identifier of the matching DRS activation zone.",
    ),
    ColumnSpec(
        "is_in_drs_detection_window",
        _OPTIONAL,
        True,
        "Whether the sample lies in a DRS detection window.",
    ),
    ColumnSpec(
        "is_in_drs_activation_zone",
        _OPTIONAL,
        True,
        "Whether the sample lies in a DRS activation zone.",
    ),
    ColumnSpec(
        "pit_projection_distance_3d_m",
        _OPTIONAL,
        True,
        "3D distance from sample to the pit reference projection in metres.",
    ),
    ColumnSpec(
        "pit_s_m", _OPTIONAL, True, "Distance along the pit reference in metres."
    ),
    ColumnSpec(
        "pit_progress", _OPTIONAL, True, "Fractional progress along the pit reference."
    ),
    ColumnSpec(
        "is_in_pit",
        _OPTIONAL,
        True,
        "Whether the sample is projected into the pit lane.",
    ),
    ColumnSpec(
        "track_s_unwrapped_m",
        _OPTIONAL,
        True,
        "Unwrapped distance along the track reference in metres.",
    ),
)

REPLAY_TABLE_SPECS = (
    TableSpec(
        "sessions",
        _SESSION_COLUMN_SPECS,
        ("session_id",),
        True,
        MergeMode.KEYED,
        ignored_identity_columns=frozenset({"source_file", "source_name"}),
    ),
    TableSpec(
        "laps",
        _LAP_COLUMN_SPECS,
        ("lap_id",),
        True,
        MergeMode.KEYED,
        (ForeignKey(("session_id",), "sessions", ("session_id",)),),
    ),
    TableSpec(
        "samples",
        _SAMPLE_COLUMN_SPECS,
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
        (
            ColumnSpec(
                "severity", _OPTIONAL, True, "Severity assigned to the quality finding."
            ),
            ColumnSpec(
                "code", _REQUIRED, False, "Machine-readable quality finding code."
            ),
            ColumnSpec(
                "session_id", _REQUIRED, False, "Identifier of the affected session."
            ),
            ColumnSpec(
                "lap_id", _REQUIRED, False, "Identifier of the affected lap segment."
            ),
            ColumnSpec(
                "sample_start", _REQUIRED, False, "First affected sample index."
            ),
            ColumnSpec("sample_end", _REQUIRED, False, "Last affected sample index."),
            ColumnSpec(
                "message", _OPTIONAL, True, "Human-readable explanation of the finding."
            ),
            ColumnSpec(
                "affected_channels",
                _OPTIONAL,
                True,
                "Comma-separated affected telemetry channels.",
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
