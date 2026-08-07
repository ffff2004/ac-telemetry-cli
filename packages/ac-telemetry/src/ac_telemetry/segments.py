from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .util import json_load


SUPPORTED_COORDINATES = {"actual_distance_m", "normalized_progress"}


def load_segment_definitions(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = json_load(path)
    coordinate = data.get("coordinate", "actual_distance_m")
    if coordinate not in SUPPORTED_COORDINATES:
        raise ValueError(
            f"Unsupported segment coordinate {coordinate!r}; expected one of {sorted(SUPPORTED_COORDINATES)}"
        )
    if not isinstance(data.get("segments"), list):
        raise ValueError("Segment definition must contain a segments list")
    return data


def _segment_mask(g: pd.DataFrame, start: float, end: float, coordinate: str) -> pd.Series:
    column = "actual_distance_m" if coordinate == "actual_distance_m" else "progress"
    values = g[column]
    if start <= end:
        return values.ge(start) & values.le(end)
    return values.ge(start) | values.le(end)


def _first_value(segment: pd.DataFrame, mask: pd.Series, column: str) -> float:
    found = segment.loc[mask, column]
    return float(found.iloc[0]) if len(found) else np.nan


def _last_value(segment: pd.DataFrame, mask: pd.Series, column: str) -> float:
    found = segment.loc[mask, column]
    return float(found.iloc[-1]) if len(found) else np.nan


def segment_passes(
    samples: pd.DataFrame,
    laps: pd.DataFrame,
    definitions: dict[str, Any] | None,
) -> pd.DataFrame:
    if definitions is None:
        return pd.DataFrame()
    coordinate = definitions.get("coordinate", "actual_distance_m")
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
        g = original.sort_values("sample_index")
        for definition in expanded:
            start = float(definition["start"])
            end = float(definition["end"])
            mask = _segment_mask(g, start, end, coordinate)
            segment = g.loc[mask]
            if len(segment) < 2:
                continue
            coordinate_column = "actual_distance_m" if coordinate == "actual_distance_m" else "progress"
            min_speed_idx = segment["speed_kmh"].idxmin()
            brake_mask = segment["is_braking"]
            throttle_mask = segment["throttle"] >= 0.05
            full_mask = segment["is_full_throttle"]
            path_dx = segment["position.x"].diff().fillna(0)
            path_dz = segment["position.z"].diff().fillna(0)
            path_length = float(np.hypot(path_dx, path_dz).sum())
            lap_info = lap_lookup.loc[lap_id]
            rows.append(
                {
                    "session_id": segment["session_id"].iloc[0],
                    "lap_id": lap_id,
                    "source_lap_number": int(segment["source_lap_number"].iloc[0]),
                    "segment_id": definition.get("id"),
                    "segment_name": definition.get("name", definition.get("id")),
                    "parent_segment_id": definition.get("parent_id"),
                    "coordinate": coordinate,
                    "segment_start": start,
                    "segment_end": end,
                    "sample_count": len(segment),
                    "segment_time_s": float(segment["lap_time_s"].iloc[-1] - segment["lap_time_s"].iloc[0]),
                    "entry_speed_kmh": float(segment["speed_kmh"].iloc[0]),
                    "exit_speed_kmh": float(segment["speed_kmh"].iloc[-1]),
                    "minimum_speed_kmh": float(segment["speed_kmh"].min()),
                    "minimum_speed_position": float(segment.loc[min_speed_idx, coordinate_column]),
                    "brake_start_position": _first_value(segment, brake_mask, coordinate_column),
                    "brake_end_position": _last_value(segment, brake_mask, coordinate_column),
                    "brake_duration_s": float(segment.loc[brake_mask, "dt_s"].sum()),
                    "peak_brake": float(segment["brake_n"].max()),
                    "brake_impulse_proxy_s": float((segment["brake_n"] * segment["dt_s"]).sum()),
                    "throttle_pickup_position": _first_value(segment, throttle_mask, coordinate_column),
                    "full_throttle_position": _first_value(segment, full_mask, coordinate_column),
                    "coasting_time_s": float(segment.loc[segment["is_coasting"], "dt_s"].sum()),
                    "partial_throttle_time_s": float(segment.loc[segment["is_partial_throttle"], "dt_s"].sum()),
                    "front_lock_time_s": float(segment.loc[segment["is_front_lock_candidate"], "dt_s"].sum()),
                    "rear_wheelspin_time_s": float(segment.loc[segment["is_rear_wheelspin_candidate"], "dt_s"].sum()),
                    "rear_slip_integral": float(
                        (segment["rear_slip_ratio_max"].clip(lower=0) * segment["dt_s"]).sum()
                    ),
                    "max_abs_steer": float(segment["steerAngle"].abs().max()),
                    "steering_sign_changes": int(
                        np.count_nonzero(np.diff(np.sign(segment["steerAngle"].fillna(0).to_numpy())) != 0)
                    ),
                    "actual_path_length_m": path_length,
                    "is_complete_lap": bool(lap_info["is_complete"]),
                    "is_valid_lap": bool(lap_info["is_valid"]),
                    "quality_score": 1.0 if bool(lap_info["is_valid"]) else 0.5,
                    "valid_for_comparison": bool(lap_info["is_valid"]),
                }
            )
    return pd.DataFrame(rows)
