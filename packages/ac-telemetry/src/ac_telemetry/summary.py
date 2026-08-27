from typing import Any, cast

import numpy as np
import pandas as pd

from .contract_types import MergeMode, TableSpec, column_specs

_STATISTICS_COLUMNS = (
    "setup_id",
    "segment_id",
    "segment_name",
    "valid_pass_count",
    "total_pass_count",
    *(
        f"{prefix}_{statistic}"
        for prefix in (
            "time_s",
            "entry_speed_kmh",
            "minimum_speed_kmh",
            "exit_speed_kmh",
            "brake_start_track_s_m",
            "full_throttle_commit_track_s_m",
            "coasting_time_s",
        )
        for statistic in (
            "count",
            "mean",
            "median",
            "std",
            "mad",
            "min",
            "p25",
            "p75",
            "max",
        )
    ),
    "best_3_mean_time_s",
)

SUMMARY_TABLE_SPECS = (
    TableSpec(
        "summaries/segment_statistics",
        column_specs(_STATISTICS_COLUMNS),
        None,
        False,
        MergeMode.REBUILD,
        rebuild_from=("segments/passes", "sessions"),
        empty_frame_columns=(),
    ),
)


def _stats(group: pd.Series, prefix: str) -> dict[str, float | int | None]:
    values = pd.to_numeric(group, errors="coerce").dropna().to_numpy(float)
    if values.size == 0:
        return {
            f"{prefix}_{key}": None
            for key in [
                "count",
                "mean",
                "median",
                "std",
                "mad",
                "min",
                "p25",
                "p75",
                "max",
            ]
        }
    median = float(np.median(values))
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": median,
        f"{prefix}_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        f"{prefix}_mad": float(np.median(np.abs(values - median))),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_max": float(np.max(values)),
    }


def build_segment_statistics(
    passes: pd.DataFrame, sessions: pd.DataFrame
) -> pd.DataFrame:
    if passes.empty:
        return pd.DataFrame()
    merged = passes.merge(
        sessions[["session_id", "setup_id"]], on="session_id", how="left"
    )
    rows: list[dict[str, Any]] = []
    for keys, group in merged.groupby(["setup_id", "segment_id"], dropna=False):
        setup_id, segment_id = cast(tuple[Any, Any], keys)
        valid = group[group["valid_for_comparison"]]
        target = valid if not valid.empty else group
        row = {
            "setup_id": setup_id,
            "segment_id": segment_id,
            "segment_name": target["segment_name"].iloc[0],
            "valid_pass_count": int(len(valid)),
            "total_pass_count": int(len(group)),
        }
        for column, prefix in [
            ("segment_time_s", "time_s"),
            ("entry_speed_kmh", "entry_speed_kmh"),
            ("minimum_speed_kmh", "minimum_speed_kmh"),
            ("exit_speed_kmh", "exit_speed_kmh"),
            ("brake_onset_track_s_m", "brake_start_track_s_m"),
            ("full_throttle_commit_track_s_m", "full_throttle_commit_track_s_m"),
            ("coasting_time_s", "coasting_time_s"),
        ]:
            row.update(_stats(target[column], prefix))
        fastest = target.nsmallest(min(3, len(target)), "segment_time_s")
        row["best_3_mean_time_s"] = (
            float(fastest["segment_time_s"].mean()) if len(fastest) else None
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_ai_context(
    sessions: pd.DataFrame,
    laps: pd.DataFrame,
    segment_statistics: pd.DataFrame,
    quality_flags: pd.DataFrame,
) -> dict[str, Any]:
    complete = laps[laps["is_complete"]]
    best_rows = complete.nsmallest(min(10, len(complete)), "lap_time_s")
    best_laps = [
        {
            "lap_id": row.lap_id,
            "session_id": row.session_id,
            "lap_time_s": float(row.lap_time_s),
            "source_lap_number": int(row.source_lap_number),
        }
        for row in best_rows.itertuples()
    ]
    segment_data: dict[str, Any] = {}
    for row in segment_statistics.itertuples():
        key = f"{row.setup_id}:{row.segment_id}"
        segment_data[key] = {
            "setup_id": row.setup_id,
            "segment_id": row.segment_id,
            "segment_name": row.segment_name,
            "valid_pass_count": int(row.valid_pass_count),
            "median_time_s": row.time_s_median,
            "mad_time_s": row.time_s_mad,
            "best_time_s": row.time_s_min,
            "median_min_speed_kmh": row.minimum_speed_kmh_median,
        }
    return {
        "dataset": {
            "sessions": int(len(sessions)),
            "complete_laps": int(laps["is_complete"].sum()),
            "valid_laps": int(laps["is_valid"].sum()),
            "setups": int(sessions["setup_id"].nunique(dropna=True)),
        },
        "best_laps": best_laps,
        "segment_statistics": segment_data,
        "known_limitations": [
            "Track coordinates are projected onto AC fast_lane.ai, which is an AI racing line rather than a geometric track centerline",
            "Replay body rotation fields are preserved raw; chassis yaw is not inferred until their semantics are independently validated",
            "Off-track classification uses AI spline side widths when those fields are populated by the track author",
            "TC activity events are spectral candidates because direct torque cut is not recorded",
        ],
        "data_quality": {
            "flag_count": int(len(quality_flags)),
            "error_count": int(
                (quality_flags.get("severity", pd.Series(dtype=str)) == "error").sum()
            ),
            "warning_count": int(
                (quality_flags.get("severity", pd.Series(dtype=str)) == "warning").sum()
            ),
        },
    }
