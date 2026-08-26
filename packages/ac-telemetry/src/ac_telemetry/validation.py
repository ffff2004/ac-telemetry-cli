from pathlib import Path
from typing import Any

import numpy as np

from .manifest import DATASET_SCHEMA_VERSION
from .storage import DatasetStorage
from .util import json_load

REQUIRED_LOGICAL_TABLES = {"sessions", "laps", "samples", "track/reference"}
REQUIRED_SAMPLE_COLUMNS = {
    "track_s_m",
    "track_progress",
    "track_projection_distance_3d_m",
    "lateral_offset_m",
    "path_distance_2d_m",
    "path_distance_3d_m",
}


def validate_dataset(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    checks: dict[str, Any] = {}
    warnings: list[dict[str, Any]] = []
    if not manifest_path.exists():
        return {
            "status": "error",
            "checks": {"manifest": "fail"},
            "warnings": [
                {"code": "MISSING_MANIFEST", "message": "manifest.json not found"}
            ],
        }
    manifest = json_load(manifest_path)
    checks["manifest"] = "pass"
    checks["schema_version"] = (
        "pass" if manifest.get("schema_version") == DATASET_SCHEMA_VERSION else "fail"
    )
    if checks["schema_version"] == "fail":
        warnings.append(
            {
                "code": "SCHEMA_VERSION_MISMATCH",
                "message": f"Expected schema {DATASET_SCHEMA_VERSION}, found {manifest.get('schema_version')}",
            }
        )
    if not manifest.get("track_reference_id"):
        checks["track_reference_id"] = "fail"
        warnings.append(
            {
                "code": "MISSING_TRACK_REFERENCE_ID",
                "message": "manifest has no track_reference_id",
            }
        )
    else:
        checks["track_reference_id"] = "pass"

    table_map = manifest.get("tables", {})
    missing = sorted(REQUIRED_LOGICAL_TABLES - set(table_map))
    checks["required_tables"] = "pass" if not missing else "fail"
    if missing:
        warnings.append(
            {"code": "MISSING_TABLES", "message": f"Missing tables: {missing}"}
        )

    storage = DatasetStorage(root, manifest.get("table_format", "csv"))
    loaded: dict[str, Any] = {}
    for logical, info in table_map.items():
        path = root / info["path"]
        if not path.exists():
            checks[f"table:{logical}"] = "fail"
            warnings.append(
                {"code": "MISSING_TABLE_FILE", "message": f"{logical}: {path}"}
            )
            continue
        try:
            frame = storage.read(info["path"])
            loaded[logical] = frame
            checks[f"table:{logical}"] = "pass"
            if len(frame) != info.get("rows"):
                warnings.append(
                    {
                        "code": "ROW_COUNT_MISMATCH",
                        "message": f"{logical}: manifest={info.get('rows')} actual={len(frame)}",
                    }
                )
        except Exception as exc:  # validation must report, not crash
            # CSV has no representable schema for a 0-row/0-column DataFrame;
            # pandas writes a blank file and read_csv raises EmptyDataError. The
            # manifest row count is the observable contract in that case.
            if info.get("rows") == 0 and path.stat().st_size <= 2:
                checks[f"table:{logical}"] = "pass"
                continue
            checks[f"table:{logical}"] = "fail"
            warnings.append(
                {"code": "TABLE_READ_ERROR", "message": f"{logical}: {exc}"}
            )

    samples = loaded.get("samples")
    if samples is not None:
        missing_columns = sorted(REQUIRED_SAMPLE_COLUMNS - set(samples.columns))
        checks["sample_coordinate_columns"] = "pass" if not missing_columns else "fail"
        if missing_columns:
            warnings.append(
                {
                    "code": "MISSING_TRACK_COLUMNS",
                    "message": f"samples missing track columns: {missing_columns}",
                }
            )
        else:
            progress = samples["track_progress"].to_numpy(float)
            finite = np.isfinite(progress)
            in_range = bool(
                (~finite | ((progress >= 0.0) & (progress < 1.0 + 1e-9))).all()
            )
            checks["track_progress_range"] = "pass" if in_range else "fail"
            if not in_range:
                warnings.append(
                    {
                        "code": "TRACK_PROGRESS_OUT_OF_RANGE",
                        "message": "track_progress must remain in [0, 1)",
                    }
                )
            path_2d = samples["path_distance_2d_m"].to_numpy(float)
            path_3d = samples["path_distance_3d_m"].to_numpy(float)
            path_order_ok = bool((path_3d + 1e-9 >= path_2d).all())
            checks["path_distance_dimensions"] = "pass" if path_order_ok else "fail"
            if not path_order_ok:
                warnings.append(
                    {
                        "code": "PATH_DISTANCE_DIMENSION_ERROR",
                        "message": "3D path distance must not be shorter than 2D path distance",
                    }
                )

    status = (
        "error"
        if any(value == "fail" for value in checks.values())
        else ("warning" if warnings else "ok")
    )
    return {"status": status, "checks": checks, "warnings": warnings}
