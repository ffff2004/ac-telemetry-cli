from pathlib import Path
from typing import Any

from .storage import DatasetStorage
from .util import json_load

REQUIRED_LOGICAL_TABLES = {"sessions", "laps", "samples"}


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
    table_map = manifest.get("tables", {})
    missing = sorted(REQUIRED_LOGICAL_TABLES - set(table_map))
    checks["required_tables"] = "pass" if not missing else "fail"
    if missing:
        warnings.append(
            {"code": "MISSING_TABLES", "message": f"Missing tables: {missing}"}
        )

    storage = DatasetStorage(root, manifest.get("table_format", "csv"))
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
            checks[f"table:{logical}"] = "pass"
            if len(frame) != info.get("rows"):
                warnings.append(
                    {
                        "code": "ROW_COUNT_MISMATCH",
                        "message": f"{logical}: manifest={info.get('rows')} actual={len(frame)}",
                    }
                )
        except Exception as exc:  # validation must report, not crash
            checks[f"table:{logical}"] = "fail"
            warnings.append(
                {"code": "TABLE_READ_ERROR", "message": f"{logical}: {exc}"}
            )

    status = (
        "error"
        if any(value == "fail" for value in checks.values())
        else ("warning" if warnings else "ok")
    )
    return {"status": status, "checks": checks, "warnings": warnings}
