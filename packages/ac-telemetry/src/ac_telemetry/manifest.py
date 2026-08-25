from __future__ import annotations

from typing import Any

from .storage import TableRef


DATASET_SCHEMA_VERSION = "3"


def table_manifest(refs: list[TableRef]) -> dict[str, dict[str, Any]]:
    return {
        ref.name: {"path": ref.relative_path, "rows": ref.rows, "columns": ref.columns}
        for ref in refs
    }


def require_compatible_schema(manifests: list[dict[str, Any]]) -> None:
    incompatible = [
        manifest.get("schema_version")
        for manifest in manifests
        if manifest.get("schema_version") != DATASET_SCHEMA_VERSION
    ]
    if incompatible:
        raise ValueError(
            f"All input datasets must use schema version {DATASET_SCHEMA_VERSION!r}; "
            f"found incompatible versions: {incompatible}"
        )
