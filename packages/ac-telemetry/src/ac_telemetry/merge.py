"""Strict, deterministic merging for generated telemetry datasets.

The public surface is deliberately small: callers supply resolved dataset roots and
an output directory to :func:`merge_datasets`.  All compatibility checks happen
before the output is touched; writing occurs in a sibling staging directory.
"""

import json
import math
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .contract_types import MergeMode, TableSpec
from .dataset_contract import DATASET_CONTRACT
from .manifest import DATASET_SCHEMA_VERSION, require_compatible_schema, table_manifest
from .pipeline import load_dataset_table
from .setup_parser import build_setup_diffs
from .storage import DatasetStorage, TableRef
from .summary import build_ai_context, build_segment_statistics
from .util import json_dump, json_load, stable_id
from .validation import require_valid_dataset, validate_dataset

_TRACK_DISPLAY_FIELDS = {"track_dir", "reference_source", "pit_reference_source"}


def _canonical(value: Any) -> Any:
    """Return a JSON value that treats pandas missing values consistently."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, np.ndarray):
        return [_canonical(item) for item in value.tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    try:
        return None if bool(pd.isna(value)) else value
    except TypeError, ValueError:
        return value


def _token(value: Any) -> str:
    return json.dumps(
        _canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {column: _canonical(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _without_display_values(value: Any, *, track_metadata: bool = False) -> Any:
    """Remove display-only provenance before semantic comparison or identity."""
    if isinstance(value, Mapping):
        ignored = _TRACK_DISPLAY_FIELDS if track_metadata else set()
        return {
            str(key): _without_display_values(item, track_metadata=track_metadata)
            for key, item in value.items()
            if str(key) not in ignored
        }
    if isinstance(value, np.ndarray):
        return [
            _without_display_values(item, track_metadata=track_metadata)
            for item in value
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _without_display_values(item, track_metadata=track_metadata)
            for item in value
        ]
    return value


def _logical_record(logical_name: str, record: dict[str, Any]) -> dict[str, Any]:
    """Return table records without fields that merely describe their source."""
    table = DATASET_CONTRACT.table(logical_name)
    ignored = table.ignored_identity_columns if table is not None else frozenset()
    return {column: value for column, value in record.items() if column not in ignored}


def _logical_records(logical_name: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_logical_record(logical_name, record) for record in _records(frame)]


def _schema(frame: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    return tuple((str(column), str(dtype)) for column, dtype in frame.dtypes.items())


def _is_missing(value: Any) -> bool:
    return _canonical(value) is None


def _merge_keyed(table: TableSpec, frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    materialized = list(frames)
    if not materialized:
        raise ValueError(f"No frames supplied for {table.name!r}")
    if table.allows_untyped_empty_frame:
        materialized = [
            frame
            for frame in materialized
            if not (frame.empty and len(frame.columns) == 0)
        ]
        if not materialized:
            return pd.DataFrame()
    keys = table.stable_key
    if keys is None:  # guarded by DatasetContract.validate_definition
        raise ValueError(f"No stable key is defined for merge table {table.name!r}")
    required = set(table.required_columns) | set(keys)
    for frame in materialized:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                f"Table {table.name!r} is missing required columns {missing}"
            )
    shared = set.intersection(*(set(frame.columns) for frame in materialized))
    incompatible = sorted(
        column
        for column in shared
        if len({str(frame[column].dtype) for frame in materialized}) != 1
    )
    if incompatible:
        raise ValueError(
            f"Incompatible schemas for shared table {table.name!r}: {incompatible}"
        )
    declared = [column.name for column in table.columns]
    extensions = [
        column
        for frame in materialized
        for column in frame.columns
        if column not in declared
    ]
    output_columns = list(
        dict.fromkeys(
            [
                column
                for column in declared
                if any(column in frame for frame in materialized)
            ]
            + extensions
        )
    )

    by_key: dict[str, dict[str, Any]] = {}
    for frame in materialized:
        for record in _records(frame):
            key = _token([record[column] for column in keys])
            previous = by_key.get(key)
            if previous is None:
                by_key[key] = {column: record.get(column) for column in output_columns}
                continue
            for column in output_columns:
                old = previous.get(column)
                new = record.get(column)
                if column in table.ignored_identity_columns:
                    if _is_missing(old) or (
                        not _is_missing(new) and _token(new) < _token(old)
                    ):
                        previous[column] = new
                elif _is_missing(old):
                    previous[column] = new
                elif not _is_missing(new) and _token(old) != _token(new):
                    raise ValueError(
                        f"Conflicting records for {table.name!r} key {key}"
                    )
    ordered = [by_key[key] for key in sorted(by_key)]
    return pd.DataFrame(ordered, columns=output_columns)


def _merge_static_equal(
    table: TableSpec, frames: Iterable[pd.DataFrame]
) -> pd.DataFrame:
    """Merge schema-compatible copies of one static table without adding rows."""
    materialized = list(frames)
    if not materialized:
        raise ValueError(f"No frames supplied for {table.name!r}")
    keys = table.stable_key
    if keys is None:  # guarded for the current producer declarations
        raise ValueError(f"No stable key is defined for static table {table.name!r}")
    if any(key not in frame.columns for frame in materialized for key in keys):
        if not all(
            _schema(materialized[0]) == _schema(frame)
            and sorted(_token(record) for record in _records(materialized[0]))
            == sorted(_token(record) for record in _records(frame))
            for frame in materialized[1:]
        ):
            raise ValueError(f"Static rows for {table.name!r} differ between inputs")
        return materialized[0].copy()

    required = set(table.required_columns) | set(keys)
    for frame in materialized:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                f"Table {table.name!r} is missing required columns {missing}"
            )
    shared = set.intersection(*(set(frame.columns) for frame in materialized))
    incompatible = sorted(
        column
        for column in shared
        if len({str(frame[column].dtype) for frame in materialized}) != 1
    )
    if incompatible:
        raise ValueError(
            f"Incompatible schemas for shared table {table.name!r}: {incompatible}"
        )

    declared = [column.name for column in table.columns]
    extensions = [
        column
        for frame in materialized
        for column in frame.columns
        if column not in declared
    ]
    output_columns = list(
        dict.fromkeys(
            [
                column
                for column in declared
                if any(column in frame for frame in materialized)
            ]
            + extensions
        )
    )

    records_by_frame: list[dict[str, dict[str, Any]]] = []
    for frame in materialized:
        records: dict[str, dict[str, Any]] = {}
        for record in _records(frame):
            key = _token([record[column] for column in keys])
            if key in records:
                raise ValueError(f"Duplicate static rows for {table.name!r} key {key}")
            records[key] = record
        records_by_frame.append(records)
    expected_keys = set(records_by_frame[0])
    if any(set(records) != expected_keys for records in records_by_frame[1:]):
        raise ValueError(f"Static rows for {table.name!r} differ between inputs")

    merged: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        result = {
            column: records_by_frame[0][key].get(column) for column in output_columns
        }
        for records in records_by_frame[1:]:
            record = records[key]
            for column in output_columns:
                old = result.get(column)
                new = record.get(column)
                if _is_missing(old):
                    result[column] = new
                elif not _is_missing(new) and _token(old) != _token(new):
                    raise ValueError(
                        f"Conflicting static rows for {table.name!r} key {key}"
                    )
        merged.append(result)
    return pd.DataFrame(merged, columns=output_columns)


def _regenerate_table(
    table: TableSpec, merged_tables: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Regenerate one declared derived table from its declared prerequisites."""
    prerequisites = tuple(merged_tables[name] for name in table.rebuild_from)
    if table.name == "setup/diffs":
        return build_setup_diffs(prerequisites[0])
    if table.name == "summaries/segment_statistics":
        return build_segment_statistics(prerequisites[0], prerequisites[1])
    raise ValueError(f"No rebuild implementation for {table.name!r}")


def _ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    order = sorted(
        range(len(frame)), key=lambda index: _token(_records(frame.iloc[[index]])[0])
    )
    return frame.iloc[order].reset_index(drop=True)


def _validate_table_names(manifest: Mapping[str, Any], root: Path) -> None:
    try:
        DATASET_CONTRACT.merge_plan(manifest["tables"])
    except ValueError as exc:
        raise ValueError(f"Dataset {root} contains {exc}") from exc


def _load_inputs(
    roots: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, list[pd.DataFrame]]]:
    if not roots:
        raise ValueError("Provide at least one input dataset")
    manifests: list[dict[str, Any]] = []
    tables: dict[str, list[pd.DataFrame]] = {}
    for root in roots:
        manifest_path = root / "manifest.json"
        if not root.is_dir() or not manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
        require_valid_dataset(root)
        manifest = json_load(manifest_path)
        if not isinstance(manifest, dict) or not isinstance(
            manifest.get("tables"), dict
        ):
            raise ValueError(f"Invalid dataset manifest: {manifest_path}")
        _validate_table_names(manifest, root)
        input_tables = {
            logical_name: load_dataset_table(root, logical_name)
            for logical_name in manifest["tables"]
        }
        manifests.append(manifest)
        for logical_name, frame in input_tables.items():
            tables.setdefault(logical_name, []).append(frame)
    return manifests, tables


def _require_identical_values(
    manifests: list[dict[str, Any]], field: str, *, strip_track_provenance: bool = False
) -> Any:
    values = [
        _token(
            _without_display_values(
                manifest.get(field), track_metadata=strip_track_provenance
            )
        )
        for manifest in manifests
    ]
    if len(set(values)) != 1:
        raise ValueError(f"All input datasets must have identical {field}")
    return _canonical(
        _without_display_values(
            manifests[0].get(field), track_metadata=strip_track_provenance
        )
    )


def _canonical_segment_definitions(value: Any, *, segment_list: bool = False) -> Any:
    """Canonicalize definition maps and the order-insensitive segment collections."""
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_segment_definitions(
                item, segment_list=str(key) in {"segments", "subsegments"}
            )
            for key, item in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = [_canonical_segment_definitions(item) for item in value]
        return sorted(items, key=_token) if segment_list else items
    return _canonical(value)


def _load_segment_definitions(roots: list[Path]) -> dict[str, Any] | None:
    paths = [root / "segments" / "definitions.json" for root in roots]
    present = [path.exists() for path in paths]
    if any(present) and not all(present):
        raise ValueError(
            "Segment definitions must be present in every input or absent in every input"
        )
    if not any(present):
        return None
    definitions = [_canonical_segment_definitions(json_load(path)) for path in paths]
    if len({_token(value) for value in definitions}) != 1:
        raise ValueError(
            "All input datasets must have semantically identical segment definitions"
        )
    return definitions[0]


def _raw_registry(roots: list[Path]) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for root in roots:
        path = root / "setup" / "raw.json"
        if not path.exists():
            continue
        raw = json_load(path)
        for setup_id, metadata in raw.items():
            semantic = {
                "source_hash": metadata.get("source_hash"),
                "raw": metadata.get("raw"),
            }
            paths = metadata.get("source_files", [metadata.get("source_file")])
            labels = metadata.get("setup_labels", [metadata.get("setup_label")])
            previous = registry.get(str(setup_id))
            if previous is not None and _token(previous["semantic"]) != _token(
                semantic
            ):
                raise ValueError(
                    f"Conflicting setup raw metadata for setup_id {setup_id!r}"
                )
            entry = previous or {
                "semantic": _canonical(semantic),
                "paths": set(),
                "labels": set(),
            }
            entry["paths"].update(str(item) for item in paths if item is not None)
            entry["labels"].update(str(item) for item in labels if item is not None)
            registry[str(setup_id)] = entry
    result: dict[str, dict[str, Any]] = {}
    for setup_id, entry in sorted(registry.items()):
        paths = sorted(entry["paths"])
        labels = sorted(entry["labels"])
        result[setup_id] = {
            "setup_id": setup_id,
            **entry["semantic"],
            "source_file": paths[0] if paths else None,
            "setup_label": labels[0] if labels else None,
            "source_files": paths,
            "setup_labels": labels,
        }
    return result


def _flatten_source_files(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group original sources by content identity while retaining all display paths."""
    flattened: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        for entry in manifest.get("source_files", []):
            if not isinstance(entry, dict):
                raise ValueError("Manifest source_files entries must be objects")
            source_hash = entry.get("sha256")
            source_type = entry.get("type")
            if not isinstance(source_hash, str) or not source_hash:
                raise ValueError("Manifest source_files entries must contain a sha256")
            if not isinstance(source_type, str) or not source_type:
                raise ValueError("Manifest source_files entries must contain a type")
            key = _token({"sha256": source_hash, "type": source_type})
            item = flattened.setdefault(
                key,
                {
                    "sha256": source_hash,
                    "type": source_type,
                    "display_paths": set(),
                    "display_names": set(),
                },
            )
            paths = entry.get("display_paths", [entry.get("path")])
            names = entry.get("display_names", [entry.get("name")])
            if not isinstance(paths, list) or not isinstance(names, list):
                raise ValueError(
                    "Manifest source file display provenance must be lists"
                )
            item["display_paths"].update(
                str(path) for path in paths if path is not None
            )
            item["display_names"].update(
                str(name) for name in names if name is not None
            )
    return [
        {
            "sha256": item["sha256"],
            "type": item["type"],
            "display_paths": sorted(item["display_paths"]),
            "display_names": sorted(item["display_names"]),
        }
        for _, item in sorted(flattened.items())
    ]


def _publish(staging: Path, output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output directory exists: {output}")
    if not output.exists():
        os.replace(staging, output)
        return
    backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
    os.replace(output, backup)
    try:
        os.replace(staging, output)
    except Exception:
        os.replace(backup, output)
        raise
    try:
        shutil.rmtree(backup)
    except OSError:
        # Publication succeeded; retaining an inaccessible backup is safer than
        # reporting a failed merge after replacing the requested output.
        pass


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def merge_datasets(
    roots: list[Path], output: Path, *, overwrite: bool = False
) -> dict[str, Any]:
    """Merge v7 datasets strictly and publish the completed result atomically."""
    input_roots = [root.expanduser().resolve() for root in roots]
    output = output.expanduser().resolve()
    if any(_paths_overlap(output, root) for root in input_roots):
        raise ValueError("Merge output cannot also be an input dataset")

    manifests, grouped_tables = _load_inputs(input_roots)
    require_compatible_schema(manifests)
    track_reference_id = _require_identical_values(manifests, "track_reference_id")
    track = _require_identical_values(manifests, "track", strip_track_provenance=True)
    processing_options = _require_identical_values(manifests, "processing_options")
    definitions = _load_segment_definitions(input_roots)
    raw = _raw_registry(input_roots)

    plan = DATASET_CONTRACT.merge_plan(grouped_tables)
    static_names = [
        table.name for table in plan if table.merge_mode is MergeMode.STATIC_EQUAL
    ]
    expected_static = set(static_names)
    for manifest in manifests:
        manifest_plan = DATASET_CONTRACT.merge_plan(manifest["tables"])
        if {
            table.name
            for table in manifest_plan
            if table.merge_mode is MergeMode.STATIC_EQUAL
        } != expected_static:
            raise ValueError(
                "All input datasets must contain the same static track tables"
            )

    merged_tables: dict[str, pd.DataFrame] = {}
    for table in plan:
        logical_name = table.name
        if table.merge_mode is MergeMode.REBUILD:
            continue
        frames = grouped_tables[logical_name]
        if table.merge_mode is MergeMode.STATIC_EQUAL:
            try:
                merged_tables[logical_name] = _ordered_frame(
                    _merge_static_equal(table, frames)
                )
            except ValueError as exc:
                raise ValueError(
                    f"Static track table {logical_name!r} differs between inputs"
                ) from exc
        else:
            merged_tables[logical_name] = _merge_keyed(table, frames)

    for table in plan:
        if table.merge_mode is not MergeMode.REBUILD:
            continue
        rebuilt = _regenerate_table(table, merged_tables)
        if not (table.omit_if_empty and rebuilt.empty):
            merged_tables[table.name] = rebuilt
    statistics = merged_tables.get("summaries/segment_statistics", pd.DataFrame())
    quality = merged_tables.get("quality/flags", pd.DataFrame())
    ai_context = build_ai_context(
        merged_tables["sessions"], merged_tables["laps"], statistics, quality
    )

    source_files = _flatten_source_files(manifests)
    source_datasets = sorted(
        {
            str(item)
            for root, manifest in zip(input_roots, manifests, strict=True)
            for item in manifest.get("source_datasets", [str(root)])
        }
    )
    identity = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "processing_options": processing_options,
        "track_reference_id": track_reference_id,
        "track": track,
        "segment_definitions": definitions,
        "source_files": [
            {"sha256": item["sha256"], "type": item["type"]} for item in source_files
        ],
        "tables": {
            name: sorted(_token(record) for record in _logical_records(name, frame))
            for name, frame in sorted(merged_tables.items())
        },
    }
    dataset_id = stable_id("merged-v7", _token(identity))

    # Only now is any output state created or altered.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        storage = DatasetStorage(staging)
        refs: list[TableRef] = []
        for logical_name, frame in sorted(merged_tables.items()):
            refs.append(storage.write(logical_name, frame))
        if definitions is not None:
            json_dump(staging / "segments" / "definitions.json", definitions)
        if raw:
            json_dump(staging / "setup" / "raw.json", raw)
        json_dump(staging / "summaries" / "ai_context.json", ai_context)
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "tool_version": __version__,
            "dataset_id": dataset_id,
            "source_files": source_files,
            "source_datasets": source_datasets,
            "processing_options": processing_options,
            "track": track,
            "track_reference_id": track_reference_id,
            "segment_definition_source": None,
            "tables": table_manifest(refs),
            "warnings": [],
        }
        json_dump(staging / "manifest.json", manifest)
        validation = validate_dataset(staging)
        json_dump(staging / "quality" / "validation.json", validation)
        if validation["status"] == "error":
            raise ValueError("Merged dataset validation failed")
        _publish(staging, output, overwrite)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
