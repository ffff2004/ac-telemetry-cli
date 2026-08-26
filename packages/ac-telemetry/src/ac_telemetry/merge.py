"""Strict, deterministic merging for generated telemetry datasets.

The public surface is deliberately small: callers supply resolved dataset roots and
an output directory to :func:`merge_datasets`.  All compatibility checks happen
before the output is touched; writing occurs in a sibling staging directory.
"""

from __future__ import annotations

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
from .manifest import DATASET_SCHEMA_VERSION, require_compatible_schema, table_manifest
from .pipeline import load_dataset_table
from .setup_parser import build_setup_diffs
from .storage import DatasetStorage, TableRef
from .summary import build_ai_context, build_segment_statistics
from .util import json_dump, json_load, stable_id
from .validation import validate_dataset

_GENERATED_TABLES = {
    "setup/diffs",
    "summaries/segment_statistics",
}
_STATIC_TABLE_PREFIX = "track/"
_REQUIRED_TABLES = {"sessions", "laps", "samples", "track/reference"}
_OPTIONAL_TABLES = {
    "quality/flags",
    "setup/normalized",
    "setup/diffs",
    "segments/passes",
    "summaries/segment_statistics",
    "events/index",
    "events/braking",
    "events/throttle",
    "events/shifts",
    "events/wheel_slip",
    "events/relations",
    "events/abs_activity",
    "events/tc_activity",
    "track/pit_reference",
    "track/sections",
    "track/drs_zones",
}
_DISPLAY_COLUMNS = {"source_file", "source_name", "setup_label"}
_TRACK_DISPLAY_FIELDS = {"track_dir", "reference_source", "pit_reference_source"}

_KEY_COLUMNS: dict[str, tuple[str, ...]] = {
    "sessions": ("session_id",),
    "laps": ("lap_id",),
    "samples": ("session_id", "lap_id", "sample_index"),
    "quality/flags": ("session_id", "lap_id", "code", "sample_start", "sample_end"),
    "setup/normalized": ("setup_id", "section", "parameter"),
    "segments/passes": ("session_id", "lap_id", "segment_id"),
    "events/relations": ("relation_id",),
}


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
            if str(key) not in ignored and str(key) not in _DISPLAY_COLUMNS
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
    ignored = (
        _DISPLAY_COLUMNS if logical_name in {"sessions", "setup/normalized"} else set()
    )
    return {column: value for column, value in record.items() if column not in ignored}


def _logical_records(logical_name: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_logical_record(logical_name, record) for record in _records(frame)]


def _schema(frame: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    return tuple((str(column), str(dtype)) for column, dtype in frame.dtypes.items())


def _same_frame(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if _schema(left) != _schema(right):
        return False
    return sorted(_token(record) for record in _records(left)) == sorted(
        _token(record) for record in _records(right)
    )


def _key_columns(logical_name: str, frame: pd.DataFrame) -> tuple[str, ...]:
    if logical_name in _KEY_COLUMNS:
        return _KEY_COLUMNS[logical_name]
    if logical_name.startswith("events/") and "event_id" in frame.columns:
        return ("event_id",)
    raise ValueError(f"No stable key is defined for merge table {logical_name!r}")


def _merge_keyed(logical_name: str, frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    materialized = list(frames)
    if not materialized:
        raise ValueError(f"No frames supplied for {logical_name!r}")
    schema = _schema(materialized[0])
    if any(_schema(frame) != schema for frame in materialized[1:]):
        raise ValueError(f"Incompatible schemas for shared table {logical_name!r}")
    keys = _key_columns(logical_name, materialized[0])
    missing = sorted(set(keys) - set(materialized[0].columns))
    if missing:
        raise ValueError(
            f"Table {logical_name!r} is missing merge key columns {missing}"
        )

    by_key: dict[str, dict[str, Any]] = {}
    for frame in materialized:
        for record in _records(frame):
            key = _token([record[column] for column in keys])
            previous = by_key.get(key)
            if previous is not None:
                if _token(_logical_record(logical_name, previous)) != _token(
                    _logical_record(logical_name, record)
                ):
                    raise ValueError(
                        f"Conflicting records for {logical_name!r} key {key}"
                    )
                if _token(record) >= _token(previous):
                    continue
            by_key[key] = record
    ordered = [by_key[key] for key in sorted(by_key)]
    return pd.DataFrame(ordered, columns=materialized[0].columns)


def _merge_normalized_setups(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Merge setup parameters while treating names and paths as provenance."""
    materialized = list(frames)
    if not materialized:
        raise ValueError("No setup/normalized frames supplied")
    schema = _schema(materialized[0])
    if any(_schema(frame) != schema for frame in materialized[1:]):
        raise ValueError("Incompatible schemas for shared table 'setup/normalized'")
    keys = _KEY_COLUMNS["setup/normalized"]
    missing = sorted(set(keys) - set(materialized[0].columns))
    if missing:
        raise ValueError(
            f"Table 'setup/normalized' is missing merge key columns {missing}"
        )
    provenance = {"setup_label", "source_file"}
    by_key: dict[str, dict[str, Any]] = {}
    for frame in materialized:
        for record in _records(frame):
            key = _token([record[column] for column in keys])
            comparable = {
                name: value for name, value in record.items() if name not in provenance
            }
            previous = by_key.get(key)
            if previous is not None:
                old_comparable = {
                    name: value
                    for name, value in previous.items()
                    if name not in provenance
                }
                if _token(old_comparable) != _token(comparable):
                    raise ValueError(
                        f"Conflicting records for 'setup/normalized' key {key}"
                    )
                if _token(record) >= _token(previous):
                    continue
            by_key[key] = record
    return pd.DataFrame(
        [by_key[key] for key in sorted(by_key)], columns=materialized[0].columns
    )


def _ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    order = sorted(
        range(len(frame)), key=lambda index: _token(_records(frame.iloc[[index]])[0])
    )
    return frame.iloc[order].reset_index(drop=True)


def _validate_table_names(manifest: Mapping[str, Any], root: Path) -> None:
    names = set(manifest["tables"])
    missing = sorted(_REQUIRED_TABLES - names)
    if missing:
        raise ValueError(f"Dataset {root} is missing required tables: {missing}")
    unknown = sorted(names - _REQUIRED_TABLES - _OPTIONAL_TABLES)
    if unknown:
        raise ValueError(f"Dataset {root} contains unsupported merge tables: {unknown}")


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
        _validate_references(input_tables, _raw_registry([root]), context=str(root))
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
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid setup raw registry: {path}")
        for setup_id, metadata in raw.items():
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid setup metadata for {setup_id!r}")
            if metadata.get("setup_id") != str(setup_id):
                raise ValueError(
                    f"Setup registry key and setup_id disagree for {setup_id!r}"
                )
            semantic = {
                "source_hash": metadata.get("source_hash"),
                "raw": metadata.get("raw"),
            }
            if semantic["source_hash"] is None or semantic["raw"] is None:
                raise ValueError(
                    f"Setup {setup_id!r} lacks source_hash or raw metadata"
                )
            paths = metadata.get("source_files", [metadata.get("source_file")])
            labels = metadata.get("setup_labels", [metadata.get("setup_label")])
            if not isinstance(paths, list) or not isinstance(labels, list):
                raise ValueError(f"Invalid setup provenance for {setup_id!r}")
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


def _required_columns(
    logical_name: str, frame: pd.DataFrame, columns: set[str]
) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(
            f"Table {logical_name!r} is missing required columns {missing}"
        )


def _require_non_null(
    logical_name: str, frame: pd.DataFrame, columns: set[str]
) -> None:
    missing_values = sorted(
        column for column in columns if bool(frame[column].isna().any())
    )
    if missing_values:
        raise ValueError(
            f"Table {logical_name!r} has missing values in relational columns {missing_values}"
        )


def _normalized_setup_values(
    frame: pd.DataFrame, setup_id: str
) -> dict[tuple[str, str], Any]:
    _required_columns(
        "setup/normalized",
        frame,
        {
            "setup_id",
            "source_hash",
            "section",
            "parameter",
            "value_numeric",
            "value_text",
        },
    )
    values: dict[tuple[str, str], Any] = {}
    rows = frame[frame["setup_id"].astype(str) == setup_id]
    for record in _records(rows):
        key = (str(record["section"]), str(record["parameter"]))
        if key in values:
            raise ValueError(
                f"Duplicate normalized setup parameter for {setup_id!r}: {key!r}"
            )
        numeric = record["value_numeric"]
        values[key] = numeric if numeric is not None else record["value_text"]
    return values


def _raw_setup_values(
    raw: Mapping[str, Any], setup_id: str
) -> dict[tuple[str, str], Any]:
    contents = raw.get(setup_id, {}).get("raw")
    if not isinstance(contents, Mapping):
        raise ValueError(f"Setup {setup_id!r} has invalid raw setup contents")
    values: dict[tuple[str, str], Any] = {}
    for section, parameters in contents.items():
        if not isinstance(parameters, Mapping):
            raise ValueError(f"Setup {setup_id!r} section {section!r} is not an object")
        for parameter, value in parameters.items():
            key = (str(section), str(parameter))
            if key in values:
                raise ValueError(f"Setup {setup_id!r} duplicates raw parameter {key!r}")
            values[key] = _canonical(value)
    return values


def _validate_setup_contents(
    normalized: pd.DataFrame, raw: Mapping[str, Any], setup_id: str
) -> None:
    _required_columns(
        "setup/normalized",
        normalized,
        {
            "setup_id",
            "source_hash",
            "section",
            "parameter",
            "value_numeric",
            "value_text",
        },
    )
    source_hashes = {
        _token(value)
        for value in normalized.loc[
            normalized["setup_id"].astype(str) == setup_id, "source_hash"
        ]
    }
    if source_hashes != {_token(raw[setup_id]["source_hash"])}:
        raise ValueError(
            f"Setup {setup_id!r} source_hash disagrees between raw registry and normalized table"
        )
    normalized_values = _normalized_setup_values(normalized, setup_id)
    raw_values = _raw_setup_values(raw, setup_id)
    if _token(normalized_values) != _token(raw_values):
        raise ValueError(
            f"Setup {setup_id!r} raw registry and normalized parameters disagree"
        )


def _validate_references(
    tables: Mapping[str, pd.DataFrame],
    raw: Mapping[str, Any],
    *,
    context: str = "merged datasets",
) -> None:
    sessions = tables.get("sessions")
    laps = tables.get("laps")
    samples = tables.get("samples")
    reference = tables.get("track/reference")
    if sessions is None or laps is None or samples is None or reference is None:
        raise ValueError(
            f"{context} must contain sessions, laps, samples, and track/reference"
        )
    _required_columns("sessions", sessions, {"session_id"})
    _required_columns("laps", laps, {"session_id", "lap_id"})
    _required_columns("samples", samples, {"session_id", "lap_id", "sample_index"})
    _require_non_null("sessions", sessions, {"session_id"})
    _require_non_null("laps", laps, {"session_id", "lap_id"})
    _require_non_null("samples", samples, {"session_id", "lap_id", "sample_index"})
    session_ids = set(sessions["session_id"].dropna().astype(str))
    lap_ids = set(laps["lap_id"].dropna().astype(str))
    lap_pairs = set(
        zip(
            laps["session_id"].dropna().astype(str),
            laps["lap_id"].dropna().astype(str),
            strict=True,
        )
    )
    normalized = tables.get("setup/normalized")
    normalized_ids = (
        set(normalized["setup_id"].dropna().astype(str))
        if normalized is not None
        else set()
    )
    session_setups = (
        sessions["setup_id"] if "setup_id" in sessions else pd.Series(dtype=object)
    )
    for setup_id in session_setups.dropna().astype(str):
        if setup_id not in normalized_ids or setup_id not in raw:
            raise ValueError(
                f"Session setup_id {setup_id!r} is not linked to normalized and raw setup data"
            )
        if normalized is None:
            raise ValueError(
                f"Session setup_id {setup_id!r} has no normalized setup data"
            )
        _validate_setup_contents(normalized, raw, setup_id)
    if normalized is not None:
        missing_raw = normalized_ids - set(raw)
        if missing_raw:
            raise ValueError(
                f"Normalized setups missing raw metadata: {sorted(missing_raw)}"
            )

    for logical_name, frame in tables.items():
        if (
            "session_id" in frame
            and not set(frame["session_id"].dropna().astype(str)) <= session_ids
        ):
            raise ValueError(f"Table {logical_name!r} references an unknown session_id")
        if (
            "lap_id" in frame
            and not set(frame["lap_id"].dropna().astype(str)) <= lap_ids
        ):
            raise ValueError(f"Table {logical_name!r} references an unknown lap_id")
        if {"session_id", "lap_id"} <= set(frame.columns):
            pairs = set(
                zip(
                    frame["session_id"].dropna().astype(str),
                    frame["lap_id"].dropna().astype(str),
                    strict=True,
                )
            )
            if not pairs <= lap_pairs:
                raise ValueError(
                    f"Table {logical_name!r} references an unknown session/lap pair"
                )
    sessions_with_laps = set(laps["session_id"].dropna().astype(str))
    if not session_ids <= sessions_with_laps:
        raise ValueError("Sessions without laps are not valid merge inputs")
    sample_pairs = set(
        zip(
            samples["session_id"].dropna().astype(str),
            samples["lap_id"].dropna().astype(str),
            strict=True,
        )
    )
    if not lap_pairs <= sample_pairs:
        raise ValueError("Laps without samples are not valid merge inputs")
    event_index = tables.get("events/index")
    event_tables = [name for name in tables if name.startswith("events/")]
    if event_tables and event_index is None:
        raise ValueError("Event fact tables require events/index")
    if event_index is not None:
        _required_columns("events/index", event_index, {"event_id"})
        event_ids = set(event_index["event_id"].dropna().astype(str))
        for logical_name, frame in tables.items():
            if logical_name == "events/index":
                continue
            for column in (
                "event_id",
                "parent_braking_event_id",
                "parent_throttle_event_id",
            ):
                if column in frame:
                    values = set(frame[column].dropna().astype(str))
                    if not values <= event_ids:
                        raise ValueError(
                            f"Table {logical_name!r} references an unknown event_id"
                        )
        relations = tables.get("events/relations")
        if relations is not None:
            _required_columns(
                "events/relations", relations, {"event_id_a", "event_id_b"}
            )
            for column in ("event_id_a", "event_id_b"):
                if not set(relations[column].dropna().astype(str)) <= event_ids:
                    raise ValueError("Event relations reference an unknown event_id")


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


def merge_datasets(
    roots: list[Path], output: Path, *, overwrite: bool = False
) -> dict[str, Any]:
    """Merge v7 datasets strictly and publish the completed result atomically."""
    input_roots = [root.expanduser().resolve() for root in roots]
    output = output.expanduser().resolve()
    if output in input_roots:
        raise ValueError("Merge output cannot also be an input dataset")

    manifests, grouped_tables = _load_inputs(input_roots)
    require_compatible_schema(manifests)
    track_reference_id = _require_identical_values(manifests, "track_reference_id")
    track = _require_identical_values(manifests, "track", strip_track_provenance=True)
    processing_options = _require_identical_values(manifests, "processing_options")
    definitions = _load_segment_definitions(input_roots)
    raw = _raw_registry(input_roots)

    all_table_names = sorted(grouped_tables)
    static_names = [
        name for name in all_table_names if name.startswith(_STATIC_TABLE_PREFIX)
    ]
    expected_static = set(static_names)
    for manifest in manifests:
        if {
            name for name in manifest["tables"] if name.startswith(_STATIC_TABLE_PREFIX)
        } != expected_static:
            raise ValueError(
                "All input datasets must contain the same static track tables"
            )

    merged_tables: dict[str, pd.DataFrame] = {}
    for logical_name in all_table_names:
        if logical_name in _GENERATED_TABLES:
            continue
        frames = grouped_tables[logical_name]
        if logical_name.startswith(_STATIC_TABLE_PREFIX):
            if len(frames) != len(manifests) or not all(
                _same_frame(frames[0], frame) for frame in frames[1:]
            ):
                raise ValueError(
                    f"Static track table {logical_name!r} differs between inputs"
                )
            merged_tables[logical_name] = _ordered_frame(frames[0])
        elif logical_name == "setup/normalized":
            merged_tables[logical_name] = _merge_normalized_setups(frames)
        else:
            merged_tables[logical_name] = _merge_keyed(logical_name, frames)

    _validate_references(merged_tables, raw)
    setups = merged_tables.get("setup/normalized", pd.DataFrame())
    if not setups.empty:
        setup_diffs = build_setup_diffs(setups)
        if not setup_diffs.empty:
            merged_tables["setup/diffs"] = setup_diffs
    passes = merged_tables.get("segments/passes", pd.DataFrame())
    statistics = build_segment_statistics(passes, merged_tables["sessions"])
    merged_tables["summaries/segment_statistics"] = statistics
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
