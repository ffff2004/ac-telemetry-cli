from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

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
STABLE_KEY_COLUMNS: dict[str, tuple[str, ...]] = {
    "sessions": ("session_id",),
    "laps": ("lap_id",),
    "samples": ("session_id", "lap_id", "sample_index"),
    "quality/flags": ("session_id", "lap_id", "code", "sample_start", "sample_end"),
    "setup/normalized": ("setup_id", "section", "parameter"),
    "segments/passes": ("session_id", "lap_id", "segment_id"),
    "events/relations": ("relation_id",),
}
_SETUP_NORMALIZED_COLUMNS = {
    "setup_id",
    "source_hash",
    "section",
    "parameter",
    "value_numeric",
    "value_text",
}
_INTEGRITY_CHECKS = {
    "required_tables",
    "required_columns",
    "identity_columns",
    "stable_key_conflicts",
    "foreign_keys",
    "setup_registry",
    "setup_consistency",
    "source_files",
    "segment_definitions",
}


class ValidationCode(StrEnum):
    MISSING_REQUIRED_COLUMNS = "MISSING_REQUIRED_COLUMNS"
    NULL_IDENTITY_VALUES = "NULL_IDENTITY_VALUES"
    STABLE_KEY_CONFLICT = "STABLE_KEY_CONFLICT"
    DUPLICATE_SETUP_PARAMETER = "DUPLICATE_SETUP_PARAMETER"
    DUPLICATE_RAW_SETUP_PARAMETER = "DUPLICATE_RAW_SETUP_PARAMETER"
    INVALID_SETUP_RAW_REGISTRY = "INVALID_SETUP_RAW_REGISTRY"
    INVALID_SETUP_ID = "INVALID_SETUP_ID"
    INVALID_SETUP_METADATA = "INVALID_SETUP_METADATA"
    SETUP_ID_MISMATCH = "SETUP_ID_MISMATCH"
    MISSING_SETUP_SOURCE_HASH = "MISSING_SETUP_SOURCE_HASH"
    INVALID_RAW_SETUP_CONTENTS = "INVALID_RAW_SETUP_CONTENTS"
    INVALID_SETUP_SECTION = "INVALID_SETUP_SECTION"
    INVALID_RAW_SETUP_SECTION = "INVALID_RAW_SETUP_SECTION"
    INVALID_SETUP_PARAMETER = "INVALID_SETUP_PARAMETER"
    INVALID_SETUP_PROVENANCE = "INVALID_SETUP_PROVENANCE"
    EMPTY_SETUP_IDENTITY = "EMPTY_SETUP_IDENTITY"
    EMPTY_SESSION_SETUP_ID = "EMPTY_SESSION_SETUP_ID"
    UNLINKED_SESSION_SETUP = "UNLINKED_SESSION_SETUP"
    MISSING_NORMALIZED_SETUPS = "MISSING_NORMALIZED_SETUPS"
    MISSING_RAW_SETUP = "MISSING_RAW_SETUP"
    SETUP_SOURCE_HASH_MISMATCH = "SETUP_SOURCE_HASH_MISMATCH"
    SETUP_RAW_NORMALIZED_MISMATCH = "SETUP_RAW_NORMALIZED_MISMATCH"
    UNKNOWN_SESSION_REFERENCE = "UNKNOWN_SESSION_REFERENCE"
    UNKNOWN_LAP_REFERENCE = "UNKNOWN_LAP_REFERENCE"
    UNKNOWN_SESSION_LAP_REFERENCE = "UNKNOWN_SESSION_LAP_REFERENCE"
    SESSION_WITHOUT_LAPS = "SESSION_WITHOUT_LAPS"
    LAP_WITHOUT_SAMPLES = "LAP_WITHOUT_SAMPLES"
    MISSING_EVENT_INDEX = "MISSING_EVENT_INDEX"
    UNKNOWN_EVENT_REFERENCE = "UNKNOWN_EVENT_REFERENCE"
    UNKNOWN_EVENT_RELATION_REFERENCE = "UNKNOWN_EVENT_RELATION_REFERENCE"
    INVALID_SOURCE_FILES = "INVALID_SOURCE_FILES"
    INVALID_SOURCE_FILE = "INVALID_SOURCE_FILE"
    INVALID_SOURCE_FILE_PROVENANCE = "INVALID_SOURCE_FILE_PROVENANCE"
    INVALID_SEGMENT_DEFINITIONS = "INVALID_SEGMENT_DEFINITIONS"
    MISSING_MANIFEST = "MISSING_MANIFEST"
    MANIFEST_READ_ERROR = "MANIFEST_READ_ERROR"
    INVALID_MANIFEST = "INVALID_MANIFEST"
    SCHEMA_VERSION_MISMATCH = "SCHEMA_VERSION_MISMATCH"
    MISSING_TRACK_REFERENCE_ID = "MISSING_TRACK_REFERENCE_ID"
    INVALID_TABLE_MANIFEST = "INVALID_TABLE_MANIFEST"
    INVALID_TABLE_ENTRY = "INVALID_TABLE_ENTRY"
    MISSING_TABLE_FILE = "MISSING_TABLE_FILE"
    ROW_COUNT_MISMATCH = "ROW_COUNT_MISMATCH"
    TABLE_READ_ERROR = "TABLE_READ_ERROR"
    RAW_REGISTRY_READ_ERROR = "RAW_REGISTRY_READ_ERROR"
    SEGMENT_DEFINITIONS_READ_ERROR = "SEGMENT_DEFINITIONS_READ_ERROR"
    INTEGRITY_VALIDATION_ERROR = "INTEGRITY_VALIDATION_ERROR"
    MISSING_TRACK_COLUMNS = "MISSING_TRACK_COLUMNS"
    TRACK_PROGRESS_OUT_OF_RANGE = "TRACK_PROGRESS_OUT_OF_RANGE"
    PATH_DISTANCE_DIMENSION_ERROR = "PATH_DISTANCE_DIMENSION_ERROR"
    INVALID_TRACK_COORDINATES = "INVALID_TRACK_COORDINATES"
    MISSING_TABLES = "MISSING_TABLES"
    UNAVAILABLE_CORE_TABLES = "UNAVAILABLE_CORE_TABLES"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: ValidationCode
    message: str
    check: str
    severity: Literal["error", "warning"] = "error"

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "message": self.message,
            "check": self.check,
            "severity": self.severity,
        }


def _issue(
    issues: list[ValidationIssue],
    code: ValidationCode,
    message: str,
    check: str,
) -> None:
    issues.append(ValidationIssue(code=code, message=message, check=check))


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except TypeError, ValueError:
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _canonical(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
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
    return value


def _token(value: Any) -> str:
    return json.dumps(
        _canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _stable_key_columns(
    logical_name: str, frame: pd.DataFrame
) -> tuple[str, ...] | None:
    if logical_name in STABLE_KEY_COLUMNS:
        return STABLE_KEY_COLUMNS[logical_name]
    if (
        logical_name.startswith("events/")
        and logical_name != "events/relations"
        and "event_id" in frame.columns
    ):
        return ("event_id",)
    return None


def _logical_record(logical_name: str, record: Mapping[str, Any]) -> dict[str, Any]:
    ignored = (
        {"source_file", "source_name", "setup_label"}
        if logical_name in {"sessions", "setup/normalized"}
        else set()
    )
    return {name: value for name, value in record.items() if name not in ignored}


def _has_columns(
    issues: list[ValidationIssue],
    logical_name: str,
    frame: pd.DataFrame,
    columns: set[str],
) -> bool:
    missing = sorted(columns - set(frame.columns))
    if not missing:
        return True
    _issue(
        issues,
        ValidationCode.MISSING_REQUIRED_COLUMNS,
        f"Table {logical_name!r} is missing required columns {missing}",
        "required_columns",
    )
    return False


def _check_identity_columns(
    issues: list[ValidationIssue],
    logical_name: str,
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> None:
    if not _has_columns(issues, logical_name, frame, set(columns)):
        return
    missing = [column for column in columns if bool(frame[column].isna().any())]
    if missing:
        _issue(
            issues,
            ValidationCode.NULL_IDENTITY_VALUES,
            f"Table {logical_name!r} has null values in identity columns {missing}",
            "identity_columns",
        )


def _check_stable_key_conflicts(
    issues: list[ValidationIssue], logical_name: str, frame: pd.DataFrame
) -> None:
    columns = _stable_key_columns(logical_name, frame)
    if columns is None or not _has_columns(issues, logical_name, frame, set(columns)):
        return
    by_key: dict[str, str] = {}
    for record in frame.to_dict(orient="records"):
        key = _token([record[column] for column in columns])
        record_token = _token(_logical_record(logical_name, record))
        previous = by_key.get(key)
        if previous is not None and previous != record_token:
            _issue(
                issues,
                ValidationCode.STABLE_KEY_CONFLICT,
                f"Table {logical_name!r} has conflicting records for key {key}",
                "stable_key_conflicts",
            )
            return
        by_key[key] = record_token


def _canonical_setup_value(value: Any) -> tuple[str, str | bool | None]:
    """Represent setup values faithfully across JSON and Parquet scalar types."""
    if _is_missing(value):
        return ("null", None)
    if isinstance(value, (bool, np.bool_)):
        return ("boolean", bool(value))
    if isinstance(value, (int, float, np.integer, np.floating, Decimal)):
        try:
            number = Decimal(str(value))
        except InvalidOperation:
            return ("value", _token(value))
        if number.is_nan():
            return ("non-finite", "nan")
        if number.is_infinite():
            return ("non-finite", "-infinity" if number.is_signed() else "infinity")
        if number == 0:
            return ("number", "0")
        return ("number", str(number.normalize()))
    return ("value", _token(value))


def _normalized_setup_value(record: Mapping[str, Any]) -> Any:
    numeric = record["value_numeric"]
    if _is_missing(numeric):
        return record["value_text"]
    if isinstance(numeric, (float, np.floating)) and not math.isfinite(float(numeric)):
        expected_text = "nan" if math.isnan(float(numeric)) else str(numeric).lower()
        if str(record["value_text"]).lower() != expected_text:
            return record["value_text"]
    return numeric


def _normalized_setup_values(
    issues: list[ValidationIssue], frame: pd.DataFrame, setup_id: str
) -> dict[tuple[str, str], Any] | None:
    if not _has_columns(issues, "setup/normalized", frame, _SETUP_NORMALIZED_COLUMNS):
        return None
    values: dict[tuple[str, str], Any] = {}
    rows = frame[frame["setup_id"].astype(str) == setup_id]
    for _, row in rows.iterrows():
        record = cast(dict[str, Any], row.to_dict())
        section = record["section"]
        parameter = record["parameter"]
        if not _nonempty_text(section) or not _nonempty_text(parameter):
            continue
        key = (section, parameter)
        if key in values:
            _issue(
                issues,
                ValidationCode.DUPLICATE_SETUP_PARAMETER,
                f"Setup {setup_id!r} duplicates normalized parameter {key!r}",
                "stable_key_conflicts",
            )
            return None
        values[key] = _normalized_setup_value(record)
    return values


def _raw_setup_values(
    issues: list[ValidationIssue], raw: Mapping[str, Any], setup_id: str
) -> dict[tuple[str, str], Any] | None:
    metadata = raw.get(setup_id)
    if not isinstance(metadata, Mapping) or not isinstance(
        metadata.get("raw"), Mapping
    ):
        return None
    values: dict[tuple[str, str], Any] = {}
    for section, parameters in metadata["raw"].items():
        if not _nonempty_text(section) or not isinstance(parameters, Mapping):
            continue
        for parameter, value in parameters.items():
            if not _nonempty_text(parameter):
                continue
            key = (section, parameter)
            if key in values:
                _issue(
                    issues,
                    ValidationCode.DUPLICATE_RAW_SETUP_PARAMETER,
                    f"Setup {setup_id!r} duplicates raw parameter {key!r}",
                    "setup_registry",
                )
                return None
            values[key] = value
    return values


def _check_raw_registry(
    issues: list[ValidationIssue], raw: Any
) -> Mapping[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        _issue(
            issues,
            ValidationCode.INVALID_SETUP_RAW_REGISTRY,
            "setup/raw.json must contain an object",
            "setup_registry",
        )
        return None
    for setup_id, metadata in raw.items():
        if not _nonempty_text(setup_id):
            _issue(
                issues,
                ValidationCode.INVALID_SETUP_ID,
                "Setup registry contains an empty setup_id",
                "setup_registry",
            )
            continue
        if not isinstance(metadata, Mapping):
            _issue(
                issues,
                ValidationCode.INVALID_SETUP_METADATA,
                f"Setup {setup_id!r} metadata must be an object",
                "setup_registry",
            )
            continue
        if metadata.get("setup_id") != setup_id:
            _issue(
                issues,
                ValidationCode.SETUP_ID_MISMATCH,
                f"Setup registry key and setup_id disagree for {setup_id!r}",
                "setup_registry",
            )
        if not _nonempty_text(metadata.get("source_hash")):
            _issue(
                issues,
                ValidationCode.MISSING_SETUP_SOURCE_HASH,
                f"Setup {setup_id!r} lacks a source_hash",
                "setup_registry",
            )
        contents = metadata.get("raw")
        if not isinstance(contents, Mapping):
            _issue(
                issues,
                ValidationCode.INVALID_RAW_SETUP_CONTENTS,
                f"Setup {setup_id!r} raw metadata must be an object",
                "setup_registry",
            )
        else:
            for section, parameters in contents.items():
                if not _nonempty_text(section):
                    _issue(
                        issues,
                        ValidationCode.INVALID_SETUP_SECTION,
                        f"Setup {setup_id!r} contains an empty raw section",
                        "setup_registry",
                    )
                if not isinstance(parameters, Mapping):
                    _issue(
                        issues,
                        ValidationCode.INVALID_RAW_SETUP_SECTION,
                        f"Setup {setup_id!r} section {section!r} is not an object",
                        "setup_registry",
                    )
                    continue
                for parameter in parameters:
                    if not _nonempty_text(parameter):
                        _issue(
                            issues,
                            ValidationCode.INVALID_SETUP_PARAMETER,
                            f"Setup {setup_id!r} contains an empty raw parameter",
                            "setup_registry",
                        )
        for plural, singular in (
            ("source_files", "source_file"),
            ("setup_labels", "setup_label"),
        ):
            if plural in metadata and not isinstance(metadata[plural], list):
                _issue(
                    issues,
                    ValidationCode.INVALID_SETUP_PROVENANCE,
                    f"Setup {setup_id!r} {plural} must be a list",
                    "setup_registry",
                )
            if (
                plural not in metadata
                and singular in metadata
                and metadata[singular] is not None
                and not isinstance(metadata[singular], str)
            ):
                _issue(
                    issues,
                    ValidationCode.INVALID_SETUP_PROVENANCE,
                    f"Setup {setup_id!r} {singular} must be a string",
                    "setup_registry",
                )
    return raw


def _check_setup_consistency(
    issues: list[ValidationIssue],
    tables: Mapping[str, pd.DataFrame],
    raw: Mapping[str, Any] | None,
) -> None:
    sessions = tables.get("sessions")
    normalized = tables.get("setup/normalized")
    if (
        normalized is not None
        and (not normalized.empty or len(normalized.columns))
        and _has_columns(
            issues, "setup/normalized", normalized, _SETUP_NORMALIZED_COLUMNS
        )
    ):
        _check_identity_columns(
            issues,
            "setup/normalized",
            normalized,
            ("setup_id", "section", "parameter"),
        )
        for column in ("setup_id", "section", "parameter"):
            empty = normalized[column].map(lambda value: not _nonempty_text(value))
            if bool(empty.any()):
                _issue(
                    issues,
                    ValidationCode.EMPTY_SETUP_IDENTITY,
                    f"setup/normalized contains an empty {column}",
                    "setup_consistency",
                )
        normalized_ids = {
            str(value) for value in normalized["setup_id"] if _nonempty_text(value)
        }
    else:
        normalized_ids = set()

    session_setup_ids: set[str] = set()
    if sessions is not None and "setup_id" in sessions:
        session_setup_ids = {
            str(value) for value in sessions["setup_id"] if not _is_missing(value)
        }
        if any(not _nonempty_text(value) for value in session_setup_ids):
            _issue(
                issues,
                ValidationCode.EMPTY_SESSION_SETUP_ID,
                "sessions contains an empty setup_id",
                "setup_consistency",
            )
    raw_ids = set(raw) if raw is not None else set()
    for setup_id in sorted(session_setup_ids):
        if setup_id not in normalized_ids or setup_id not in raw_ids:
            _issue(
                issues,
                ValidationCode.UNLINKED_SESSION_SETUP,
                f"Session setup_id {setup_id!r} is not linked to normalized and raw setup data",
                "setup_consistency",
            )
    if normalized is None and session_setup_ids:
        _issue(
            issues,
            ValidationCode.MISSING_NORMALIZED_SETUPS,
            "sessions reference setup_id values but setup/normalized is absent",
            "setup_consistency",
        )
    if normalized is None:
        return
    for setup_id in sorted(normalized_ids):
        if setup_id not in raw_ids:
            _issue(
                issues,
                ValidationCode.MISSING_RAW_SETUP,
                f"Normalized setup {setup_id!r} has no raw registry metadata",
                "setup_consistency",
            )
            continue
        if raw is None or not isinstance(raw.get(setup_id), Mapping):
            continue
        normalized_rows = normalized[normalized["setup_id"].astype(str) == setup_id]
        source_hashes = {_token(value) for value in normalized_rows["source_hash"]}
        if source_hashes != {_token(raw[setup_id].get("source_hash"))}:
            _issue(
                issues,
                ValidationCode.SETUP_SOURCE_HASH_MISMATCH,
                f"Setup {setup_id!r} source_hash disagrees between raw registry and normalized table",
                "setup_consistency",
            )
        normalized_values = _normalized_setup_values(issues, normalized, setup_id)
        raw_values = _raw_setup_values(issues, raw, setup_id)
        if normalized_values is None or raw_values is None:
            continue
        normalized_semantic = {
            key: _canonical_setup_value(value)
            for key, value in normalized_values.items()
        }
        raw_semantic = {
            key: _canonical_setup_value(value) for key, value in raw_values.items()
        }
        if normalized_semantic != raw_semantic:
            _issue(
                issues,
                ValidationCode.SETUP_RAW_NORMALIZED_MISMATCH,
                f"Setup {setup_id!r} raw registry and normalized parameters disagree",
                "setup_consistency",
            )


def _string_values(frame: pd.DataFrame, column: str) -> set[str]:
    return {str(value) for value in frame[column] if not _is_missing(value)}


def _check_foreign_keys(
    issues: list[ValidationIssue], tables: Mapping[str, pd.DataFrame]
) -> None:
    sessions = tables.get("sessions")
    laps = tables.get("laps")
    samples = tables.get("samples")
    if sessions is None or laps is None or samples is None:
        return
    if not _has_columns(issues, "sessions", sessions, {"session_id"}):
        return
    if not _has_columns(issues, "laps", laps, {"session_id", "lap_id"}):
        return
    if not _has_columns(issues, "samples", samples, {"session_id", "lap_id"}):
        return
    session_ids = _string_values(sessions, "session_id")
    lap_ids = _string_values(laps, "lap_id")
    lap_pairs = set(
        zip(
            laps["session_id"].dropna().astype(str),
            laps["lap_id"].dropna().astype(str),
            strict=True,
        )
    )
    for logical_name, frame in tables.items():
        if (
            "session_id" in frame
            and not _string_values(frame, "session_id") <= session_ids
        ):
            _issue(
                issues,
                ValidationCode.UNKNOWN_SESSION_REFERENCE,
                f"Table {logical_name!r} references an unknown session_id",
                "foreign_keys",
            )
        if "lap_id" in frame and not _string_values(frame, "lap_id") <= lap_ids:
            _issue(
                issues,
                ValidationCode.UNKNOWN_LAP_REFERENCE,
                f"Table {logical_name!r} references an unknown lap_id",
                "foreign_keys",
            )
        if {"session_id", "lap_id"} <= set(frame.columns):
            pairs = set(
                zip(
                    frame["session_id"].dropna().astype(str),
                    frame["lap_id"].dropna().astype(str),
                    strict=True,
                )
            )
            if not pairs <= lap_pairs:
                _issue(
                    issues,
                    ValidationCode.UNKNOWN_SESSION_LAP_REFERENCE,
                    f"Table {logical_name!r} references an unknown session/lap pair",
                    "foreign_keys",
                )
    sessions_with_laps = _string_values(laps, "session_id")
    if not session_ids <= sessions_with_laps:
        _issue(
            issues,
            ValidationCode.SESSION_WITHOUT_LAPS,
            "Sessions without laps are not valid datasets",
            "foreign_keys",
        )
    sample_pairs = set(
        zip(
            samples["session_id"].dropna().astype(str),
            samples["lap_id"].dropna().astype(str),
            strict=True,
        )
    )
    if not lap_pairs <= sample_pairs:
        _issue(
            issues,
            ValidationCode.LAP_WITHOUT_SAMPLES,
            "Laps without samples are not valid datasets",
            "foreign_keys",
        )
    event_tables = [name for name in tables if name.startswith("events/")]
    event_index = tables.get("events/index")
    if event_tables and event_index is None:
        _issue(
            issues,
            ValidationCode.MISSING_EVENT_INDEX,
            "Event fact tables require events/index",
            "foreign_keys",
        )
        return
    if event_index is None or not _has_columns(
        issues, "events/index", event_index, {"event_id"}
    ):
        return
    event_ids = _string_values(event_index, "event_id")
    for logical_name, frame in tables.items():
        if logical_name == "events/index":
            continue
        for column in (
            "event_id",
            "parent_braking_event_id",
            "parent_throttle_event_id",
        ):
            if column in frame and not _string_values(frame, column) <= event_ids:
                _issue(
                    issues,
                    ValidationCode.UNKNOWN_EVENT_REFERENCE,
                    f"Table {logical_name!r} references an unknown event_id",
                    "foreign_keys",
                )
    relations = tables.get("events/relations")
    if relations is not None and _has_columns(
        issues, "events/relations", relations, {"event_id_a", "event_id_b"}
    ):
        for column in ("event_id_a", "event_id_b"):
            if not _string_values(relations, column) <= event_ids:
                _issue(
                    issues,
                    ValidationCode.UNKNOWN_EVENT_RELATION_REFERENCE,
                    "Event relations reference an unknown event_id",
                    "foreign_keys",
                )


def segment_definition_issues(definitions: Any) -> list[str]:
    if not isinstance(definitions, Mapping):
        return ["Segment definitions must contain an object"]
    coordinate = definitions.get("coordinate", "track_s_m")
    if coordinate not in {"track_s_m", "track_progress"}:
        return [f"Unsupported segment coordinate {coordinate!r}"]
    parents = definitions.get("segments")
    if not isinstance(parents, list):
        return ["Segment definition must contain a segments list"]
    messages: list[str] = []
    for parent in parents:
        if not isinstance(parent, Mapping):
            messages.append("Each segment definition must be an object")
            continue
        children = parent.get("subsegments", [])
        if not isinstance(children, list):
            messages.append("Segment subsegments must be a list")
            continue
        for definition in [parent, *children]:
            if not isinstance(definition, Mapping):
                messages.append("Each segment definition must be an object")
                continue
            try:
                start = float(definition["start"])
                end = float(definition["end"])
            except KeyError, TypeError, ValueError:
                messages.append(
                    "Segment definitions require numeric start and end values"
                )
                continue
            if not math.isfinite(start) or not math.isfinite(end):
                messages.append("Segment start and end values must be finite")
            elif end < start:
                identifier = definition.get("id", "<unnamed>")
                messages.append(
                    f"Segment {identifier!r} crosses the lap boundary (start={start:g}, end={end:g})"
                )
    return messages


def _check_source_files(
    issues: list[ValidationIssue], manifest: Mapping[str, Any]
) -> None:
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list):
        _issue(
            issues,
            ValidationCode.INVALID_SOURCE_FILES,
            "manifest source_files must be a list",
            "source_files",
        )
        return
    for index, entry in enumerate(source_files):
        if not isinstance(entry, Mapping):
            _issue(
                issues,
                ValidationCode.INVALID_SOURCE_FILE,
                f"manifest source_files[{index}] must be an object",
                "source_files",
            )
            continue
        for field in ("sha256", "type"):
            if not _nonempty_text(entry.get(field)):
                _issue(
                    issues,
                    ValidationCode.INVALID_SOURCE_FILE,
                    f"manifest source_files[{index}] must contain a non-empty {field}",
                    "source_files",
                )
        for field in ("display_paths", "display_names"):
            if field in entry and not isinstance(entry[field], list):
                _issue(
                    issues,
                    ValidationCode.INVALID_SOURCE_FILE_PROVENANCE,
                    f"manifest source_files[{index}] {field} must be a list",
                    "source_files",
                )


def dataset_integrity_issues(
    manifest: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    raw_registry: Any,
    segment_definitions: Any | None,
) -> list[ValidationIssue]:
    """Return structured single-dataset integrity failures without raising.

    This intentionally excludes cross-input merge compatibility, provenance,
    generated-output, and publication concerns.
    """
    issues: list[ValidationIssue] = []
    table_manifest = manifest.get("tables")
    declared_tables = (
        set(table_manifest) if isinstance(table_manifest, Mapping) else set()
    )
    missing = sorted(REQUIRED_LOGICAL_TABLES - declared_tables)
    unavailable = sorted(REQUIRED_LOGICAL_TABLES - set(tables))
    if missing:
        _issue(
            issues,
            ValidationCode.MISSING_TABLES,
            f"Missing tables: {missing}",
            "required_tables",
        )
    if unavailable:
        _issue(
            issues,
            ValidationCode.UNAVAILABLE_CORE_TABLES,
            f"Core tables could not be loaded: {unavailable}",
            "required_tables",
        )
    required_columns = {
        "sessions": {"session_id"},
        "laps": {"session_id", "lap_id"},
        "samples": {"session_id", "lap_id", "sample_index"},
        "setup/normalized": _SETUP_NORMALIZED_COLUMNS,
        "segments/passes": {"session_id", "lap_id", "segment_id"},
        "events/index": {"event_id"},
        "events/relations": {"relation_id", "event_id_a", "event_id_b"},
    }
    for logical_name, frame in tables.items():
        columns = required_columns.get(logical_name)
        if logical_name.startswith("events/") and logical_name not in {
            "events/index",
            "events/relations",
        }:
            columns = {"event_id"}
        is_core_table = logical_name in REQUIRED_LOGICAL_TABLES
        has_schema = not frame.empty or bool(len(frame.columns))
        if columns is not None and (is_core_table or has_schema):
            _has_columns(issues, logical_name, frame, columns)
        keys = _stable_key_columns(logical_name, frame)
        if keys is not None and (is_core_table or has_schema):
            _check_identity_columns(issues, logical_name, frame, keys)
            _check_stable_key_conflicts(issues, logical_name, frame)
    raw = _check_raw_registry(issues, raw_registry)
    _check_setup_consistency(issues, tables, raw)
    _check_foreign_keys(issues, tables)
    _check_source_files(issues, manifest)
    if segment_definitions is not None:
        for message in segment_definition_issues(segment_definitions):
            _issue(
                issues,
                ValidationCode.INVALID_SEGMENT_DEFINITIONS,
                message,
                "segment_definitions",
            )
    return issues


def _add_issue_checks(
    checks: dict[str, Any],
    warnings: list[dict[str, Any]],
    issues: list[ValidationIssue],
) -> None:
    for check in _INTEGRITY_CHECKS:
        checks.setdefault(check, "pass")
    for issue in issues:
        checks[issue.check] = "fail" if issue.severity == "error" else "warning"
        warnings.append(issue.as_dict())


def validate_dataset(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    checks: dict[str, Any] = {}
    warnings: list[dict[str, Any]] = []
    if not manifest_path.exists():
        return {
            "status": "error",
            "checks": {"manifest": "fail"},
            "warnings": [
                {
                    "code": ValidationCode.MISSING_MANIFEST.value,
                    "message": "manifest.json not found",
                }
            ],
        }
    try:
        manifest = json_load(manifest_path)
    except Exception as exc:  # validation must report, not crash
        return {
            "status": "error",
            "checks": {"manifest": "fail"},
            "warnings": [
                {
                    "code": ValidationCode.MANIFEST_READ_ERROR.value,
                    "message": str(exc),
                }
            ],
        }
    if not isinstance(manifest, Mapping):
        return {
            "status": "error",
            "checks": {"manifest": "fail"},
            "warnings": [
                {
                    "code": ValidationCode.INVALID_MANIFEST.value,
                    "message": "manifest.json must contain an object",
                }
            ],
        }
    checks["manifest"] = "pass"
    checks["schema_version"] = (
        "pass" if manifest.get("schema_version") == DATASET_SCHEMA_VERSION else "fail"
    )
    if checks["schema_version"] == "fail":
        warnings.append(
            {
                "code": ValidationCode.SCHEMA_VERSION_MISMATCH.value,
                "message": f"Expected schema {DATASET_SCHEMA_VERSION}, found {manifest.get('schema_version')}",
            }
        )
    if not manifest.get("track_reference_id"):
        checks["track_reference_id"] = "fail"
        warnings.append(
            {
                "code": ValidationCode.MISSING_TRACK_REFERENCE_ID.value,
                "message": "manifest has no track_reference_id",
            }
        )
    else:
        checks["track_reference_id"] = "pass"

    table_map = manifest.get("tables")
    if not isinstance(table_map, Mapping):
        table_map = {}
        warnings.append(
            {
                "code": ValidationCode.INVALID_TABLE_MANIFEST.value,
                "message": "manifest tables must be an object",
            }
        )
        checks["required_tables"] = "fail"
    storage = DatasetStorage(root)
    loaded: dict[str, pd.DataFrame] = {}
    for logical, info in table_map.items():
        if (
            not isinstance(logical, str)
            or not isinstance(info, Mapping)
            or not isinstance(info.get("path"), str)
        ):
            checks[f"table:{logical}"] = "fail"
            warnings.append(
                {
                    "code": ValidationCode.INVALID_TABLE_ENTRY.value,
                    "message": f"Invalid manifest entry for table {logical!r}",
                }
            )
            continue
        path = root / info["path"]
        if not path.exists():
            checks[f"table:{logical}"] = "fail"
            warnings.append(
                {
                    "code": ValidationCode.MISSING_TABLE_FILE.value,
                    "message": f"{logical}: {path}",
                }
            )
            continue
        try:
            frame = storage.read(info["path"])
            loaded[logical] = frame
            checks[f"table:{logical}"] = "pass"
            if len(frame) != info.get("rows"):
                warnings.append(
                    {
                        "code": ValidationCode.ROW_COUNT_MISMATCH.value,
                        "message": f"{logical}: manifest={info.get('rows')} actual={len(frame)}",
                    }
                )
        except Exception as exc:  # validation must report, not crash
            checks[f"table:{logical}"] = "fail"
            warnings.append(
                {
                    "code": ValidationCode.TABLE_READ_ERROR.value,
                    "message": f"{logical}: {exc}",
                }
            )

    raw: Any = None
    raw_path = root / "setup" / "raw.json"
    if raw_path.exists():
        try:
            raw = json_load(raw_path)
        except Exception as exc:  # validation must report, not crash
            warnings.append(
                {
                    "code": ValidationCode.RAW_REGISTRY_READ_ERROR.value,
                    "message": str(exc),
                }
            )
            checks["setup_registry"] = "fail"
    definitions: Any | None = None
    definitions_path = root / "segments" / "definitions.json"
    if definitions_path.exists():
        try:
            definitions = json_load(definitions_path)
        except Exception as exc:  # validation must report, not crash
            warnings.append(
                {
                    "code": ValidationCode.SEGMENT_DEFINITIONS_READ_ERROR.value,
                    "message": str(exc),
                }
            )
            checks["segment_definitions"] = "fail"

    try:
        _add_issue_checks(
            checks,
            warnings,
            dataset_integrity_issues(manifest, loaded, raw, definitions),
        )
    except Exception as exc:  # validation must report, not crash
        checks["dataset_integrity"] = "fail"
        warnings.append(
            {
                "code": ValidationCode.INTEGRITY_VALIDATION_ERROR.value,
                "message": str(exc),
            }
        )

    samples = loaded.get("samples")
    if samples is not None:
        missing_columns = sorted(REQUIRED_SAMPLE_COLUMNS - set(samples.columns))
        checks["sample_coordinate_columns"] = "pass" if not missing_columns else "fail"
        if missing_columns:
            warnings.append(
                {
                    "code": ValidationCode.MISSING_TRACK_COLUMNS.value,
                    "message": f"samples missing track columns: {missing_columns}",
                }
            )
        else:
            try:
                progress = samples["track_progress"].to_numpy(float)
                finite = np.isfinite(progress)
                in_range = bool(
                    (~finite | ((progress >= 0.0) & (progress < 1.0 + 1e-9))).all()
                )
                checks["track_progress_range"] = "pass" if in_range else "fail"
                if not in_range:
                    warnings.append(
                        {
                            "code": ValidationCode.TRACK_PROGRESS_OUT_OF_RANGE.value,
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
                            "code": ValidationCode.PATH_DISTANCE_DIMENSION_ERROR.value,
                            "message": "3D path distance must not be shorter than 2D path distance",
                        }
                    )
            except (TypeError, ValueError) as exc:
                checks["sample_coordinate_values"] = "fail"
                warnings.append(
                    {
                        "code": ValidationCode.INVALID_TRACK_COORDINATES.value,
                        "message": str(exc),
                    }
                )

    status = (
        "error"
        if any(value == "fail" for value in checks.values())
        else ("warning" if warnings else "ok")
    )
    return {"status": status, "checks": checks, "warnings": warnings}
