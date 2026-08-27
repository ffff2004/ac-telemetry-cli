import configparser
from pathlib import Path
from typing import Any

import pandas as pd

from .contract_types import ColumnAvailability, ColumnSpec, MergeMode, TableSpec
from .util import sha256_file, stable_id

CATEGORY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("WING", "aero"),
    ("INTERNAL_GEAR", "gearing"),
    ("FINAL_RATIO", "gearing"),
    ("DIFF", "differential"),
    ("BRAKE", "brakes"),
    ("FRONT_BIAS", "brakes"),
    ("TRACTION_CONTROL", "driver_aids"),
    ("ABS", "driver_aids"),
    ("FUEL", "fuel"),
    ("TYRES", "tyres"),
    ("PRESSURE", "tyres"),
    ("CAMBER", "alignment"),
    ("TOE", "alignment"),
    ("ARB", "suspension"),
    ("SPRING", "suspension"),
    ("DAMP", "dampers"),
    ("BUMP_STOP", "suspension"),
    ("PACKER", "suspension"),
    ("ROD_LENGTH", "ride_height"),
)

SETUP_TABLE_SPECS = (
    TableSpec(
        "setup/normalized",
        (
            ColumnSpec(
                "setup_id",
                ColumnAvailability.REQUIRED,
                False,
                "Stable identifier for the setup file content.",
            ),
            ColumnSpec(
                "setup_label",
                ColumnAvailability.OPTIONAL,
                False,
                "Human-readable setup label supplied with the input.",
            ),
            ColumnSpec(
                "source_file",
                ColumnAvailability.OPTIONAL,
                False,
                "Original setup INI filename.",
            ),
            ColumnSpec(
                "source_hash",
                ColumnAvailability.REQUIRED,
                False,
                "SHA-256 digest of the source setup file.",
            ),
            ColumnSpec(
                "section",
                ColumnAvailability.REQUIRED,
                False,
                "INI section containing the setup parameter.",
            ),
            ColumnSpec(
                "parameter",
                ColumnAvailability.REQUIRED,
                False,
                "Parameter name within the INI section.",
            ),
            ColumnSpec(
                "raw_value",
                ColumnAvailability.OPTIONAL,
                False,
                "Unmodified parameter value read from the INI file.",
            ),
            ColumnSpec(
                "value_numeric",
                ColumnAvailability.REQUIRED,
                True,
                "Numeric interpretation of the parameter value, when applicable.",
            ),
            ColumnSpec(
                "value_text",
                ColumnAvailability.REQUIRED,
                False,
                "Canonical text representation of the parsed parameter value.",
            ),
            ColumnSpec(
                "category",
                ColumnAvailability.REQUIRED,
                False,
                "Functional setup category inferred from the INI section.",
            ),
        ),
        ("setup_id", "section", "parameter"),
        False,
        MergeMode.KEYED,
        ignored_identity_columns=frozenset({"setup_label", "source_file"}),
    ),
    TableSpec(
        "setup/diffs",
        (
            ColumnSpec(
                "base_setup_id",
                ColumnAvailability.OPTIONAL,
                False,
                "Setup identifier used as the comparison baseline.",
            ),
            ColumnSpec(
                "comparison_setup_id",
                ColumnAvailability.OPTIONAL,
                False,
                "Setup identifier compared with the baseline.",
            ),
            ColumnSpec(
                "section",
                ColumnAvailability.OPTIONAL,
                False,
                "INI section containing the compared parameter.",
            ),
            ColumnSpec(
                "parameter",
                ColumnAvailability.OPTIONAL,
                False,
                "Parameter name compared within the INI section.",
            ),
            ColumnSpec(
                "category",
                ColumnAvailability.OPTIONAL,
                False,
                "Functional setup category inferred from the INI section.",
            ),
            ColumnSpec(
                "base_value",
                ColumnAvailability.OPTIONAL,
                True,
                "Parsed text value in the baseline setup, if that parameter exists.",
            ),
            ColumnSpec(
                "comparison_value",
                ColumnAvailability.OPTIONAL,
                True,
                "Parsed text value in the comparison setup, if that parameter exists.",
            ),
            ColumnSpec(
                "absolute_numeric_change",
                ColumnAvailability.OPTIONAL,
                True,
                "Comparison numeric value minus baseline numeric value, when both exist.",
            ),
            ColumnSpec(
                "values_equal",
                ColumnAvailability.OPTIONAL,
                False,
                "Whether both setups contain equal parsed text values for the parameter.",
            ),
        ),
        None,
        False,
        MergeMode.REBUILD,
        rebuild_from=("setup/normalized",),
        omit_if_empty=True,
    ),
)


def _coerce(value: str) -> Any:
    text = value.strip()
    for converter in (int, float):
        try:
            return converter(text)
        except ValueError:
            pass
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return text


def _category(section: str) -> str:
    upper = section.upper()
    for prefix, category in CATEGORY_PREFIXES:
        if upper.startswith(prefix):
            return category
    return "other"


def parse_setup_file(
    path: Path, setup_label: str | None = None
) -> tuple[dict[str, Any], pd.DataFrame]:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    parser.read(path, encoding="utf-8-sig")
    file_hash = sha256_file(path)
    setup_id = stable_id(path.name, file_hash)
    rows: list[dict[str, Any]] = []
    raw: dict[str, dict[str, Any]] = {}

    for section in parser.sections():
        raw[section] = {}
        for key, value in parser.items(section):
            parsed = _coerce(value)
            raw[section][key] = parsed
            rows.append(
                {
                    "setup_id": setup_id,
                    "setup_label": setup_label or path.stem,
                    "source_file": path.name,
                    "source_hash": file_hash,
                    "section": section,
                    "parameter": key,
                    "raw_value": value,
                    "value_numeric": parsed
                    if isinstance(parsed, (int, float))
                    else None,
                    "value_text": str(parsed),
                    "category": _category(section),
                }
            )

    metadata = {
        "setup_id": setup_id,
        "setup_label": setup_label or path.stem,
        "source_file": str(path),
        "source_hash": file_hash,
        "raw": raw,
        "car_model": raw.get("CAR", {}).get("MODEL"),
    }
    return metadata, pd.DataFrame(rows)


def parse_setup_bundle(
    ini_path: Path | None, setup_label: str | None = None
) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    if ini_path is None:
        return None, pd.DataFrame()
    return parse_setup_file(ini_path, setup_label)


def build_setup_diffs(table: pd.DataFrame) -> pd.DataFrame:
    """Build pairwise parameter differences without pretending UI indices are physical units."""
    if table.empty or table["setup_id"].nunique() < 2:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    setup_ids = list(table["setup_id"].drop_duplicates())
    indexed = {
        setup_id: group.set_index(["section", "parameter"], drop=False)
        for setup_id, group in table.groupby("setup_id", sort=False)
    }
    for i, base_id in enumerate(setup_ids):
        for comparison_id in setup_ids[i + 1 :]:
            base = indexed[base_id]
            comparison = indexed[comparison_id]
            keys = sorted(set(base.index) | set(comparison.index))
            for key in keys:
                base_row = base.loc[key] if key in base.index else None
                comp_row = comparison.loc[key] if key in comparison.index else None
                if isinstance(base_row, pd.DataFrame):
                    base_row = base_row.iloc[0]
                if isinstance(comp_row, pd.DataFrame):
                    comp_row = comp_row.iloc[0]
                base_num = base_row["value_numeric"] if base_row is not None else None
                comp_num = comp_row["value_numeric"] if comp_row is not None else None
                absolute = (
                    float(comp_num) - float(base_num)
                    if bool(pd.notna(base_num)) and bool(pd.notna(comp_num))
                    else None
                )
                if comp_row is not None:
                    category = comp_row["category"]
                elif base_row is not None:
                    category = base_row["category"]
                else:
                    continue
                rows.append(
                    {
                        "base_setup_id": base_id,
                        "comparison_setup_id": comparison_id,
                        "section": key[0],
                        "parameter": key[1],
                        "category": category,
                        "base_value": base_row["value_text"]
                        if base_row is not None
                        else None,
                        "comparison_value": comp_row["value_text"]
                        if comp_row is not None
                        else None,
                        "absolute_numeric_change": absolute,
                        "values_equal": (
                            str(base_row["value_text"]) == str(comp_row["value_text"])
                            if base_row is not None and comp_row is not None
                            else False
                        ),
                    }
                )
    return pd.DataFrame(rows)
