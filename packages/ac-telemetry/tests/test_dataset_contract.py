from pathlib import Path

import pandas as pd
import pytest
from ac_telemetry.contract_types import (
    ColumnAvailability,
    ColumnSpec,
    DatasetContract,
    ForeignKey,
    MergeMode,
    TableSpec,
)
from ac_telemetry.dataset_contract import DATASET_CONTRACT
from ac_telemetry.setup_parser import build_setup_diffs, parse_setup_file
from ac_telemetry.summary import build_segment_statistics


def _column(name: str) -> ColumnSpec:
    return ColumnSpec(name, ColumnAvailability.REQUIRED, False, name)


def test_producer_contract_is_complete_and_valid() -> None:
    DATASET_CONTRACT.validate_definition()
    assert DATASET_CONTRACT.table("sessions") is not None
    assert DATASET_CONTRACT.table("events/index") is not None
    assert DATASET_CONTRACT.table("track/reference") is not None


@pytest.mark.parametrize(
    "contract",
    [
        DatasetContract(
            (
                TableSpec("a", (_column("id"),), ("id",), False, MergeMode.KEYED),
                TableSpec("a", (_column("id"),), ("id",), False, MergeMode.KEYED),
            )
        ),
        DatasetContract(
            (TableSpec("bad//name", (_column("id"),), ("id",), False, MergeMode.KEYED),)
        ),
        DatasetContract(
            (TableSpec("a", (_column("id"),), ("missing",), False, MergeMode.KEYED),)
        ),
        DatasetContract(
            (
                TableSpec(
                    "a",
                    (_column("id"),),
                    ("id",),
                    False,
                    MergeMode.KEYED,
                    (ForeignKey(("id",), "missing", ("id",)),),
                ),
            )
        ),
    ],
)
def test_contract_definition_rejects_invalid_declarations(
    contract: DatasetContract,
) -> None:
    with pytest.raises(ValueError):
        contract.validate_definition()


def test_merge_plan_preserves_declared_order_and_rejects_unknown_tables() -> None:
    names = ("samples", "sessions", "events/index")
    assert [table.name for table in DATASET_CONTRACT.merge_plan(names)] == [
        "sessions",
        "samples",
        "events/index",
    ]
    with pytest.raises(ValueError, match="Unsupported merge tables"):
        DATASET_CONTRACT.merge_plan(("not/a/table",))


def test_merge_plan_selects_rebuilds_from_declared_prerequisites() -> None:
    setup_plan = DATASET_CONTRACT.merge_plan(("setup/normalized",))
    assert [table.name for table in setup_plan] == [
        "setup/normalized",
        "setup/diffs",
    ]

    without_passes = DATASET_CONTRACT.merge_plan(("sessions",))
    assert "summaries/segment_statistics" not in {
        table.name for table in without_passes
    }


def test_column_nullability_is_independent_from_column_availability() -> None:
    sessions = DATASET_CONTRACT.table("sessions")
    assert sessions is not None
    columns = {column.name: column for column in sessions.columns}

    assert columns["session_id"].availability is ColumnAvailability.REQUIRED
    assert not columns["session_id"].nullable
    assert columns["setup_id"].availability is ColumnAvailability.REQUIRED
    assert columns["setup_id"].nullable

    normalized = DATASET_CONTRACT.table("setup/normalized")
    assert normalized is not None
    normalized_columns = {column.name: column for column in normalized.columns}
    assert normalized_columns["value_numeric"].nullable


def test_empty_segment_statistics_keep_the_serialized_empty_layout() -> None:
    sessions = pd.DataFrame(columns=["session_id", "setup_id"])
    statistics = build_segment_statistics(pd.DataFrame(), sessions)

    assert statistics.empty
    assert list(statistics.columns) == []


def test_setup_producers_match_their_declared_public_columns(tmp_path: Path) -> None:
    first = tmp_path / "first.ini"
    first.write_text("[GEARS]\nFINAL=3.0\n", encoding="utf-8")
    second = tmp_path / "second.ini"
    second.write_text("[GEARS]\nFINAL=3.5\n", encoding="utf-8")
    _, first_table = parse_setup_file(first)
    _, second_table = parse_setup_file(second)
    normalized = first_table
    normalized_spec = DATASET_CONTRACT.table("setup/normalized")
    assert normalized_spec is not None
    assert list(normalized.columns) == [
        column.name for column in normalized_spec.columns
    ]

    diffs = build_setup_diffs(pd.concat([first_table, second_table], ignore_index=True))
    diffs_spec = DATASET_CONTRACT.table("setup/diffs")
    assert diffs_spec is not None
    assert list(diffs.columns) == [column.name for column in diffs_spec.columns]
