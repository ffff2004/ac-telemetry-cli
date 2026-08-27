import json
from pathlib import Path

import pandas as pd
import pytest
from ac_telemetry.cli import main
from ac_telemetry.manifest import DATASET_SCHEMA_VERSION, table_manifest
from ac_telemetry.merge import merge_datasets
from ac_telemetry.storage import DatasetStorage, TableRef
from ac_telemetry.util import json_dump, json_load
from ac_telemetry.validation import ValidationCode, validate_dataset


def _passes(session_id: str, lap_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "session_id": session_id,
                "lap_id": lap_id,
                "segment_id": "t1",
                "segment_name": "Turn 1",
                "valid_for_comparison": True,
                "segment_time_s": 10.0,
                "entry_speed_kmh": 200.0,
                "minimum_speed_kmh": 100.0,
                "exit_speed_kmh": 160.0,
                "brake_onset_track_s_m": 10.0,
                "full_throttle_commit_track_s_m": 90.0,
                "coasting_time_s": 1.0,
            }
        ]
    )


def _write_dataset(
    root: Path,
    *,
    session_id: str,
    setup_label: str,
    setup_source: str,
    processing_options: dict[str, object] | None = None,
    schema_version: str = DATASET_SCHEMA_VERSION,
    include_quality: bool = True,
    track_value: int = 1,
    setup_id: str = "setup-a",
    setup_hash: str = "setup-content",
    setup_value: object = 3.0,
    normalized_setup_value: object | None = None,
    source_hash: str | None = None,
    source_path: str | None = None,
    source_name: str | None = None,
) -> None:
    lap_id = f"{session_id}-lap"
    normalized_value = (
        setup_value if normalized_setup_value is None else normalized_setup_value
    )
    storage = DatasetStorage(root)
    frames = {
        "sessions": pd.DataFrame(
            [
                {
                    "session_id": session_id,
                    "setup_id": setup_id,
                    "source_file": source_path or f"/{session_id}.acreplay",
                    "source_name": source_name or f"{session_id}.acreplay",
                }
            ]
        ),
        "laps": pd.DataFrame(
            [
                {
                    "session_id": session_id,
                    "lap_id": lap_id,
                    "is_complete": True,
                    "is_valid": True,
                    "lap_time_s": 60.0,
                    "source_lap_number": 1,
                }
            ]
        ),
        "samples": pd.DataFrame(
            [
                {
                    "session_id": session_id,
                    "lap_id": lap_id,
                    "sample_index": 0,
                    "track_s_m": 0.0,
                    "track_progress": 0.0,
                    "track_projection_distance_3d_m": 0.0,
                    "lateral_offset_m": 0.0,
                    "path_distance_2d_m": 0.0,
                    "path_distance_3d_m": 0.0,
                }
            ]
        ),
        "track/reference": pd.DataFrame({"point": [track_value]}),
        "setup/normalized": pd.DataFrame(
            [
                {
                    "setup_id": setup_id,
                    "setup_label": setup_label,
                    "source_file": setup_source,
                    "source_hash": setup_hash,
                    "section": "GEARS",
                    "parameter": "FINAL",
                    "raw_value": str(normalized_value),
                    "value_numeric": normalized_value,
                    "value_text": str(normalized_value),
                    "category": "gearing",
                }
            ]
        ),
        "segments/passes": _passes(session_id, lap_id),
    }
    if include_quality:
        frames["quality/flags"] = pd.DataFrame(
            [
                {
                    "session_id": session_id,
                    "lap_id": lap_id,
                    "code": "TEST",
                    "sample_start": 0,
                    "sample_end": 0,
                    "severity": "warning",
                }
            ]
        )
    refs: list[TableRef] = [
        storage.write(name, frame) for name, frame in frames.items()
    ]
    json_dump(
        root / "setup" / "raw.json",
        {
            setup_id: {
                "setup_id": setup_id,
                "setup_label": setup_label,
                "source_file": setup_source,
                "source_hash": setup_hash,
                "raw": {"GEARS": {"FINAL": setup_value}},
            }
        },
    )
    definitions = {
        "coordinate": "track_s_m",
        "segments": [{"id": "t1", "start": 0, "end": 100}],
    }
    json_dump(root / "segments" / "definitions.json", definitions)
    json_dump(
        root / "manifest.json",
        {
            "schema_version": schema_version,
            "tool_version": "test",
            "dataset_id": f"dataset-{session_id}",
            "source_files": [
                {
                    "path": source_path or f"/{session_id}.acreplay",
                    "name": source_name or f"{session_id}.acreplay",
                    "sha256": source_hash or session_id,
                    "type": "ac_replay",
                }
            ],
            "processing_options": processing_options or {"threshold": 1},
            "track_reference_id": "track-a",
            "track": {"name": "Track A"},
            "tables": table_manifest(refs),
        },
    )


def _replace_table(root: Path, logical_name: str, frame: pd.DataFrame) -> None:
    storage = DatasetStorage(root)
    reference = storage.write(logical_name, frame)
    manifest = json_load(root / "manifest.json")
    manifest["tables"][logical_name] = table_manifest([reference])[logical_name]
    json_dump(root / "manifest.json", manifest)


def _append_setup(
    root: Path,
    *,
    setup_id: str,
    raw_value: object,
    normalized_value: object,
) -> None:
    storage = DatasetStorage(root)
    normalized = storage.read("setup/normalized.parquet")
    row = normalized.iloc[0].to_dict()
    row.update(
        {
            "setup_id": setup_id,
            "source_hash": f"{setup_id}-content",
            "raw_value": str(normalized_value),
            "value_numeric": normalized_value,
            "value_text": str(normalized_value),
        }
    )
    normalized = pd.concat([normalized, pd.DataFrame([row])], ignore_index=True)
    reference = storage.write("setup/normalized", normalized)
    manifest = json_load(root / "manifest.json")
    manifest["tables"]["setup/normalized"] = table_manifest([reference])[
        "setup/normalized"
    ]
    json_dump(root / "manifest.json", manifest)

    raw = json_load(root / "setup" / "raw.json")
    raw[setup_id] = {
        "setup_id": setup_id,
        "setup_label": row["setup_label"],
        "source_file": row["source_file"],
        "source_hash": f"{setup_id}-content",
        "raw": {"GEARS": {"FINAL": raw_value}},
    }
    json_dump(root / "setup" / "raw.json", raw)


@pytest.mark.parametrize(
    ("logical_name", "column"),
    [
        ("sessions", "setup_id"),
        ("laps", "is_complete"),
        ("laps", "is_valid"),
        ("laps", "lap_time_s"),
        ("laps", "source_lap_number"),
        ("setup/normalized", "category"),
        ("segments/passes", "segment_name"),
        ("segments/passes", "valid_for_comparison"),
        ("segments/passes", "segment_time_s"),
        ("segments/passes", "entry_speed_kmh"),
        ("segments/passes", "minimum_speed_kmh"),
        ("segments/passes", "exit_speed_kmh"),
        ("segments/passes", "brake_onset_track_s_m"),
        ("segments/passes", "full_throttle_commit_track_s_m"),
        ("segments/passes", "coasting_time_s"),
    ],
)
def test_validate_dataset_rejects_missing_summary_consumer_columns(
    tmp_path: Path, logical_name: str, column: str
) -> None:
    source = tmp_path / "source"
    _write_dataset(
        source, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    frame = (
        DatasetStorage(source).read(f"{logical_name}.parquet").drop(columns=[column])
    )
    _replace_table(source, logical_name, frame)

    report = validate_dataset(source)

    assert report["status"] == "error"
    assert any(
        issue["code"] == ValidationCode.MISSING_REQUIRED_COLUMNS.value
        and logical_name in issue["message"]
        and column in issue["message"]
        for issue in report["warnings"]
    )


def test_validate_dataset_accepts_empty_zero_column_passes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_dataset(
        source, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    _replace_table(source, "segments/passes", pd.DataFrame())

    report = validate_dataset(source)

    assert report["status"] == "ok", report


def test_merge_accepts_empty_zero_column_passes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_dataset(
        source, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    _replace_table(source, "segments/passes", pd.DataFrame())

    output = tmp_path / "output"
    merge_datasets([source], output)

    passes = DatasetStorage(output).read("segments/passes.parquet")
    assert passes.empty
    assert len(passes.columns) == 0


def test_merge_rejects_missing_setup_normalized_category_before_output_creation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_dataset(
        source, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    normalized = (
        DatasetStorage(source)
        .read("setup/normalized.parquet")
        .drop(columns=["category"])
    )
    _replace_table(source, "setup/normalized", normalized)

    output = tmp_path / "output"
    with pytest.raises(ValueError, match="setup/normalized.*category"):
        merge_datasets([source], output)

    assert not output.exists()


def test_merge_public_interface_is_deterministic_and_preserves_sidecars(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_dataset(
        left, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    _write_dataset(
        right,
        session_id="two",
        setup_label="alternate",
        setup_source="/b.ini",
        include_quality=False,
    )

    output = tmp_path / "merged"
    manifest = merge_datasets([left, right], output)
    assert manifest["schema_version"] == "7"
    assert len(DatasetStorage(output).read("sessions.parquet")) == 2
    assert len(DatasetStorage(output).read("quality/flags.parquet")) == 1
    raw = json_load(output / "setup" / "raw.json")["setup-a"]
    assert raw["source_files"] == ["/a.ini", "/b.ini"]
    assert raw["setup_labels"] == ["alternate", "baseline"]
    assert (
        json_load(output / "segments" / "definitions.json")["segments"][0]["id"] == "t1"
    )
    assert (output / "summaries" / "ai_context.json").is_file()

    repeat = tmp_path / "repeat"
    repeated = merge_datasets([right, left, output], repeat)
    assert repeated["dataset_id"] == manifest["dataset_id"]
    assert json_load(repeat / "setup" / "raw.json") == json_load(
        output / "setup" / "raw.json"
    )
    for logical_name in manifest["tables"]:
        assert (
            DatasetStorage(repeat)
            .read(manifest["tables"][logical_name]["path"])
            .equals(
                DatasetStorage(output).read(manifest["tables"][logical_name]["path"])
            )
        )
    assert json_load(repeat / "summaries" / "ai_context.json") == json_load(
        output / "summaries" / "ai_context.json"
    )


def test_merge_identity_ignores_source_display_paths_and_deduplicates_sources(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_dataset(
        left,
        session_id="same",
        setup_label="baseline",
        setup_source="/setups/a.ini",
        source_hash="replay-content",
        source_path="/replays/a/race.acreplay",
        source_name="race.acreplay",
    )
    _write_dataset(
        right,
        session_id="same",
        setup_label="renamed",
        setup_source="/setups/b.ini",
        source_hash="replay-content",
        source_path="/archive/b/renamed.acreplay",
        source_name="renamed.acreplay",
    )

    left_only = merge_datasets([left], tmp_path / "left-only")
    merged = merge_datasets([right, left], tmp_path / "merged")
    assert merged["dataset_id"] == left_only["dataset_id"]
    provenance = merged["source_files"]
    assert provenance == [
        {
            "sha256": "replay-content",
            "type": "ac_replay",
            "display_paths": [
                "/archive/b/renamed.acreplay",
                "/replays/a/race.acreplay",
            ],
            "display_names": ["race.acreplay", "renamed.acreplay"],
        }
    ]


def test_merge_rejects_raw_and_normalized_setup_contradictions(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_dataset(
        source, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    raw = json_load(source / "setup" / "raw.json")
    raw["setup-a"]["raw"]["GEARS"]["FINAL"] = 3.5
    json_dump(source / "setup" / "raw.json", raw)
    with pytest.raises(ValueError, match="normalized parameters disagree"):
        merge_datasets([source], tmp_path / "output")

    wrong_hash = tmp_path / "wrong-hash"
    _write_dataset(
        wrong_hash, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    raw = json_load(wrong_hash / "setup" / "raw.json")
    raw["setup-a"]["source_hash"] = "wrong-content"
    json_dump(wrong_hash / "setup" / "raw.json", raw)
    with pytest.raises(ValueError, match="source_hash disagrees"):
        merge_datasets([wrong_hash], tmp_path / "wrong-hash-output")

    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_dataset(
        left, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    _write_dataset(
        right,
        session_id="two",
        setup_label="baseline",
        setup_source="/b.ini",
        setup_hash="different-content",
        setup_value=3.5,
    )
    with pytest.raises(ValueError, match="Conflicting setup raw metadata"):
        merge_datasets([left, right], tmp_path / "raw-conflict")


def test_validate_command_reports_shared_setup_integrity_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source"
    _write_dataset(
        source, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    raw = json_load(source / "setup" / "raw.json")
    raw["setup-a"]["raw"] = {"": {"FINAL": 3.0}}
    json_dump(source / "setup" / "raw.json", raw)

    assert main(["validate", str(source)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["checks"]["setup_registry"] == "fail"
    assert any(issue["code"] == "INVALID_SETUP_SECTION" for issue in report["warnings"])

    with pytest.raises(ValueError, match="empty raw section"):
        merge_datasets([source], tmp_path / "output")


def test_merge_accepts_equivalent_integer_and_float_setup_values(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_dataset(
        source,
        session_id="one",
        setup_label="baseline",
        setup_source="/a.ini",
        setup_value=3,
        normalized_setup_value=3.0,
    )

    manifest = merge_datasets([source], tmp_path / "output")

    assert manifest["dataset_id"]


def test_merge_rejects_unreferenced_conflicting_normalized_setup(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_dataset(
        source, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    _append_setup(source, setup_id="unreferenced", raw_value=3, normalized_value=3.5)

    with pytest.raises(ValueError, match="normalized parameters disagree"):
        merge_datasets([source], tmp_path / "output")


def test_merge_surfaces_first_integrity_issue_without_reclassifying_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_dataset(
        source, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    storage = DatasetStorage(source)

    sessions = storage.read("sessions.parquet")
    sessions.loc[0, "session_id"] = None
    sessions_ref = storage.write("sessions", sessions)

    laps = storage.read("laps.parquet").drop(columns=["lap_id"])
    laps_ref = storage.write("laps", laps)

    manifest = json_load(source / "manifest.json")
    manifest["tables"].update(
        {
            **table_manifest([sessions_ref]),
            **table_manifest([laps_ref]),
        }
    )
    json_dump(source / "manifest.json", manifest)

    report = validate_dataset(source)
    first_issue = report["warnings"][0]
    assert first_issue["code"] == ValidationCode.NULL_IDENTITY_VALUES.value

    with pytest.raises(ValueError) as error:
        merge_datasets([source], tmp_path / "output")

    message = str(error.value)
    assert (
        ValidationCode.NULL_IDENTITY_VALUES.value.lower().replace("_", " ") in message
    )
    assert first_issue["message"] in message
    assert "missing required tables" not in message
    assert "incompatible schema version" not in message
    assert not (tmp_path / "output").exists()


def test_merge_rejects_missing_core_tables_and_orphan_relations(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    _write_dataset(
        missing, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    manifest = json_load(missing / "manifest.json")
    del manifest["tables"]["samples"]
    json_dump(missing / "manifest.json", manifest)
    with pytest.raises(ValueError, match="missing required tables"):
        merge_datasets([missing], tmp_path / "missing-output")

    orphan = tmp_path / "orphan"
    _write_dataset(
        orphan, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    storage = DatasetStorage(orphan)
    relation = storage.write(
        "events/relations",
        pd.DataFrame(
            [
                {
                    "relation_id": "relation-a",
                    "event_id_a": "missing-a",
                    "event_id_b": "missing-b",
                }
            ]
        ),
    )
    manifest = json_load(orphan / "manifest.json")
    manifest["tables"].update(table_manifest([relation]))
    json_dump(orphan / "manifest.json", manifest)
    with pytest.raises(ValueError, match="require events/index"):
        merge_datasets([orphan], tmp_path / "orphan-output")


def test_merge_canonicalizes_segment_order_and_regenerates_derived_tables(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_dataset(
        left, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    _write_dataset(
        right, session_id="two", setup_label="baseline", setup_source="/b.ini"
    )
    definitions = json_load(right / "segments" / "definitions.json")
    definitions["segments"] = [
        {"id": "t2", "start": 101, "end": 200},
        *definitions["segments"],
    ]
    left_definitions = json_load(left / "segments" / "definitions.json")
    left_definitions["segments"] = list(reversed(definitions["segments"]))
    json_dump(right / "segments" / "definitions.json", definitions)
    json_dump(left / "segments" / "definitions.json", left_definitions)

    for root in (left, right):
        storage = DatasetStorage(root)
        refs = [
            storage.write("setup/diffs", pd.DataFrame([{"stale": True}])),
            storage.write(
                "summaries/segment_statistics", pd.DataFrame([{"stale": True}])
            ),
        ]
        manifest = json_load(root / "manifest.json")
        manifest["tables"].update(table_manifest(refs))
        json_dump(root / "manifest.json", manifest)

    output = tmp_path / "output"
    merge_datasets([left, right], output)
    merged_definitions = json_load(output / "segments" / "definitions.json")
    assert [item["id"] for item in merged_definitions["segments"]] == ["t1", "t2"]
    output_manifest = json_load(output / "manifest.json")
    assert "setup/diffs" not in output_manifest["tables"]
    statistics = DatasetStorage(output).read("summaries/segment_statistics.parquet")
    assert "stale" not in statistics.columns


def test_merge_rejects_keyed_quality_conflicts(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_dataset(
        left, session_id="same", setup_label="baseline", setup_source="/a.ini"
    )
    _write_dataset(
        right, session_id="same", setup_label="baseline", setup_source="/b.ini"
    )
    manifest = json_load(right / "manifest.json")
    quality_path = manifest["tables"]["quality/flags"]["path"]
    quality = DatasetStorage(right).read(quality_path)
    quality.loc[0, "severity"] = "error"
    DatasetStorage(right).write("quality/flags", quality)
    with pytest.raises(ValueError, match="Conflicting records for 'quality/flags'"):
        merge_datasets([left, right], tmp_path / "output")


def test_merge_rejects_incompatible_inputs_without_replacing_output(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_dataset(
        left, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    _write_dataset(
        right,
        session_id="two",
        setup_label="baseline",
        setup_source="/b.ini",
        processing_options={"threshold": 2},
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "sentinel").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="processing_options"):
        merge_datasets([left, right], output, overwrite=True)
    assert (output / "sentinel").read_text(encoding="utf-8") == "keep"

    with pytest.raises(ValueError, match="cannot also be an input"):
        merge_datasets([left], left, overwrite=True)


def test_merge_rejects_overwrite_output_that_contains_an_input(
    tmp_path: Path,
) -> None:
    output = tmp_path / "container"
    source = output / "source"
    _write_dataset(
        source, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    source_manifest = (source / "manifest.json").read_bytes()

    with pytest.raises(ValueError, match="cannot also be an input"):
        merge_datasets([source], output, overwrite=True)

    assert output.is_dir()
    assert source.is_dir()
    assert (source / "manifest.json").read_bytes() == source_manifest


def test_merge_rejects_output_nested_inside_an_input(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_dataset(
        source, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    output = source / "nested-output"

    with pytest.raises(ValueError, match="cannot also be an input"):
        merge_datasets([source], output, overwrite=True)

    assert source.is_dir()
    assert not output.exists()


def test_merge_rejects_schema_and_static_track_conflicts(tmp_path: Path) -> None:
    left = tmp_path / "left"
    old_schema = tmp_path / "old-schema"
    wrong_track = tmp_path / "wrong-track"
    _write_dataset(
        left, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    _write_dataset(
        old_schema,
        session_id="two",
        setup_label="baseline",
        setup_source="/b.ini",
        schema_version="6",
    )
    _write_dataset(
        wrong_track,
        session_id="two",
        setup_label="baseline",
        setup_source="/b.ini",
        track_value=2,
    )

    with pytest.raises(ValueError, match="schema version"):
        merge_datasets([left, old_schema], tmp_path / "schema-output")
    with pytest.raises(ValueError, match="Static track table"):
        merge_datasets([left, wrong_track], tmp_path / "track-output")


def test_merge_cli_overwrite_is_required(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source"
    _write_dataset(
        source, session_id="one", setup_label="baseline", setup_source="/a.ini"
    )
    output = tmp_path / "output"
    assert main(["merge", str(source), "--output", str(output)]) == 0
    assert main(["merge", str(source), "--output", str(output)]) == 2
    assert "Output directory exists" in capsys.readouterr().err
    assert main(["merge", str(source), "--output", str(output), "--overwrite"]) == 0
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))[
        "dataset_id"
    ]
