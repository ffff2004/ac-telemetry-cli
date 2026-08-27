import io
import json
from pathlib import Path

import pandas as pd
import pytest
from ac_telemetry.cli import main
from ac_telemetry.storage import DatasetStorage
from ac_telemetry.util import json_dump


def test_sections_to_segments_reads_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            "[SECTION_0]\nIN=0.25\nOUT=0.5\nTEXT=First Corner\n"
            "[SECTION_1]\nIN=0.75\nOUT=0.9\nTEXT=Second Corner\n"
        ),
    )

    assert (
        main(
            [
                "sections-to-segments",
                "-",
                "--track",
                "test_track",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert [(item["start"], item["end"]) for item in result["segments"]] == [
        (0.0, 0.25),
        (0.25, 0.75),
        (0.75, 1.0),
    ]


def test_sections_to_segments_reads_file_and_writes_output(tmp_path: Path) -> None:
    sections = tmp_path / "sections.ini"
    sections.write_text("[SECTION_0]\nIN=0.4\nOUT=0.5\nTEXT=Corner\n", encoding="utf-8")
    output = tmp_path / "segments.json"

    assert (
        main(
            [
                "sections-to-segments",
                str(sections),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["segments"] == [
        {
            "id": "start_to_section_0",
            "name": "Start to Corner",
            "start": 0.0,
            "end": 0.4,
        },
        {
            "id": "section_0_to_finish",
            "name": "Corner + exit to finish",
            "start": 0.4,
            "end": 1.0,
        },
    ]


def test_export_csv_writes_csv_from_parquet_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    table = pd.DataFrame({"sample_index": [1], "value": [2.5]})
    ref = DatasetStorage(dataset).write("samples", table)
    json_dump(
        dataset / "manifest.json",
        {"tables": {ref.name: {"path": ref.relative_path}}},
    )

    output = tmp_path / "samples.csv"
    assert (
        main(
            [
                "export-csv",
                str(dataset),
                "--table",
                "samples",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    pd.testing.assert_frame_equal(pd.read_csv(output), table)


def test_preprocess_no_longer_accepts_setup_sp(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        main(["preprocess", "--setup-sp", "legacy.sp", "--output", "output"])
    assert "unrecognized arguments: --setup-sp" in capsys.readouterr().err
    config = tmp_path / "dataset.json"
    json_dump(
        config,
        {
            "track": "track",
            "sessions": [{"replay": "run.acreplay", "setup_sp": "legacy.sp"}],
        },
    )
    assert main(["preprocess", "--config", str(config), "--output", "output"]) == 2
    assert "setup_sp is no longer supported" in capsys.readouterr().err
