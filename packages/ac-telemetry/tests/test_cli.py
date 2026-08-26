from pathlib import Path

import pandas as pd
from ac_telemetry.cli import main
from ac_telemetry.storage import DatasetStorage
from ac_telemetry.util import json_dump


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
