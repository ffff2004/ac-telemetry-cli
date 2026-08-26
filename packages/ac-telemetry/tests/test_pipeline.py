import importlib.util
from pathlib import Path

from ac_telemetry.pipeline import preprocess_dataset
from ac_telemetry.validation import validate_dataset
from track_fixture import make_track

GENERATOR_PATH = (
    Path(__file__).parents[2]
    / "ac-replay-parser"
    / "tests"
    / "generate_multi_car_csp_replay.py"
)


def _make_replay() -> bytes:
    spec = importlib.util.spec_from_file_location(
        "test_pipeline_replay_generator", GENERATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load fixture generator: {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_replay(["Alice", "Bob"], 4)


def test_pipeline_writes_normalized_event_tables(tmp_path: Path) -> None:
    replay_path = tmp_path / "fixture.acreplay"
    replay_path.write_bytes(_make_replay())

    manifest = preprocess_dataset(
        [{"replay": replay_path}],
        tmp_path / "dataset",
        track_dir=make_track(tmp_path / "track"),
    )

    tables = manifest["tables"]
    assert "table_format" not in manifest
    assert all(info["path"].endswith(".parquet") for info in tables.values())
    assert "events/index" in tables
    assert "events/wheel_slip" in tables
    assert "events/relations" in tables
    assert "events/lockups" not in tables
    assert "events/wheelspin" not in tables

    validation = validate_dataset(tmp_path / "dataset")
    assert validation["status"] == "ok", validation
