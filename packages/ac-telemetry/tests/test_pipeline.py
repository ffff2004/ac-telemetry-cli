import importlib.util
from pathlib import Path

from ac_telemetry.pipeline import preprocess_dataset

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
        [{"replay": replay_path}], tmp_path / "dataset", storage_format="csv"
    )

    tables = manifest["tables"]
    assert "events/index" in tables
    assert "events/wheel_slip" in tables
    assert "events/relations" in tables
    assert "events/lockups" not in tables
    assert "events/wheelspin" not in tables
