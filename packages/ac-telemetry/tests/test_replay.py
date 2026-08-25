import importlib.util
from pathlib import Path

from ac_telemetry.config import ProcessingConfig
from ac_telemetry.replay import inspect_replay, load_replay

GENERATOR_PATH = (
    Path(__file__).parents[2]
    / "ac-replay-parser"
    / "tests"
    / "generate_multi_car_csp_replay.py"
)


def _make_replay(driver_names: list[str], frames: int) -> bytes:
    spec = importlib.util.spec_from_file_location(
        "test_replay_generator", GENERATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load fixture generator: {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_replay(driver_names, frames)


def test_load_replay_expands_cars_into_sessions(tmp_path: Path) -> None:
    path = tmp_path / "fixture.acreplay"
    path.write_bytes(_make_replay(["Alice", "Bob"], 3))

    results = load_replay(path, ProcessingConfig())

    assert [result.metadata["driver_name"] for result in results] == ["Alice", "Bob"]
    assert [len(result.samples) for result in results] == [3, 3]
    assert results[0].samples["clutch_raw"].tolist() == [200, 199, 198]
    assert results[1].samples["position.x"].iloc[0] == 101.25
    assert all(result.metadata["source_format"] == "acreplay" for result in results)


def test_inspect_replay_reports_cars(tmp_path: Path) -> None:
    path = tmp_path / "fixture.acreplay"
    path.write_bytes(_make_replay(["Alice", "Bob"], 2))

    result = inspect_replay(path)

    assert result["car_count"] == 2
    assert result["driver_names"] == ["Alice", "Bob"]
    assert result["cars"][1]["extra_version"] == 7
