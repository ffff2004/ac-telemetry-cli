import numpy as np
import pandas as pd
import pytest
from ac_telemetry.config import ProcessingConfig
from ac_telemetry.events import VehicleProfile, detect_events


def _samples() -> pd.DataFrame:
    count = 24
    dt_s = np.full(count, 0.02)
    brake = np.zeros(count)
    brake[2:16] = 0.60
    throttle = np.zeros(count)
    throttle[10:22] = 0.70
    gears = np.array([3] * 9 + [0, 0] + [4] * 13)
    data: dict[str, object] = {
        "session_id": ["session"] * count,
        "lap_id": ["lap"] * count,
        "sample_index": np.arange(count),
        "lap_time_s": np.arange(count) * 0.02,
        "dt_s": dt_s,
        "track_s_m": np.arange(count, dtype=float),
        "track_progress": np.arange(count, dtype=float) / count,
        "speed_kmh": np.full(count, 100.0),
        "brake_n": brake,
        "throttle": throttle,
        "is_braking": brake >= 0.10,
        "gear_physical": gears,
        "rpm": np.linspace(7000.0, 6500.0, count),
        "track_long_g": np.zeros(count),
        "steerAngle": np.zeros(count),
        "velocity_heading_rate_rad_s": np.zeros(count),
    }
    for wheel in ("fl", "fr", "rl", "rr"):
        data[f"wheel_{wheel}_slip_ratio"] = np.zeros(count)
        data[f"wheel_{wheel}_load"] = np.full(count, 1000.0)

    # One FL event with a one-sample false gap, one simultaneous FR event,
    # and RL wheelspin spanning the gear shift.
    data["wheel_fl_slip_ratio"][4:6] = -0.30  # type: ignore[index]
    data["wheel_fl_slip_ratio"][7:9] = -0.35  # type: ignore[index]
    data["wheel_fr_slip_ratio"][5:9] = -0.40  # type: ignore[index]
    data["wheel_rl_slip_ratio"][9:14] = 0.12  # type: ignore[index]
    return pd.DataFrame(data)


def test_detects_per_wheel_slip_and_preserves_evidence_duration() -> None:
    result = detect_events(
        _samples(),
        ProcessingConfig(),
        {"session": VehicleProfile(frozenset({"rl", "rr"}))},
    )

    details = result.wheel_slip.set_index(["slip_kind", "wheel"])
    assert set(details.index) == {
        ("lockup", "fl"),
        ("lockup", "fr"),
        ("wheelspin", "rl"),
    }
    assert details.loc[("wheelspin", "rl"), "driven_status"] == "driven"

    fl_id = details.loc[("lockup", "fl"), "event_id"]
    fl_event = result.events.set_index("event_id").loc[fl_id]
    assert fl_event["active_duration_s"] == pytest.approx(0.08)
    assert fl_event["span_duration_s"] == pytest.approx(0.10)


def test_relates_independent_physical_events_to_control_and_shift_context() -> None:
    result = detect_events(_samples(), ProcessingConfig())

    relation_types = {
        frozenset((row.event_type_a, row.event_type_b))
        for row in result.relations.itertuples(index=False)
        if row.relation_type == "overlap"
    }
    assert frozenset(("braking", "lockup")) in relation_types
    assert frozenset(("throttle", "wheelspin")) in relation_types
    assert frozenset(("shift", "wheelspin")) in relation_types

    braking_lockup = result.relations[
        result.relations[["event_type_a", "event_type_b"]]
        .apply(set, axis=1)
        .map(lambda kinds: kinds == {"braking", "lockup"})
    ]
    assert braking_lockup["coactive_duration_s"].max() > 0


def test_shift_span_includes_neutral_transition() -> None:
    result = detect_events(_samples(), ProcessingConfig())

    shift_id = result.shifts.iloc[0]["event_id"]
    shift = result.events.set_index("event_id").loc[shift_id]
    details = result.shifts.set_index("event_id").loc[shift_id]
    assert shift["start_sample"] == 9
    assert shift["end_sample_exclusive"] == 12
    assert shift["span_duration_s"] == pytest.approx(0.06)
    assert details["neutral_duration_s"] == pytest.approx(0.04)
