import numpy as np
import pandas as pd
import pytest
from ac_telemetry.assist_activity import detect_tc_activity
from ac_telemetry.config import ProcessingConfig
from ac_telemetry.events import detect_throttle

DT = 0.015


def _fixture(
    rl: np.ndarray,
    rr: np.ndarray,
    throttle: np.ndarray | None = None,
    brake: np.ndarray | None = None,
    fl: np.ndarray | None = None,
    fr: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    count = len(rl)
    time = np.arange(count) * DT
    throttle_values = np.full(count, 0.9) if throttle is None else throttle
    brake_values = np.zeros(count) if brake is None else brake
    fl_values = np.zeros(count) if fl is None else fl
    fr_values = np.zeros(count) if fr is None else fr
    samples = pd.DataFrame(
        {
            "sample_index": np.arange(count),
            "session_id": "session",
            "lap_id": "lap",
            "lap_time_s": time,
            "dt_s": DT,
            "actual_distance_m": time * 40.0,
            "progress": time / max(time[-1], DT),
            "speed_kmh": 60.0 + time * 20.0,
            "throttle": throttle_values,
            "brake_n": brake_values,
            "gear_physical": np.full(count, 3),
            "steerAngle": np.zeros(count),
            "yaw_rate_rad_s": np.zeros(count),
            "rear_slip_ratio_max": np.maximum(rl, rr),
            "wheel_fl_slip_ratio": fl_values,
            "wheel_fr_slip_ratio": fr_values,
            "wheel_rl_slip_ratio": rl,
            "wheel_rr_slip_ratio": rr,
            "wheel_fl_load": np.full(count, 3000.0),
            "wheel_fr_load": np.full(count, 3000.0),
            "wheel_rl_load": np.full(count, 3000.0),
            "wheel_rr_load": np.full(count, 3000.0),
        }
    )
    throttle_events = pd.DataFrame(
        [
            {
                "event_id": "throttle-event",
                "session_id": "session",
                "lap_id": "lap",
                "start_sample": 0,
                "end_sample": count - 1,
            }
        ]
    )
    return samples, throttle_events


def test_detects_rear_wheel_high_frequency_activity_without_slip_threshold() -> None:
    time = np.arange(267) * DT
    rl = -0.01 + 0.025 * np.sin(2 * np.pi * 23 * time)
    rr = 0.01 + 0.020 * np.sin(2 * np.pi * 19 * time + 1.0)
    samples, throttle_events = _fixture(rl, rr)

    events = detect_tc_activity(
        samples,
        throttle_events,
        ProcessingConfig(),
        {"session": frozenset({"rl", "rr"})},
    )

    assert set(events["wheel"]) == {"rl", "rr"}
    assert set(events["event_type"]) == {"tc_intervention_candidate"}
    assert (events["high_band_power_fraction"] > 0.80).all()
    assert (events["parent_throttle_event_id"] == "throttle-event").all()
    assert set(events["driven_status"]) == {"driven"}
    assert not events["quality_flags"].str.contains("unknown_drivetrain").any()


@pytest.mark.parametrize(
    ("driven_wheels", "expected_wheels", "expected_status"),
    [
        (frozenset({"fl", "fr"}), {"fl", "fr"}, "driven"),
        (frozenset({"rl", "rr"}), {"rl", "rr"}, "driven"),
        (frozenset({"fl", "fr", "rl", "rr"}), {"fl", "fr", "rl", "rr"}, "driven"),
        (None, {"fl", "fr", "rl", "rr"}, "unknown"),
    ],
)
def test_selects_tc_wheels_from_session_drivetrain(
    driven_wheels: frozenset[str] | None,
    expected_wheels: set[str],
    expected_status: str,
) -> None:
    time = np.arange(267) * DT
    left = -0.01 + 0.025 * np.sin(2 * np.pi * 23 * time)
    right = 0.01 + 0.020 * np.sin(2 * np.pi * 19 * time + 1.0)
    samples, throttle_events = _fixture(left, right, fl=left, fr=right)

    events = detect_tc_activity(
        samples,
        throttle_events,
        ProcessingConfig(),
        {"session": driven_wheels},
    )

    assert set(events["wheel"]) == expected_wheels
    assert set(events["driven_status"]) == {expected_status}
    has_unknown_flag = bool(
        events["quality_flags"].str.contains("unknown_drivetrain").all()
    )
    assert has_unknown_flag == (expected_status == "unknown")


def test_does_not_detect_smooth_acceleration_or_sustained_wheelspin() -> None:
    time = np.arange(267) * DT
    smooth = 0.03 + 0.02 * np.sin(2 * np.pi * 5 * time)
    smooth_samples, throttle_events = _fixture(smooth, smooth * 1.1)
    spinning = np.full_like(time, 0.8)
    spin_samples, spin_events = _fixture(spinning, spinning)

    assert detect_tc_activity(smooth_samples, throttle_events, ProcessingConfig()).empty
    assert detect_tc_activity(spin_samples, spin_events, ProcessingConfig()).empty


def test_rejects_high_frequency_motion_present_in_throttle_input() -> None:
    time = np.arange(267) * DT
    slip = 0.04 + 0.025 * np.sin(2 * np.pi * 22 * time)
    throttle = 0.75 + 0.10 * np.sin(2 * np.pi * 22 * time)
    samples, throttle_events = _fixture(slip, slip * 0.9, throttle)

    events = detect_tc_activity(samples, throttle_events, ProcessingConfig())

    assert events.empty


def test_rejects_spectral_noise_that_does_not_exceed_session_floor() -> None:
    time = np.arange(400) * DT
    throttle_start = 134
    noise = 0.002 * np.sin(2 * np.pi * 22 * time)
    throttle = np.zeros(len(time))
    throttle[throttle_start:] = 0.9
    samples, throttle_events = _fixture(0.03 + noise, 0.03 + 0.9 * noise, throttle)
    throttle_events.loc[0, "start_sample"] = throttle_start

    events = detect_tc_activity(samples, throttle_events, ProcessingConfig())

    assert events.empty


def test_rejects_driven_wheel_transient_during_gear_change() -> None:
    time = np.arange(267) * DT
    slip = 0.03 + 0.002 * np.sin(2 * np.pi * 5 * time)
    slip[108:158] += 0.025 * np.sin(2 * np.pi * 22 * time[108:158])
    samples, throttle_events = _fixture(slip, slip * 0.9)
    samples.loc[133:, "gear_physical"] = 4

    events = detect_tc_activity(samples, throttle_events, ProcessingConfig())

    assert events.empty


def test_throttle_event_no_longer_contains_legacy_tc_proxy() -> None:
    time = np.arange(80) * DT
    samples, _ = _fixture(np.full_like(time, 0.02), np.full_like(time, 0.02))

    events = detect_throttle(samples, ProcessingConfig())

    assert not events.empty
    assert "tc_activity_proxy" not in events.columns
