import numpy as np
import pandas as pd
from ac_telemetry.assist_activity import detect_abs_activity
from ac_telemetry.config import ProcessingConfig

DT = 0.015


def _fixture(
    fl: np.ndarray,
    fr: np.ndarray,
    brake: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    count = len(fl)
    time = np.arange(count) * DT
    brake_values = np.full(count, 0.9) if brake is None else brake
    samples = pd.DataFrame(
        {
            "sample_index": np.arange(count),
            "session_id": "session",
            "lap_id": "lap",
            "lap_time_s": time,
            "dt_s": DT,
            "actual_distance_m": time * 50.0,
            "progress": time / max(time[-1], DT),
            "speed_kmh": 220.0 - time * 25.0,
            "brake_n": brake_values,
            "wheel_fl_slip_ratio": fl,
            "wheel_fr_slip_ratio": fr,
            "wheel_rl_slip_ratio": -0.02 + 0.002 * np.sin(2 * np.pi * 5 * time),
            "wheel_rr_slip_ratio": -0.02 + 0.002 * np.sin(2 * np.pi * 5 * time + 0.2),
            "wheel_fl_load": np.full(count, 4000.0),
            "wheel_fr_load": np.full(count, 4000.0),
            "wheel_rl_load": np.full(count, 2500.0),
            "wheel_rr_load": np.full(count, 2500.0),
        }
    )
    braking = pd.DataFrame(
        [
            {
                "event_id": "braking-event",
                "session_id": "session",
                "lap_id": "lap",
                "start_sample": 0,
                "end_sample": count - 1,
            }
        ]
    )
    return samples, braking


def test_detects_independent_per_wheel_high_frequency_activity() -> None:
    time = np.arange(267) * DT
    fl = -0.06 + 0.025 * np.sin(2 * np.pi * 24 * time)
    fr = -0.07 + 0.020 * np.sin(2 * np.pi * 20 * time + 1.1)
    samples, braking = _fixture(fl, fr)

    events = detect_abs_activity(samples, braking, ProcessingConfig())

    assert set(events["wheel"]) == {"fl", "fr"}
    assert set(events["event_type"]) == {"abs_intervention_candidate"}
    assert (events["high_band_power_fraction"] > 0.80).all()
    assert (events["observed_peak_frequency_hz"] >= 19.0).all()
    assert (events["parent_braking_event_id"] == "braking-event").all()


def test_does_not_detect_smooth_threshold_braking_or_sustained_lock() -> None:
    time = np.arange(267) * DT
    smooth = -0.06 + 0.02 * np.sin(2 * np.pi * 5 * time)
    smooth_samples, braking = _fixture(smooth, smooth * 1.05)
    locked = np.full_like(time, -1.0)
    lock_samples, lock_braking = _fixture(locked, locked)

    smooth_events = detect_abs_activity(smooth_samples, braking, ProcessingConfig())
    lock_events = detect_abs_activity(lock_samples, lock_braking, ProcessingConfig())

    assert smooth_events.empty
    assert lock_events.empty


def test_rejects_high_frequency_motion_present_in_brake_input() -> None:
    time = np.arange(267) * DT
    slip = -0.06 + 0.025 * np.sin(2 * np.pi * 24 * time)
    brake = 0.80 + 0.10 * np.sin(2 * np.pi * 24 * time)
    samples, braking = _fixture(slip, slip * 0.9, brake)

    events = detect_abs_activity(samples, braking, ProcessingConfig())

    assert events.empty


def test_rejects_spectral_noise_that_does_not_exceed_the_session_floor() -> None:
    time = np.arange(400) * DT
    braking_start = 134
    noise = 0.002 * np.sin(2 * np.pi * 24 * time)
    fl = -0.04 + noise
    fr = -0.04 + 0.9 * noise
    brake = np.zeros(len(time))
    brake[braking_start:] = 0.9
    samples, braking = _fixture(fl, fr, brake)
    braking.loc[0, "start_sample"] = braking_start

    events = detect_abs_activity(samples, braking, ProcessingConfig())

    assert events.empty
