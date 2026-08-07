from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    """All thresholds that materially affect generated data.

    These values are deliberately explicit and serialized into the manifest.
    """

    position_jump_threshold_m: float = 10.0
    moving_speed_threshold_kmh: float = 50.0
    brake_active_threshold: float = 0.10
    full_throttle_threshold: float = 0.98
    pedal_zero_threshold: float = 0.05
    brake_gap_close_s: float = 0.15
    minimum_brake_event_s: float = 0.20
    minimum_brake_entry_speed_kmh: float = 50.0
    lockup_slip_ratio_threshold: float = -0.25
    lockup_minimum_speed_kmh: float = 80.0
    wheelspin_slip_ratio_threshold: float = 0.08
    wheelspin_minimum_throttle: float = 0.50
    wheelspin_minimum_speed_kmh: float = 30.0
    throttle_event_threshold: float = 0.05
    minimum_throttle_event_s: float = 0.10
    wheel_event_gap_close_s: float = 0.045
    minimum_wheel_event_s: float = 0.030
    time_reset_tolerance_ms: float = 5.0
    duplicate_distance_tolerance_m: float = 1e-6

    def to_dict(self) -> dict[str, float]:
        return asdict(self)
