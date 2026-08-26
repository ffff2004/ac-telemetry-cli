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
    wheelspin_minimum_speed_kmh: float = 30.0
    throttle_event_threshold: float = 0.05
    throttle_gap_close_s: float = 0.0
    minimum_throttle_event_s: float = 0.10
    wheel_event_gap_close_s: float = 0.045
    minimum_wheel_event_s: float = 0.030
    shift_stable_samples: int = 2
    event_near_shift_s: float = 0.50
    abs_analysis_window_s: float = 0.75
    abs_analysis_hop_s: float = 0.075
    abs_low_frequency_min_hz: float = 3.0
    abs_low_frequency_max_hz: float = 8.0
    abs_high_frequency_min_hz: float = 15.0
    abs_high_frequency_max_hz: float = 32.0
    abs_nyquist_fraction: float = 0.48
    abs_min_high_to_low_power_ratio: float = 0.50
    abs_min_high_band_power_fraction: float = 0.30
    abs_max_brake_high_band_power_fraction: float = 0.25
    abs_noise_floor_percentile: float = 90.0
    abs_min_high_band_noise_excess_ratio: float = 1.5
    abs_event_gap_close_s: float = 0.15
    minimum_abs_event_s: float = 0.30
    tc_analysis_window_s: float = 0.75
    tc_analysis_hop_s: float = 0.075
    tc_low_frequency_min_hz: float = 3.0
    tc_low_frequency_max_hz: float = 8.0
    tc_high_frequency_min_hz: float = 15.0
    tc_high_frequency_max_hz: float = 32.0
    tc_nyquist_fraction: float = 0.48
    tc_min_high_to_low_power_ratio: float = 0.50
    tc_min_high_band_power_fraction: float = 0.30
    tc_max_throttle_high_band_power_fraction: float = 0.25
    tc_noise_floor_percentile: float = 90.0
    tc_min_high_band_noise_excess_ratio: float = 1.5
    tc_event_gap_close_s: float = 0.15
    minimum_tc_event_s: float = 0.30
    tc_shift_exclusion_s: float = 0.50
    time_reset_tolerance_ms: float = 5.0
    duplicate_distance_tolerance_m: float = 1e-6

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)
