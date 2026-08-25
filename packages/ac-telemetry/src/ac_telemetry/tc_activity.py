from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._spectral_activity import (
    band_power,
    band_signal,
    detrend,
    high_band_rms,
    safe_correlation,
    spectrum,
    window_starts,
)
from .config import ProcessingConfig
from .util import close_short_false_gaps, contiguous_true_runs, stable_id


# AC replay data has no native TC torque-cut channel.  The detector therefore
# looks for the same observable mechanism as the ABS detector: high-frequency
# slip-ratio suppression that is absent from the driver's pedal input.  Unlike
# the controlled ABS experiment documented in abs_activity.py, no labelled TC
# experiment has yet calibrated a universal frequency.  The default 15-32 Hz
# band is deliberately provisional, configurable, and guarded by a per-session
# non-throttle noise floor.  No positive-slip target or trigger threshold is
# imposed: sustained wheelspin alone is not evidence of feedback intervention.
#
# Only the rear wheels are considered because the pipeline's existing wheelspin
# model is rear-drive-specific and replay metadata does not expose driven axles.
# Front- or all-wheel-drive support requires drivetrain metadata or an explicit
# driven-wheel configuration, rather than guessing from transient slip.
DRIVEN_WHEELS = ("rl", "rr")
OPPOSITE_WHEEL = {"rl": "rr", "rr": "rl"}
DETECTION_METHOD = "driven_wheel_slip_spectral_activity_v1"

EVENT_COLUMNS = [
    "event_id",
    "parent_throttle_event_id",
    "session_id",
    "lap_id",
    "event_type",
    "detection_method",
    "wheel",
    "start_sample",
    "end_sample",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "sample_count",
    "start_distance_m",
    "end_distance_m",
    "start_progress",
    "end_progress",
    "entry_speed_kmh",
    "exit_speed_kmh",
    "mean_throttle",
    "peak_throttle",
    "mean_brake",
    "mean_wheel_load",
    "mean_slip_ratio",
    "maximum_slip_ratio",
    "detrended_slip_rms",
    "observed_peak_frequency_hz",
    "spectral_centroid_hz",
    "high_band_power_fraction",
    "high_to_low_power_ratio",
    "high_band_noise_excess_ratio",
    "throttle_high_band_power_fraction",
    "same_axle_high_band_correlation",
    "activity_score",
    "confidence",
    "quality_flags",
]


@dataclass(frozen=True, slots=True)
class _WindowEvidence:
    start: int
    end: int
    candidate: bool
    peak_frequency_hz: float
    spectral_centroid_hz: float
    high_band_power_fraction: float
    high_to_low_power_ratio: float
    high_band_noise_excess_ratio: float
    throttle_high_band_power_fraction: float
    same_axle_high_band_correlation: float
    activity_score: float


def _noise_floors(
    samples: pd.DataFrame,
    config: ProcessingConfig,
) -> dict[tuple[str, str], float]:
    """Estimate each rear wheel's high-frequency floor while power is not applied."""
    floors: dict[tuple[str, str], float] = {}
    for session_id, original in samples.groupby("session_id", sort=False):
        g = original.sort_values("sample_index").reset_index(drop=True)
        median_dt = float(g["dt_s"].median())
        if not np.isfinite(median_dt) or median_dt <= 0:
            continue
        sample_rate_hz = 1.0 / median_dt
        high_max_hz = min(
            config.tc_high_frequency_max_hz,
            config.tc_nyquist_fraction * sample_rate_hz,
        )
        if high_max_hz <= config.tc_high_frequency_min_hz:
            continue
        window_samples = max(16, round(config.tc_analysis_window_s / median_dt))
        baseline_mask = (
            (g["throttle"] < config.throttle_event_threshold)
            & (g["brake_n"] < config.brake_active_threshold)
            & (g["speed_kmh"] >= config.wheelspin_minimum_speed_kmh)
        ).to_numpy(dtype=bool)
        runs = contiguous_true_runs(baseline_mask)
        for wheel in DRIVEN_WHEELS:
            column = f"wheel_{wheel}_slip_ratio"
            if column not in g:
                continue
            rms_values: list[float] = []
            for start, end in runs:
                for offset in range(start, end - window_samples + 2, window_samples):
                    values = g[column].iloc[offset : offset + window_samples].to_numpy(dtype=float)
                    if len(values) == window_samples:
                        rms_values.append(
                            high_band_rms(
                                values,
                                sample_rate_hz,
                                config.tc_high_frequency_min_hz,
                                high_max_hz,
                            )
                        )
            floors[(str(session_id), wheel)] = (
                float(np.percentile(rms_values, config.tc_noise_floor_percentile))
                if rms_values
                else float("nan")
            )
    return floors


def _window_evidence(
    segment: pd.DataFrame,
    wheel: str,
    start: int,
    end: int,
    sample_rate_hz: float,
    high_max_hz: float,
    noise_floor: float,
    config: ProcessingConfig,
) -> _WindowEvidence:
    wheel_values = segment[f"wheel_{wheel}_slip_ratio"].iloc[start : end + 1].to_numpy(float)
    opposite_values = segment[
        f"wheel_{OPPOSITE_WHEEL[wheel]}_slip_ratio"
    ].iloc[start : end + 1].to_numpy(float)
    throttle_values = segment["throttle"].iloc[start : end + 1].to_numpy(float)
    brake_values = segment["brake_n"].iloc[start : end + 1].to_numpy(float)
    gear_values = (
        segment["gear_physical"].iloc[start : end + 1].dropna().to_numpy(int)
        if "gear_physical" in segment
        else np.ones(1, dtype=int)
    )

    frequencies, wheel_power = spectrum(wheel_values, sample_rate_hz)
    _, throttle_power = spectrum(throttle_values, sample_rate_hz)
    low_power = band_power(
        frequencies,
        wheel_power,
        config.tc_low_frequency_min_hz,
        config.tc_low_frequency_max_hz,
    )
    high_power = band_power(
        frequencies,
        wheel_power,
        config.tc_high_frequency_min_hz,
        high_max_hz,
    )
    analysis_power = band_power(frequencies, wheel_power, 2.0, high_max_hz)
    throttle_high_power = band_power(
        frequencies,
        throttle_power,
        config.tc_high_frequency_min_hz,
        high_max_hz,
    )
    throttle_analysis_power = band_power(frequencies, throttle_power, 2.0, high_max_hz)
    epsilon = np.finfo(float).eps
    high_to_low = high_power / max(low_power, epsilon)
    high_fraction = high_power / max(analysis_power, epsilon)
    throttle_high_fraction = throttle_high_power / max(throttle_analysis_power, epsilon)
    high_rms = high_band_rms(
        wheel_values,
        sample_rate_hz,
        config.tc_high_frequency_min_hz,
        high_max_hz,
    )
    noise_excess = (
        high_rms / max(noise_floor, epsilon) if np.isfinite(noise_floor) else float("nan")
    )

    high_mask = (
        (frequencies >= config.tc_high_frequency_min_hz) & (frequencies < high_max_hz)
    )
    if high_mask.any() and high_power > epsilon:
        high_indices = np.flatnonzero(high_mask)
        peak_index = int(high_indices[np.argmax(wheel_power[high_mask])])
        peak_frequency = float(frequencies[peak_index])
        centroid = float(
            np.sum(frequencies[high_mask] * wheel_power[high_mask]) / high_power
        )
    else:
        peak_frequency = float("nan")
        centroid = float("nan")

    wheel_high = band_signal(
        wheel_values, sample_rate_hz, config.tc_high_frequency_min_hz, high_max_hz
    )
    opposite_high = band_signal(
        opposite_values, sample_rate_hz, config.tc_high_frequency_min_hz, high_max_hz
    )
    same_axle_correlation = safe_correlation(wheel_high, opposite_high)

    ratio_score = float(
        np.clip(
            np.log10(max(high_to_low, epsilon) / config.tc_min_high_to_low_power_ratio)
            + 1.0,
            0.0,
            2.0,
        )
        / 2.0
    )
    fraction_score = float(
        np.clip(
            (high_fraction - config.tc_min_high_band_power_fraction)
            / max(1.0 - config.tc_min_high_band_power_fraction, epsilon),
            0.0,
            1.0,
        )
    )
    pedal_score = float(
        np.clip(
            1.0 - throttle_high_fraction / config.tc_max_throttle_high_band_power_fraction,
            0.0,
            1.0,
        )
    )
    noise_score = (
        float(
            np.clip(
                np.log10(
                    max(noise_excess, epsilon)
                    / config.tc_min_high_band_noise_excess_ratio
                )
                + 1.0,
                0.0,
                2.0,
            )
            / 2.0
        )
        if np.isfinite(noise_excess)
        else 1.0
    )
    activity_score = (
        0.40 * ratio_score
        + 0.30 * fraction_score
        + 0.20 * pedal_score
        + 0.10 * noise_score
    )
    braking = float(np.mean(brake_values)) >= config.brake_active_threshold
    # A paddle shift creates a short, highly correlated driven-wheel transient
    # while the pedal can remain flat.  That is commanded shift torque-cut, not
    # feedback TC, so every analysis window must remain in one physical gear.
    stable_forward_gear = (
        len(gear_values) > 0
        and np.all(gear_values == gear_values[0])
        and gear_values[0] > 0
    )
    away_from_shift = True
    if "gear_physical" in segment:
        all_gears = segment["gear_physical"].fillna(0).to_numpy(int)
        shift_indices = np.flatnonzero(all_gears[1:] != all_gears[:-1]) + 1
        shift_margin = round(config.tc_shift_exclusion_s * sample_rate_hz)
        away_from_shift = not np.any(
            (shift_indices >= start - shift_margin) & (shift_indices <= end + shift_margin)
        )
    candidate = bool(
        not braking
        and stable_forward_gear
        and away_from_shift
        and high_to_low >= config.tc_min_high_to_low_power_ratio
        and high_fraction >= config.tc_min_high_band_power_fraction
        and throttle_high_fraction <= config.tc_max_throttle_high_band_power_fraction
        and (
            not np.isfinite(noise_excess)
            or noise_excess >= config.tc_min_high_band_noise_excess_ratio
        )
    )
    return _WindowEvidence(
        start=start,
        end=end,
        candidate=candidate,
        peak_frequency_hz=peak_frequency,
        spectral_centroid_hz=centroid,
        high_band_power_fraction=high_fraction,
        high_to_low_power_ratio=high_to_low,
        high_band_noise_excess_ratio=noise_excess,
        throttle_high_band_power_fraction=throttle_high_fraction,
        same_axle_high_band_correlation=same_axle_correlation,
        activity_score=activity_score,
    )


def _quality_flags(
    evidence: list[_WindowEvidence],
    sample_rate_hz: float,
    window_samples: int,
    event_samples: int,
) -> str:
    flags: list[str] = []
    peaks = [item.peak_frequency_hz for item in evidence if np.isfinite(item.peak_frequency_hz)]
    if peaks and float(np.median(peaks)) >= 0.80 * sample_rate_hz / 2.0:
        flags.append("near_nyquist_alias_risk")
    if event_samples <= window_samples:
        flags.append("single_analysis_window")
    if not any(np.isfinite(item.high_band_noise_excess_ratio) for item in evidence):
        flags.append("no_non_throttle_noise_baseline")
    flags.append("rear_drive_assumption")
    return ";".join(flags)


def _event_row(
    parent_throttle_event_id: str,
    segment: pd.DataFrame,
    wheel: str,
    start: int,
    end: int,
    evidence: list[_WindowEvidence],
    sample_rate_hz: float,
    window_samples: int,
) -> dict[str, Any]:
    event_segment = segment.iloc[start : end + 1]
    slip = event_segment[f"wheel_{wheel}_slip_ratio"].to_numpy(float)
    first = event_segment.iloc[0]
    last = event_segment.iloc[-1]
    mean_score = float(np.mean([item.activity_score for item in evidence]))
    alias_penalty = 0.85 if any(
        item.peak_frequency_hz >= 0.80 * sample_rate_hz / 2.0
        for item in evidence
        if np.isfinite(item.peak_frequency_hz)
    ) else 1.0
    parent_id = str(parent_throttle_event_id)
    return {
        "event_id": stable_id(parent_id, wheel, int(first["sample_index"])),
        "parent_throttle_event_id": parent_id,
        "session_id": first["session_id"],
        "lap_id": first["lap_id"],
        "event_type": "tc_intervention_candidate",
        "detection_method": DETECTION_METHOD,
        "wheel": wheel,
        "start_sample": int(first["sample_index"]),
        "end_sample": int(last["sample_index"]),
        "start_time_s": float(first["lap_time_s"]),
        "end_time_s": float(last["lap_time_s"]),
        "duration_s": float(event_segment["dt_s"].sum()),
        "sample_count": len(event_segment),
        "start_distance_m": float(first["actual_distance_m"]),
        "end_distance_m": float(last["actual_distance_m"]),
        "start_progress": float(first["progress"]),
        "end_progress": float(last["progress"]),
        "entry_speed_kmh": float(first["speed_kmh"]),
        "exit_speed_kmh": float(last["speed_kmh"]),
        "mean_throttle": float(event_segment["throttle"].mean()),
        "peak_throttle": float(event_segment["throttle"].max()),
        "mean_brake": float(event_segment["brake_n"].mean()),
        "mean_wheel_load": float(event_segment[f"wheel_{wheel}_load"].mean()),
        "mean_slip_ratio": float(np.mean(slip)),
        "maximum_slip_ratio": float(np.max(slip)),
        "detrended_slip_rms": float(np.sqrt(np.mean(detrend(slip) ** 2))),
        "observed_peak_frequency_hz": float(
            np.nanmedian([item.peak_frequency_hz for item in evidence])
        ),
        "spectral_centroid_hz": float(
            np.nanmedian([item.spectral_centroid_hz for item in evidence])
        ),
        "high_band_power_fraction": float(
            np.mean([item.high_band_power_fraction for item in evidence])
        ),
        "high_to_low_power_ratio": float(
            np.mean([item.high_to_low_power_ratio for item in evidence])
        ),
        "high_band_noise_excess_ratio": float(
            np.nanmean([item.high_band_noise_excess_ratio for item in evidence])
        )
        if any(np.isfinite(item.high_band_noise_excess_ratio) for item in evidence)
        else float("nan"),
        "throttle_high_band_power_fraction": float(
            np.mean([item.throttle_high_band_power_fraction for item in evidence])
        ),
        "same_axle_high_band_correlation": float(
            np.mean([item.same_axle_high_band_correlation for item in evidence])
        ),
        "activity_score": mean_score,
        "confidence": float(np.clip(mean_score * alias_penalty, 0.0, 1.0)),
        "quality_flags": _quality_flags(
            evidence, sample_rate_hz, window_samples, len(event_segment)
        ),
    }


def detect_tc_activity(
    samples: pd.DataFrame,
    throttle_events: pd.DataFrame,
    config: ProcessingConfig,
) -> pd.DataFrame:
    """Detect per-driven-wheel spectral TC activity inside throttle events."""
    if samples.empty or throttle_events.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    rows: list[dict[str, Any]] = []
    noise_floors = _noise_floors(samples, config)
    sample_groups = {
        (str(session_id), str(lap_id)): group.sort_values("sample_index").reset_index(
            drop=True
        )
        for (session_id, lap_id), group in samples.groupby(
            ["session_id", "lap_id"], sort=False
        )
    }
    for throttle_event in throttle_events.itertuples(index=False):
        lap_samples = sample_groups.get(
            (str(throttle_event.session_id), str(throttle_event.lap_id))
        )
        if lap_samples is None:
            continue
        segment = lap_samples[
            (lap_samples["sample_index"] >= int(throttle_event.start_sample))
            & (lap_samples["sample_index"] <= int(throttle_event.end_sample))
        ].reset_index(drop=True)
        if segment.empty:
            continue
        median_dt = float(segment["dt_s"].median())
        if not np.isfinite(median_dt) or median_dt <= 0:
            continue
        sample_rate_hz = 1.0 / median_dt
        high_max_hz = min(
            config.tc_high_frequency_max_hz,
            config.tc_nyquist_fraction * sample_rate_hz,
        )
        if high_max_hz <= config.tc_high_frequency_min_hz:
            continue
        window_samples = max(16, round(config.tc_analysis_window_s / median_dt))
        hop_samples = max(1, round(config.tc_analysis_hop_s / median_dt))
        starts = window_starts(len(segment), window_samples, hop_samples)
        for wheel in DRIVEN_WHEELS:
            required = [
                f"wheel_{wheel}_slip_ratio",
                f"wheel_{OPPOSITE_WHEEL[wheel]}_slip_ratio",
                f"wheel_{wheel}_load",
            ]
            if any(column not in segment for column in required):
                continue
            windows = [
                _window_evidence(
                    segment,
                    wheel,
                    start,
                    start + window_samples - 1,
                    sample_rate_hz,
                    high_max_hz,
                    noise_floors.get(
                        (str(throttle_event.session_id), wheel), float("nan")
                    ),
                    config,
                )
                for start in starts
            ]
            mask = np.asarray([item.candidate for item in windows], dtype=bool)
            gap_windows = max(
                0,
                round(config.tc_event_gap_close_s / max(config.tc_analysis_hop_s, median_dt)),
            )
            mask = close_short_false_gaps(mask, gap_windows)
            for first_window, last_window in contiguous_true_runs(mask):
                selected = [
                    item for item in windows[first_window : last_window + 1] if item.candidate
                ]
                if not selected:
                    continue
                start = windows[first_window].start
                end = windows[last_window].end
                duration_s = float(segment.iloc[start : end + 1]["dt_s"].sum())
                if duration_s < config.minimum_tc_event_s:
                    continue
                rows.append(
                    _event_row(
                        str(throttle_event.event_id),
                        segment,
                        wheel,
                        start,
                        end,
                        selected,
                        sample_rate_hz,
                        window_samples,
                    )
                )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)
