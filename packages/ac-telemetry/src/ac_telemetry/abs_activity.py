from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ProcessingConfig
from .util import close_short_false_gaps, contiguous_true_runs, stable_id


# Detector design evidence from the controlled brake test in
# AC_250826-200255_O_ks_mercedes_amg_gt3_ks_silverstone1967_.acreplay.
# The first braking event of each lap was analysed after trimming 0.25 s from
# both ends and removing its quadratic trend.  Replay sampling was 66.67 Hz.
#
#   Laps  ABS / driver input       Recorded front-wheel response
#   1-3   off, threshold braking   peak brake 0.812-0.871; spectrum centred at
#                                  5.4 Hz; 15-32 Hz power 0.8%; mean slip -0.057
#   4-6   ABS 1, hard braking      peak brake 1.000; spectrum centred at 22.4 Hz;
#                                  15-32 Hz power 84.3%; mean slip -0.051
#   7-9   ABS 6, hard braking      peak brake 1.000; spectrum centred at 16.5 Hz;
#                                  15-32 Hz power 56.1%; mean slip -0.088
#   10    ABS 12, active           peak brake 1.000; spectrum centred at 16.3 Hz;
#                                  15-32 Hz power 51.6%; mean slip -0.126
#   11    ABS 12, barely active    peak brake 0.737; spectrum centred at 5.1 Hz;
#                                  15-32 Hz power 1.4%; mean slip -0.057
#   12    off, sustained lock      peak brake 1.000; spectrum centred at 3.7 Hz;
#                                  15-32 Hz power 1.4%; mean slip -0.920
#
# ABS-active front-wheel high-frequency signals were nearly independent across
# the axle, while the brake-input channel remained low-frequency.  This supports
# per-wheel feedback modulation downstream of the driver's pedal.  The observed
# 15-32 Hz band is close to the 33.3 Hz Nyquist limit and may contain aliases.
# These measurements justify the spectral evidence used below; they are not
# universal slip targets, ABS frequencies, or manually imposed slip thresholds.
WHEELS = ("fl", "fr", "rl", "rr")
OPPOSITE_WHEEL = {"fl": "fr", "fr": "fl", "rl": "rr", "rr": "rl"}
DETECTION_METHOD = "slip_ratio_spectral_activity_v1"

EVENT_COLUMNS = [
    "event_id",
    "parent_braking_event_id",
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
    "mean_brake",
    "peak_brake",
    "mean_wheel_load",
    "mean_slip_ratio",
    "minimum_slip_ratio",
    "detrended_slip_rms",
    "observed_peak_frequency_hz",
    "spectral_centroid_hz",
    "high_band_power_fraction",
    "high_to_low_power_ratio",
    "high_band_noise_excess_ratio",
    "brake_high_band_power_fraction",
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
    brake_high_band_power_fraction: float
    same_axle_high_band_correlation: float
    activity_score: float


def _detrend(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return values - np.nanmean(values)
    x = np.linspace(-1.0, 1.0, len(values))
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return np.zeros_like(values)
    filled = np.interp(x, x[finite], values[finite])
    return filled - np.polyval(np.polyfit(x, filled, 2), x)


def _spectrum(values: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    detrended = _detrend(values)
    windowed = detrended * np.hanning(len(detrended))
    frequencies = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate_hz)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    return frequencies, power


def _band_power(frequencies: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies < high)
    return float(power[mask].sum())


def _band_signal(values: np.ndarray, sample_rate_hz: float, low: float, high: float) -> np.ndarray:
    detrended = _detrend(values)
    spectrum = np.fft.rfft(detrended)
    frequencies = np.fft.rfftfreq(len(detrended), d=1.0 / sample_rate_hz)
    spectrum[(frequencies < low) | (frequencies > high)] = 0
    return np.fft.irfft(spectrum, n=len(detrended))


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= np.finfo(float).eps or np.std(right) <= np.finfo(float).eps:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _high_band_rms(
    values: np.ndarray,
    sample_rate_hz: float,
    high_min_hz: float,
    high_max_hz: float,
) -> float:
    filtered = _band_signal(values, sample_rate_hz, high_min_hz, high_max_hz)
    return float(np.sqrt(np.mean(filtered**2)))


def _noise_floors(
    samples: pd.DataFrame,
    config: ProcessingConfig,
) -> dict[tuple[str, str], float]:
    """Estimate each wheel's high-frequency floor from non-braking rolling data."""
    floors: dict[tuple[str, str], float] = {}
    for session_id, original in samples.groupby("session_id", sort=False):
        g = original.sort_values("sample_index").reset_index(drop=True)
        median_dt = float(g["dt_s"].median())
        if not np.isfinite(median_dt) or median_dt <= 0:
            continue
        sample_rate_hz = 1.0 / median_dt
        high_max_hz = min(
            config.abs_high_frequency_max_hz,
            config.abs_nyquist_fraction * sample_rate_hz,
        )
        if high_max_hz <= config.abs_high_frequency_min_hz:
            continue
        window_samples = max(16, round(config.abs_analysis_window_s / median_dt))
        rolling_mask = (
            (g["brake_n"] < config.brake_active_threshold)
            & (g["speed_kmh"] >= config.minimum_brake_entry_speed_kmh)
        ).to_numpy(dtype=bool)
        runs = contiguous_true_runs(rolling_mask)
        for wheel in WHEELS:
            column = f"wheel_{wheel}_slip_ratio"
            if column not in g:
                continue
            rms_values: list[float] = []
            for start, end in runs:
                for offset in range(start, end - window_samples + 2, window_samples):
                    values = g[column].iloc[offset : offset + window_samples].to_numpy(dtype=float)
                    if len(values) != window_samples:
                        continue
                    rms_values.append(
                        _high_band_rms(
                            values,
                            sample_rate_hz,
                            config.abs_high_frequency_min_hz,
                            high_max_hz,
                        )
                    )
            floors[(str(session_id), wheel)] = (
                float(np.percentile(rms_values, config.abs_noise_floor_percentile))
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
    wheel_values = segment[f"wheel_{wheel}_slip_ratio"].iloc[start : end + 1].to_numpy(dtype=float)
    opposite_values = segment[
        f"wheel_{OPPOSITE_WHEEL[wheel]}_slip_ratio"
    ].iloc[start : end + 1].to_numpy(dtype=float)
    brake_values = segment["brake_n"].iloc[start : end + 1].to_numpy(dtype=float)

    frequencies, wheel_power = _spectrum(wheel_values, sample_rate_hz)
    _, brake_power = _spectrum(brake_values, sample_rate_hz)
    low_power = _band_power(
        frequencies,
        wheel_power,
        config.abs_low_frequency_min_hz,
        config.abs_low_frequency_max_hz,
    )
    high_power = _band_power(
        frequencies,
        wheel_power,
        config.abs_high_frequency_min_hz,
        high_max_hz,
    )
    analysis_power = _band_power(frequencies, wheel_power, 2.0, high_max_hz)
    brake_high_power = _band_power(
        frequencies,
        brake_power,
        config.abs_high_frequency_min_hz,
        high_max_hz,
    )
    brake_analysis_power = _band_power(frequencies, brake_power, 2.0, high_max_hz)
    epsilon = np.finfo(float).eps
    high_to_low = high_power / max(low_power, epsilon)
    high_fraction = high_power / max(analysis_power, epsilon)
    brake_high_fraction = brake_high_power / max(brake_analysis_power, epsilon)
    high_rms = _high_band_rms(
        wheel_values,
        sample_rate_hz,
        config.abs_high_frequency_min_hz,
        high_max_hz,
    )
    noise_excess = (
        high_rms / max(noise_floor, epsilon) if np.isfinite(noise_floor) else float("nan")
    )

    high_mask = (
        (frequencies >= config.abs_high_frequency_min_hz)
        & (frequencies < high_max_hz)
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

    wheel_high = _band_signal(
        wheel_values, sample_rate_hz, config.abs_high_frequency_min_hz, high_max_hz
    )
    opposite_high = _band_signal(
        opposite_values, sample_rate_hz, config.abs_high_frequency_min_hz, high_max_hz
    )
    same_axle_correlation = _safe_correlation(wheel_high, opposite_high)

    ratio_score = float(
        np.clip(
            np.log10(max(high_to_low, epsilon) / config.abs_min_high_to_low_power_ratio) + 1.0,
            0.0,
            2.0,
        )
        / 2.0
    )
    fraction_score = float(
        np.clip(
            (high_fraction - config.abs_min_high_band_power_fraction)
            / max(1.0 - config.abs_min_high_band_power_fraction, epsilon),
            0.0,
            1.0,
        )
    )
    pedal_score = float(
        np.clip(
            1.0 - brake_high_fraction / config.abs_max_brake_high_band_power_fraction,
            0.0,
            1.0,
        )
    )
    independence_score = float(np.clip(1.0 - abs(same_axle_correlation), 0.0, 1.0))
    noise_score = (
        float(
            np.clip(
                np.log10(
                    max(noise_excess, epsilon)
                    / config.abs_min_high_band_noise_excess_ratio
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
        0.35 * ratio_score
        + 0.25 * fraction_score
        + 0.20 * pedal_score
        + 0.10 * independence_score
        + 0.10 * noise_score
    )
    candidate = bool(
        high_to_low >= config.abs_min_high_to_low_power_ratio
        and high_fraction >= config.abs_min_high_band_power_fraction
        and brake_high_fraction <= config.abs_max_brake_high_band_power_fraction
        and (
            not np.isfinite(noise_excess)
            or noise_excess >= config.abs_min_high_band_noise_excess_ratio
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
        brake_high_band_power_fraction=brake_high_fraction,
        same_axle_high_band_correlation=same_axle_correlation,
        activity_score=activity_score,
    )


def _window_starts(length: int, window_samples: int, hop_samples: int) -> list[int]:
    if length < window_samples:
        return []
    starts = list(range(0, length - window_samples + 1, hop_samples))
    last = length - window_samples
    if starts[-1] != last:
        starts.append(last)
    return starts


def _quality_flags(
    evidence: list[_WindowEvidence],
    sample_rate_hz: float,
    window_samples: int,
    event_samples: int,
) -> str:
    flags: list[str] = []
    nyquist_hz = sample_rate_hz / 2.0
    finite_peaks = [item.peak_frequency_hz for item in evidence if np.isfinite(item.peak_frequency_hz)]
    if finite_peaks and float(np.median(finite_peaks)) >= 0.80 * nyquist_hz:
        flags.append("near_nyquist_alias_risk")
    if event_samples <= window_samples:
        flags.append("single_analysis_window")
    if not any(np.isfinite(item.high_band_noise_excess_ratio) for item in evidence):
        flags.append("no_non_braking_noise_baseline")
    return ";".join(flags)


def _event_row(
    parent_braking_event_id: str,
    segment: pd.DataFrame,
    wheel: str,
    start: int,
    end: int,
    evidence: list[_WindowEvidence],
    sample_rate_hz: float,
    window_samples: int,
) -> dict[str, Any]:
    event_segment = segment.iloc[start : end + 1]
    slip = event_segment[f"wheel_{wheel}_slip_ratio"].to_numpy(dtype=float)
    loads = event_segment[f"wheel_{wheel}_load"]
    first = event_segment.iloc[0]
    last = event_segment.iloc[-1]
    mean_score = float(np.mean([item.activity_score for item in evidence]))
    alias_penalty = 0.85 if any(
        item.peak_frequency_hz >= 0.80 * sample_rate_hz / 2.0
        for item in evidence
        if np.isfinite(item.peak_frequency_hz)
    ) else 1.0
    confidence = float(np.clip(mean_score * alias_penalty, 0.0, 1.0))
    parent_id = str(parent_braking_event_id)
    return {
        "event_id": stable_id(parent_id, wheel, int(first["sample_index"])),
        "parent_braking_event_id": parent_id,
        "session_id": first["session_id"],
        "lap_id": first["lap_id"],
        "event_type": "abs_intervention_candidate",
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
        "mean_brake": float(event_segment["brake_n"].mean()),
        "peak_brake": float(event_segment["brake_n"].max()),
        "mean_wheel_load": float(loads.mean()),
        "mean_slip_ratio": float(np.mean(slip)),
        "minimum_slip_ratio": float(np.min(slip)),
        "detrended_slip_rms": float(np.sqrt(np.mean(_detrend(slip) ** 2))),
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
        "brake_high_band_power_fraction": float(
            np.mean([item.brake_high_band_power_fraction for item in evidence])
        ),
        "same_axle_high_band_correlation": float(
            np.mean([item.same_axle_high_band_correlation for item in evidence])
        ),
        "activity_score": mean_score,
        "confidence": confidence,
        "quality_flags": _quality_flags(
            evidence, sample_rate_hz, window_samples, len(event_segment)
        ),
    }


def detect_abs_activity(
    samples: pd.DataFrame,
    braking_events: pd.DataFrame,
    config: ProcessingConfig,
) -> pd.DataFrame:
    """Detect per-wheel spectral ABS activity inside previously detected braking events."""
    if samples.empty or braking_events.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    rows: list[dict[str, Any]] = []
    noise_floors = _noise_floors(samples, config)
    sample_groups = {
        (str(session_id), str(lap_id)): group.sort_values("sample_index").reset_index(drop=True)
        for (session_id, lap_id), group in samples.groupby(
            ["session_id", "lap_id"], sort=False
        )
    }
    for braking_event in braking_events.itertuples(index=False):
        lap_samples = sample_groups.get(
            (str(braking_event.session_id), str(braking_event.lap_id))
        )
        if lap_samples is None:
            continue
        segment = lap_samples[
            (lap_samples["sample_index"] >= int(braking_event.start_sample))
            & (lap_samples["sample_index"] <= int(braking_event.end_sample))
        ].reset_index(drop=True)
        if segment.empty:
            continue
        median_dt = float(segment["dt_s"].median())
        if not np.isfinite(median_dt) or median_dt <= 0:
            continue
        sample_rate_hz = 1.0 / median_dt
        high_max_hz = min(
            config.abs_high_frequency_max_hz,
            config.abs_nyquist_fraction * sample_rate_hz,
        )
        if high_max_hz <= config.abs_high_frequency_min_hz:
            continue
        window_samples = max(16, round(config.abs_analysis_window_s / median_dt))
        hop_samples = max(1, round(config.abs_analysis_hop_s / median_dt))
        starts = _window_starts(len(segment), window_samples, hop_samples)
        if not starts:
            continue

        for wheel in WHEELS:
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
                    noise_floors.get((str(braking_event.session_id), wheel), float("nan")),
                    config,
                )
                for start in starts
            ]
            mask = np.asarray([item.candidate for item in windows], dtype=bool)
            gap_windows = max(
                0,
                round(config.abs_event_gap_close_s / max(config.abs_analysis_hop_s, median_dt)),
            )
            mask = close_short_false_gaps(mask, gap_windows)
            for first_window, last_window in contiguous_true_runs(mask):
                selected = windows[first_window : last_window + 1]
                selected = [item for item in selected if item.candidate]
                if not selected:
                    continue
                start = windows[first_window].start
                end = windows[last_window].end
                duration_s = float(segment.iloc[start : end + 1]["dt_s"].sum())
                if duration_s < config.minimum_abs_event_s:
                    continue
                event = _event_row(
                    str(braking_event.event_id),
                    segment,
                    wheel,
                    start,
                    end,
                    selected,
                    sample_rate_hz,
                    window_samples,
                )
                rows.append(event)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)
