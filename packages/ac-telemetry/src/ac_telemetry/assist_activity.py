from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from .config import ProcessingConfig
from .util import close_short_false_gaps, contiguous_true_runs, stable_id


# ABS design evidence from the controlled brake test in
# AC_250826-200255_O_ks_mercedes_amg_gt3_ks_silverstone1967_.acreplay.
# The first braking event of each lap was analysed after trimming 0.25 s from
# both ends and removing its quadratic trend. Replay sampling was 66.67 Hz.
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
# the axle, while the brake input remained low-frequency. This supports
# per-wheel feedback modulation downstream of the pedal. The observed band is
# close to the 33.3 Hz Nyquist limit and may contain aliases. These measurements
# are detector design evidence, not universal slip targets or ABS frequencies.
#
# AC replay data has no native TC torque-cut channel. TC detection uses the same
# observable mechanism: driven-wheel high-frequency slip suppression absent from
# the driver's throttle input. No labelled TC experiment has calibrated a
# universal band, so the default 15-32 Hz is provisional and configurable. The
# detector imposes no positive-slip trigger threshold: sustained wheelspin alone
# is not evidence of feedback intervention. Only rear wheels are considered
# because replay metadata does not expose the driven axle.

_ABS_WHEELS = ("fl", "fr", "rl", "rr")
_TC_WHEELS = ("rl", "rr")
_OPPOSITE_WHEEL = {"fl": "fr", "fr": "fl", "rl": "rr", "rr": "rl"}

_ABS_COLUMNS = [
    "event_id", "parent_braking_event_id", "session_id", "lap_id",
    "event_type", "detection_method", "wheel", "start_sample", "end_sample",
    "start_time_s", "end_time_s", "duration_s", "sample_count",
    "start_distance_m", "end_distance_m", "start_progress", "end_progress",
    "entry_speed_kmh", "exit_speed_kmh", "mean_brake", "peak_brake",
    "mean_wheel_load", "mean_slip_ratio", "minimum_slip_ratio",
    "detrended_slip_rms", "observed_peak_frequency_hz", "spectral_centroid_hz",
    "high_band_power_fraction", "high_to_low_power_ratio",
    "high_band_noise_excess_ratio", "brake_high_band_power_fraction",
    "same_axle_high_band_correlation", "activity_score", "confidence",
    "quality_flags",
]

_TC_COLUMNS = [
    "event_id", "parent_throttle_event_id", "session_id", "lap_id",
    "event_type", "detection_method", "wheel", "start_sample", "end_sample",
    "start_time_s", "end_time_s", "duration_s", "sample_count",
    "start_distance_m", "end_distance_m", "start_progress", "end_progress",
    "entry_speed_kmh", "exit_speed_kmh", "mean_throttle", "peak_throttle",
    "mean_brake", "mean_wheel_load", "mean_slip_ratio", "maximum_slip_ratio",
    "detrended_slip_rms", "observed_peak_frequency_hz", "spectral_centroid_hz",
    "high_band_power_fraction", "high_to_low_power_ratio",
    "high_band_noise_excess_ratio", "throttle_high_band_power_fraction",
    "same_axle_high_band_correlation", "activity_score", "confidence",
    "quality_flags",
]


@dataclass(frozen=True, slots=True)
class _ActivitySpec:
    kind: Literal["abs", "tc"]
    wheels: tuple[str, ...]
    parent_id_column: str
    event_type: str
    detection_method: str
    control_column: str
    columns: list[str]
    analysis_window_s: float
    analysis_hop_s: float
    low_frequency_min_hz: float
    low_frequency_max_hz: float
    high_frequency_min_hz: float
    high_frequency_max_hz: float
    nyquist_fraction: float
    min_high_to_low_power_ratio: float
    min_high_band_power_fraction: float
    max_control_high_band_power_fraction: float
    noise_floor_percentile: float
    min_high_band_noise_excess_ratio: float
    event_gap_close_s: float
    minimum_event_s: float


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
    control_high_band_power_fraction: float
    same_axle_high_band_correlation: float
    activity_score: float


def _abs_spec(config: ProcessingConfig) -> _ActivitySpec:
    return _ActivitySpec(
        kind="abs", wheels=_ABS_WHEELS,
        parent_id_column="parent_braking_event_id",
        event_type="abs_intervention_candidate",
        detection_method="slip_ratio_spectral_activity_v1",
        control_column="brake_n", columns=_ABS_COLUMNS,
        analysis_window_s=config.abs_analysis_window_s,
        analysis_hop_s=config.abs_analysis_hop_s,
        low_frequency_min_hz=config.abs_low_frequency_min_hz,
        low_frequency_max_hz=config.abs_low_frequency_max_hz,
        high_frequency_min_hz=config.abs_high_frequency_min_hz,
        high_frequency_max_hz=config.abs_high_frequency_max_hz,
        nyquist_fraction=config.abs_nyquist_fraction,
        min_high_to_low_power_ratio=config.abs_min_high_to_low_power_ratio,
        min_high_band_power_fraction=config.abs_min_high_band_power_fraction,
        max_control_high_band_power_fraction=config.abs_max_brake_high_band_power_fraction,
        noise_floor_percentile=config.abs_noise_floor_percentile,
        min_high_band_noise_excess_ratio=config.abs_min_high_band_noise_excess_ratio,
        event_gap_close_s=config.abs_event_gap_close_s,
        minimum_event_s=config.minimum_abs_event_s,
    )


def _tc_spec(config: ProcessingConfig) -> _ActivitySpec:
    return _ActivitySpec(
        kind="tc", wheels=_TC_WHEELS,
        parent_id_column="parent_throttle_event_id",
        event_type="tc_intervention_candidate",
        detection_method="driven_wheel_slip_spectral_activity_v1",
        control_column="throttle", columns=_TC_COLUMNS,
        analysis_window_s=config.tc_analysis_window_s,
        analysis_hop_s=config.tc_analysis_hop_s,
        low_frequency_min_hz=config.tc_low_frequency_min_hz,
        low_frequency_max_hz=config.tc_low_frequency_max_hz,
        high_frequency_min_hz=config.tc_high_frequency_min_hz,
        high_frequency_max_hz=config.tc_high_frequency_max_hz,
        nyquist_fraction=config.tc_nyquist_fraction,
        min_high_to_low_power_ratio=config.tc_min_high_to_low_power_ratio,
        min_high_band_power_fraction=config.tc_min_high_band_power_fraction,
        max_control_high_band_power_fraction=config.tc_max_throttle_high_band_power_fraction,
        noise_floor_percentile=config.tc_noise_floor_percentile,
        min_high_band_noise_excess_ratio=config.tc_min_high_band_noise_excess_ratio,
        event_gap_close_s=config.tc_event_gap_close_s,
        minimum_event_s=config.minimum_tc_event_s,
    )


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
    windowed = _detrend(values) * np.hanning(len(values))
    frequencies = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate_hz)
    return frequencies, np.abs(np.fft.rfft(windowed)) ** 2


def _band_power(frequencies: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    return float(power[(frequencies >= low) & (frequencies < high)].sum())


def _band_signal(values: np.ndarray, sample_rate_hz: float, low: float, high: float) -> np.ndarray:
    transformed = np.fft.rfft(_detrend(values))
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / sample_rate_hz)
    transformed[(frequencies < low) | (frequencies > high)] = 0
    return np.fft.irfft(transformed, n=len(values))


def _high_band_rms(values: np.ndarray, sample_rate_hz: float, spec: _ActivitySpec, high_max_hz: float) -> float:
    filtered = _band_signal(
        values, sample_rate_hz, spec.high_frequency_min_hz, high_max_hz
    )
    return float(np.sqrt(np.mean(filtered**2)))


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= np.finfo(float).eps or np.std(right) <= np.finfo(float).eps:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _window_starts(length: int, window_samples: int, hop_samples: int) -> list[int]:
    if length < window_samples:
        return []
    starts = list(range(0, length - window_samples + 1, hop_samples))
    last = length - window_samples
    if starts[-1] != last:
        starts.append(last)
    return starts


def _baseline_mask(
    samples: pd.DataFrame,
    spec: _ActivitySpec,
    config: ProcessingConfig,
) -> np.ndarray:
    if spec.kind == "abs":
        mask = (
            (samples["brake_n"] < config.brake_active_threshold)
            & (samples["speed_kmh"] >= config.minimum_brake_entry_speed_kmh)
        )
    else:
        mask = (
            (samples["throttle"] < config.throttle_event_threshold)
            & (samples["brake_n"] < config.brake_active_threshold)
            & (samples["speed_kmh"] >= config.wheelspin_minimum_speed_kmh)
        )
    return mask.to_numpy(dtype=bool)


def _noise_floors(
    samples: pd.DataFrame,
    spec: _ActivitySpec,
    config: ProcessingConfig,
) -> dict[tuple[str, str], float]:
    floors: dict[tuple[str, str], float] = {}
    for session_id, original in samples.groupby("session_id", sort=False):
        g = original.sort_values("sample_index").reset_index(drop=True)
        median_dt = float(g["dt_s"].median())
        if not np.isfinite(median_dt) or median_dt <= 0:
            continue
        sample_rate_hz = 1.0 / median_dt
        high_max_hz = min(
            spec.high_frequency_max_hz, spec.nyquist_fraction * sample_rate_hz
        )
        if high_max_hz <= spec.high_frequency_min_hz:
            continue
        window_samples = max(16, round(spec.analysis_window_s / median_dt))
        runs = contiguous_true_runs(_baseline_mask(g, spec, config))
        for wheel in spec.wheels:
            column = f"wheel_{wheel}_slip_ratio"
            if column not in g:
                continue
            rms_values: list[float] = []
            for start, end in runs:
                for offset in range(start, end - window_samples + 2, window_samples):
                    values = g[column].iloc[offset : offset + window_samples].to_numpy(float)
                    if len(values) == window_samples:
                        rms_values.append(
                            _high_band_rms(values, sample_rate_hz, spec, high_max_hz)
                        )
            floors[(str(session_id), wheel)] = (
                float(np.percentile(rms_values, spec.noise_floor_percentile))
                if rms_values else float("nan")
            )
    return floors


def _tc_window_allowed(
    segment: pd.DataFrame,
    start: int,
    end: int,
    sample_rate_hz: float,
    config: ProcessingConfig,
) -> bool:
    if float(segment["brake_n"].iloc[start : end + 1].mean()) >= config.brake_active_threshold:
        return False
    if "gear_physical" not in segment:
        return True
    gears = segment["gear_physical"].fillna(0).to_numpy(int)
    window_gears = gears[start : end + 1]
    stable_forward = (
        len(window_gears) > 0
        and np.all(window_gears == window_gears[0])
        and window_gears[0] > 0
    )
    shift_indices = np.flatnonzero(gears[1:] != gears[:-1]) + 1
    margin = round(config.tc_shift_exclusion_s * sample_rate_hz)
    away_from_shift = not np.any(
        (shift_indices >= start - margin) & (shift_indices <= end + margin)
    )
    return bool(stable_forward and away_from_shift)


def _window_evidence(
    segment: pd.DataFrame,
    wheel: str,
    start: int,
    end: int,
    sample_rate_hz: float,
    high_max_hz: float,
    noise_floor: float,
    spec: _ActivitySpec,
    config: ProcessingConfig,
) -> _WindowEvidence:
    wheel_values = segment[f"wheel_{wheel}_slip_ratio"].iloc[start : end + 1].to_numpy(float)
    opposite_values = segment[
        f"wheel_{_OPPOSITE_WHEEL[wheel]}_slip_ratio"
    ].iloc[start : end + 1].to_numpy(float)
    control_values = segment[spec.control_column].iloc[start : end + 1].to_numpy(float)
    frequencies, wheel_power = _spectrum(wheel_values, sample_rate_hz)
    _, control_power = _spectrum(control_values, sample_rate_hz)
    low_power = _band_power(
        frequencies, wheel_power, spec.low_frequency_min_hz, spec.low_frequency_max_hz
    )
    high_power = _band_power(
        frequencies, wheel_power, spec.high_frequency_min_hz, high_max_hz
    )
    analysis_power = _band_power(frequencies, wheel_power, 2.0, high_max_hz)
    control_high_power = _band_power(
        frequencies, control_power, spec.high_frequency_min_hz, high_max_hz
    )
    control_analysis_power = _band_power(frequencies, control_power, 2.0, high_max_hz)
    epsilon = np.finfo(float).eps
    high_to_low = high_power / max(low_power, epsilon)
    high_fraction = high_power / max(analysis_power, epsilon)
    control_high_fraction = control_high_power / max(control_analysis_power, epsilon)
    high_rms = _high_band_rms(wheel_values, sample_rate_hz, spec, high_max_hz)
    noise_excess = (
        high_rms / max(noise_floor, epsilon)
        if np.isfinite(noise_floor) else float("nan")
    )

    high_mask = (
        (frequencies >= spec.high_frequency_min_hz) & (frequencies < high_max_hz)
    )
    if high_mask.any() and high_power > epsilon:
        high_indices = np.flatnonzero(high_mask)
        peak_index = int(high_indices[np.argmax(wheel_power[high_mask])])
        peak_frequency = float(frequencies[peak_index])
        centroid = float(
            np.sum(frequencies[high_mask] * wheel_power[high_mask]) / high_power
        )
    else:
        peak_frequency = centroid = float("nan")

    wheel_high = _band_signal(
        wheel_values, sample_rate_hz, spec.high_frequency_min_hz, high_max_hz
    )
    opposite_high = _band_signal(
        opposite_values, sample_rate_hz, spec.high_frequency_min_hz, high_max_hz
    )
    axle_correlation = _safe_correlation(wheel_high, opposite_high)
    ratio_score = float(np.clip(
        np.log10(max(high_to_low, epsilon) / spec.min_high_to_low_power_ratio) + 1.0,
        0.0, 2.0,
    ) / 2.0)
    fraction_score = float(np.clip(
        (high_fraction - spec.min_high_band_power_fraction)
        / max(1.0 - spec.min_high_band_power_fraction, epsilon),
        0.0, 1.0,
    ))
    pedal_score = float(np.clip(
        1.0 - control_high_fraction / spec.max_control_high_band_power_fraction,
        0.0, 1.0,
    ))
    noise_score = (
        float(np.clip(
            np.log10(max(noise_excess, epsilon) / spec.min_high_band_noise_excess_ratio)
            + 1.0,
            0.0, 2.0,
        ) / 2.0)
        if np.isfinite(noise_excess) else 1.0
    )
    if spec.kind == "abs":
        independence_score = float(np.clip(1.0 - abs(axle_correlation), 0.0, 1.0))
        activity_score = (
            0.35 * ratio_score + 0.25 * fraction_score + 0.20 * pedal_score
            + 0.10 * independence_score + 0.10 * noise_score
        )
        allowed = True
    else:
        activity_score = (
            0.40 * ratio_score + 0.30 * fraction_score
            + 0.20 * pedal_score + 0.10 * noise_score
        )
        allowed = _tc_window_allowed(segment, start, end, sample_rate_hz, config)
    candidate = bool(
        allowed
        and high_to_low >= spec.min_high_to_low_power_ratio
        and high_fraction >= spec.min_high_band_power_fraction
        and control_high_fraction <= spec.max_control_high_band_power_fraction
        and (
            not np.isfinite(noise_excess)
            or noise_excess >= spec.min_high_band_noise_excess_ratio
        )
    )
    return _WindowEvidence(
        start, end, candidate, peak_frequency, centroid, high_fraction,
        high_to_low, noise_excess, control_high_fraction, axle_correlation,
        activity_score,
    )


def _quality_flags(
    evidence: list[_WindowEvidence],
    sample_rate_hz: float,
    window_samples: int,
    event_samples: int,
    spec: _ActivitySpec,
) -> str:
    flags: list[str] = []
    peaks = [item.peak_frequency_hz for item in evidence if np.isfinite(item.peak_frequency_hz)]
    if peaks and float(np.median(peaks)) >= 0.80 * sample_rate_hz / 2.0:
        flags.append("near_nyquist_alias_risk")
    if event_samples <= window_samples:
        flags.append("single_analysis_window")
    if not any(np.isfinite(item.high_band_noise_excess_ratio) for item in evidence):
        flags.append(
            "no_non_braking_noise_baseline"
            if spec.kind == "abs" else "no_non_throttle_noise_baseline"
        )
    if spec.kind == "tc":
        flags.append("rear_drive_assumption")
    return ";".join(flags)


def _event_row(
    parent_event_id: str,
    segment: pd.DataFrame,
    wheel: str,
    start: int,
    end: int,
    evidence: list[_WindowEvidence],
    sample_rate_hz: float,
    window_samples: int,
    spec: _ActivitySpec,
) -> dict[str, Any]:
    event_segment = segment.iloc[start : end + 1]
    slip = event_segment[f"wheel_{wheel}_slip_ratio"].to_numpy(float)
    first, last = event_segment.iloc[0], event_segment.iloc[-1]
    mean_score = float(np.mean([item.activity_score for item in evidence]))
    alias_penalty = 0.85 if any(
        item.peak_frequency_hz >= 0.80 * sample_rate_hz / 2.0
        for item in evidence if np.isfinite(item.peak_frequency_hz)
    ) else 1.0
    parent_id = str(parent_event_id)
    row: dict[str, Any] = {
        "event_id": stable_id(parent_id, wheel, int(first["sample_index"])),
        spec.parent_id_column: parent_id,
        "session_id": first["session_id"], "lap_id": first["lap_id"],
        "event_type": spec.event_type, "detection_method": spec.detection_method,
        "wheel": wheel, "start_sample": int(first["sample_index"]),
        "end_sample": int(last["sample_index"]),
        "start_time_s": float(first["lap_time_s"]),
        "end_time_s": float(last["lap_time_s"]),
        "duration_s": float(event_segment["dt_s"].sum()),
        "sample_count": len(event_segment),
        "start_distance_m": float(first["actual_distance_m"]),
        "end_distance_m": float(last["actual_distance_m"]),
        "start_progress": float(first["progress"]), "end_progress": float(last["progress"]),
        "entry_speed_kmh": float(first["speed_kmh"]),
        "exit_speed_kmh": float(last["speed_kmh"]),
        "mean_wheel_load": float(event_segment[f"wheel_{wheel}_load"].mean()),
        "mean_slip_ratio": float(np.mean(slip)),
        "detrended_slip_rms": float(np.sqrt(np.mean(_detrend(slip) ** 2))),
        "observed_peak_frequency_hz": float(np.nanmedian(
            [item.peak_frequency_hz for item in evidence]
        )),
        "spectral_centroid_hz": float(np.nanmedian(
            [item.spectral_centroid_hz for item in evidence]
        )),
        "high_band_power_fraction": float(np.mean(
            [item.high_band_power_fraction for item in evidence]
        )),
        "high_to_low_power_ratio": float(np.mean(
            [item.high_to_low_power_ratio for item in evidence]
        )),
        "high_band_noise_excess_ratio": float(np.nanmean(
            [item.high_band_noise_excess_ratio for item in evidence]
        )) if any(np.isfinite(item.high_band_noise_excess_ratio) for item in evidence) else float("nan"),
        "same_axle_high_band_correlation": float(np.mean(
            [item.same_axle_high_band_correlation for item in evidence]
        )),
        "activity_score": mean_score,
        "confidence": float(np.clip(mean_score * alias_penalty, 0.0, 1.0)),
        "quality_flags": _quality_flags(
            evidence, sample_rate_hz, window_samples, len(event_segment), spec
        ),
    }
    control_fraction = float(np.mean(
        [item.control_high_band_power_fraction for item in evidence]
    ))
    if spec.kind == "abs":
        row.update({
            "mean_brake": float(event_segment["brake_n"].mean()),
            "peak_brake": float(event_segment["brake_n"].max()),
            "minimum_slip_ratio": float(np.min(slip)),
            "brake_high_band_power_fraction": control_fraction,
        })
    else:
        row.update({
            "mean_throttle": float(event_segment["throttle"].mean()),
            "peak_throttle": float(event_segment["throttle"].max()),
            "mean_brake": float(event_segment["brake_n"].mean()),
            "maximum_slip_ratio": float(np.max(slip)),
            "throttle_high_band_power_fraction": control_fraction,
        })
    return row


def _detect_activity(
    samples: pd.DataFrame,
    parent_events: pd.DataFrame,
    config: ProcessingConfig,
    spec: _ActivitySpec,
) -> pd.DataFrame:
    if samples.empty or parent_events.empty:
        return pd.DataFrame(columns=spec.columns)
    rows: list[dict[str, Any]] = []
    noise_floors = _noise_floors(samples, spec, config)
    sample_groups = {
        (str(session_id), str(lap_id)): group.sort_values("sample_index").reset_index(drop=True)
        for (session_id, lap_id), group in samples.groupby(["session_id", "lap_id"], sort=False)
    }
    for parent_event in parent_events.itertuples(index=False):
        lap_samples = sample_groups.get((str(parent_event.session_id), str(parent_event.lap_id)))
        if lap_samples is None:
            continue
        segment = lap_samples[
            (lap_samples["sample_index"] >= int(parent_event.start_sample))
            & (lap_samples["sample_index"] <= int(parent_event.end_sample))
        ].reset_index(drop=True)
        if segment.empty:
            continue
        median_dt = float(segment["dt_s"].median())
        if not np.isfinite(median_dt) or median_dt <= 0:
            continue
        sample_rate_hz = 1.0 / median_dt
        high_max_hz = min(
            spec.high_frequency_max_hz, spec.nyquist_fraction * sample_rate_hz
        )
        if high_max_hz <= spec.high_frequency_min_hz:
            continue
        window_samples = max(16, round(spec.analysis_window_s / median_dt))
        hop_samples = max(1, round(spec.analysis_hop_s / median_dt))
        starts = _window_starts(len(segment), window_samples, hop_samples)
        for wheel in spec.wheels:
            required = [
                f"wheel_{wheel}_slip_ratio",
                f"wheel_{_OPPOSITE_WHEEL[wheel]}_slip_ratio",
                f"wheel_{wheel}_load",
            ]
            if any(column not in segment for column in required):
                continue
            windows = [
                _window_evidence(
                    segment, wheel, start, start + window_samples - 1,
                    sample_rate_hz, high_max_hz,
                    noise_floors.get((str(parent_event.session_id), wheel), float("nan")),
                    spec, config,
                )
                for start in starts
            ]
            mask = close_short_false_gaps(
                np.asarray([item.candidate for item in windows], dtype=bool),
                max(0, round(spec.event_gap_close_s / max(spec.analysis_hop_s, median_dt))),
            )
            for first_window, last_window in contiguous_true_runs(mask):
                selected = [
                    item for item in windows[first_window : last_window + 1]
                    if item.candidate
                ]
                if not selected:
                    continue
                start, end = windows[first_window].start, windows[last_window].end
                if float(segment.iloc[start : end + 1]["dt_s"].sum()) < spec.minimum_event_s:
                    continue
                rows.append(_event_row(
                    str(parent_event.event_id), segment, wheel, start, end, selected,
                    sample_rate_hz, window_samples, spec,
                ))
    return pd.DataFrame(rows, columns=spec.columns)


def detect_abs_activity(
    samples: pd.DataFrame,
    braking_events: pd.DataFrame,
    config: ProcessingConfig,
) -> pd.DataFrame:
    """Detect per-wheel spectral ABS activity inside braking events."""
    return _detect_activity(samples, braking_events, config, _abs_spec(config))


def detect_tc_activity(
    samples: pd.DataFrame,
    throttle_events: pd.DataFrame,
    config: ProcessingConfig,
) -> pd.DataFrame:
    """Detect per-rear-wheel spectral TC activity inside throttle events."""
    return _detect_activity(samples, throttle_events, config, _tc_spec(config))
