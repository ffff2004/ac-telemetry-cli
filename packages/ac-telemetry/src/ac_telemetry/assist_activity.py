from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from .config import ProcessingConfig
from .contract_types import ForeignKey, MergeMode, TableSpec, column_specs
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
# is not evidence of feedback intervention. Known vehicle profiles restrict TC
# analysis to driven wheels; unknown profiles preserve four-wheel candidates.

_ABS_WHEELS = ("fl", "fr", "rl", "rr")
_TC_WHEELS = _ABS_WHEELS
_OPPOSITE_WHEEL = {"fl": "fr", "fr": "fl", "rl": "rr", "rr": "rl"}

_ABS_COLUMNS = [
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
    "start_track_s_m",
    "end_track_s_m",
    "start_track_progress",
    "end_track_progress",
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

_TC_COLUMNS = [
    "event_id",
    "parent_throttle_event_id",
    "session_id",
    "lap_id",
    "event_type",
    "detection_method",
    "wheel",
    "driven_status",
    "start_sample",
    "end_sample",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "sample_count",
    "start_track_s_m",
    "end_track_s_m",
    "start_track_progress",
    "end_track_progress",
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

ACTIVITY_TABLE_SPECS = (
    TableSpec(
        "events/abs_activity",
        column_specs(
            _ABS_COLUMNS,
            required=frozenset({"event_id"}),
            non_nullable=frozenset({"event_id"}),
        ),
        ("event_id",),
        False,
        MergeMode.KEYED,
        (
            ForeignKey(("event_id",), "events/index", ("event_id",)),
            ForeignKey(("parent_braking_event_id",), "events/index", ("event_id",)),
            ForeignKey(("session_id",), "sessions", ("session_id",)),
            ForeignKey(("lap_id",), "laps", ("lap_id",)),
            ForeignKey(("session_id", "lap_id"), "laps", ("session_id", "lap_id")),
        ),
    ),
    TableSpec(
        "events/tc_activity",
        column_specs(
            _TC_COLUMNS,
            required=frozenset({"event_id"}),
            non_nullable=frozenset({"event_id"}),
        ),
        ("event_id",),
        False,
        MergeMode.KEYED,
        (
            ForeignKey(("event_id",), "events/index", ("event_id",)),
            ForeignKey(("parent_throttle_event_id",), "events/index", ("event_id",)),
            ForeignKey(("session_id",), "sessions", ("session_id",)),
            ForeignKey(("lap_id",), "laps", ("lap_id",)),
            ForeignKey(("session_id", "lap_id"), "laps", ("session_id", "lap_id")),
        ),
    ),
)


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


@dataclass(slots=True)
class _WindowSignals:
    """Spectral values shared by all wheels in one analysis window."""

    geometry: _SpectralGeometry
    powers: dict[str, np.ndarray]
    high_signals: dict[str, np.ndarray]
    high_rms: dict[str, float]
    allowed: bool


@dataclass(frozen=True, slots=True)
class _DetrendGeometry:
    x: np.ndarray
    vandermonde: np.ndarray
    coefficients: np.ndarray


@dataclass(frozen=True, slots=True)
class _SpectralGeometry:
    frequencies: np.ndarray
    hanning: np.ndarray
    signal_high_mask: np.ndarray
    low_band_mask: np.ndarray
    high_band_mask: np.ndarray
    analysis_band_mask: np.ndarray


def _abs_spec(config: ProcessingConfig) -> _ActivitySpec:
    return _ActivitySpec(
        kind="abs",
        wheels=_ABS_WHEELS,
        parent_id_column="parent_braking_event_id",
        event_type="abs_intervention_candidate",
        detection_method="slip_ratio_spectral_activity_v1",
        control_column="brake_n",
        columns=_ABS_COLUMNS,
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
        kind="tc",
        wheels=_TC_WHEELS,
        parent_id_column="parent_throttle_event_id",
        event_type="tc_intervention_candidate",
        detection_method="driven_wheel_slip_spectral_activity_v1",
        control_column="throttle",
        columns=_TC_COLUMNS,
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


@lru_cache(maxsize=64)
def _detrend_geometry(length: int) -> _DetrendGeometry:
    x = np.linspace(-1.0, 1.0, length)
    vandermonde = np.vander(x, 3)
    scale = np.sqrt(np.sum(vandermonde * vandermonde, axis=0))
    scaled = vandermonde / scale
    coefficients = np.linalg.pinv(scaled) / scale[:, np.newaxis]
    for values in (x, vandermonde, coefficients):
        values.setflags(write=False)
    return _DetrendGeometry(x, vandermonde, coefficients)


def _detrend_many(values: np.ndarray) -> np.ndarray:
    """Quadratically detrend a batch of equally sized windows."""
    windows = np.asarray(values, dtype=float)
    if windows.ndim != 2:
        raise ValueError("Expected a two-dimensional window array")
    if windows.shape[1] < 3:
        return windows - np.nanmean(windows, axis=1, keepdims=True)

    geometry = _detrend_geometry(windows.shape[1])
    finite = np.isfinite(windows)
    filled = windows if finite.all() else windows.copy()
    for index in np.flatnonzero(~finite.all(axis=1)):
        present = finite[index]
        if present.sum() < 3:
            filled[index] = 0.0
        else:
            filled[index] = np.interp(
                geometry.x,
                geometry.x[present],
                windows[index, present],
            )
    trend = (filled @ geometry.coefficients.T) @ geometry.vandermonde.T
    return filled - trend


def _detrend(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return values - np.nanmean(values)
    if np.isfinite(values).sum() < 3:
        return np.zeros_like(values)
    return _detrend_many(values[np.newaxis, :])[0]


@lru_cache(maxsize=128)
def _spectral_geometry(
    window_samples: int,
    sample_rate_hz: float,
    low_frequency_min_hz: float,
    low_frequency_max_hz: float,
    high_frequency_min_hz: float,
    high_max_hz: float,
) -> _SpectralGeometry:
    frequencies = np.fft.rfftfreq(window_samples, d=1.0 / sample_rate_hz)
    signal_high_mask = (frequencies < high_frequency_min_hz) | (
        frequencies > high_max_hz
    )
    low_band_mask = (frequencies >= low_frequency_min_hz) & (
        frequencies < low_frequency_max_hz
    )
    high_band_mask = (frequencies >= high_frequency_min_hz) & (
        frequencies < high_max_hz
    )
    analysis_band_mask = (frequencies >= 2.0) & (frequencies < high_max_hz)
    hanning = np.hanning(window_samples)
    for values in (
        frequencies,
        signal_high_mask,
        low_band_mask,
        high_band_mask,
        analysis_band_mask,
        hanning,
    ):
        values.setflags(write=False)
    return _SpectralGeometry(
        frequencies,
        hanning,
        signal_high_mask,
        low_band_mask,
        high_band_mask,
        analysis_band_mask,
    )


def _high_band_signal(
    detrended: np.ndarray,
    high_mask: np.ndarray,
) -> np.ndarray:
    transformed = np.fft.rfft(detrended)
    transformed[high_mask] = 0
    return np.fft.irfft(transformed, n=len(detrended))


def _band_signal(
    values: np.ndarray, sample_rate_hz: float, low: float, high: float
) -> np.ndarray:
    detrended = _detrend(values)
    frequencies = np.fft.rfftfreq(len(detrended), d=1.0 / sample_rate_hz)
    high_mask = (frequencies < low) | (frequencies > high)
    return _high_band_signal(detrended, high_mask)


def _high_band_rms(
    values: np.ndarray, sample_rate_hz: float, spec: _ActivitySpec, high_max_hz: float
) -> float:
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
        mask = (samples["brake_n"] < config.brake_active_threshold) & (
            samples["speed_kmh"] >= config.minimum_brake_entry_speed_kmh
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
                    values = (
                        g[column].iloc[offset : offset + window_samples].to_numpy(float)
                    )
                    if len(values) == window_samples:
                        rms_values.append(
                            _high_band_rms(values, sample_rate_hz, spec, high_max_hz)
                        )
            floors[(str(session_id), wheel)] = (
                float(np.percentile(rms_values, spec.noise_floor_percentile))
                if rms_values
                else float("nan")
            )
    return floors


def _tc_window_allowed(
    brake_values: np.ndarray,
    gear_values: np.ndarray | None,
    shift_indices: np.ndarray,
    start: int,
    end: int,
    shift_margin: int,
    brake_active_threshold: float,
) -> bool:
    if float(np.mean(brake_values[start:end])) >= brake_active_threshold:
        return False
    if gear_values is None:
        return True
    window_gears = gear_values[start:end]
    stable_forward = (
        len(window_gears) > 0
        and bool(np.all(window_gears == window_gears[0]))
        and window_gears[0] > 0
    )
    away_from_shift = not np.any(
        (shift_indices >= start - shift_margin)
        & (shift_indices <= end - 1 + shift_margin)
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
    cached: _WindowSignals,
) -> _WindowEvidence:
    if not cached.allowed:
        return _WindowEvidence(
            start,
            end,
            False,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0.0,
        )
    wheel_column = f"wheel_{wheel}_slip_ratio"
    opposite_column = f"wheel_{_OPPOSITE_WHEEL[wheel]}_slip_ratio"
    geometry = cached.geometry
    frequencies = geometry.frequencies
    wheel_power = cached.powers[wheel_column]
    control_power = cached.powers[spec.control_column]
    wheel_high = cached.high_signals[wheel_column]
    opposite_high = cached.high_signals[opposite_column]
    high_rms = cached.high_rms[wheel_column]
    allowed = cached.allowed
    low_power = float(wheel_power[geometry.low_band_mask].sum())
    high_power = float(wheel_power[geometry.high_band_mask].sum())
    analysis_power = float(wheel_power[geometry.analysis_band_mask].sum())
    control_high_power = float(control_power[geometry.high_band_mask].sum())
    control_analysis_power = float(control_power[geometry.analysis_band_mask].sum())
    epsilon = np.finfo(float).eps
    high_to_low = high_power / max(low_power, epsilon)
    high_fraction = high_power / max(analysis_power, epsilon)
    control_high_fraction = control_high_power / max(control_analysis_power, epsilon)
    noise_excess = (
        high_rms / max(noise_floor, epsilon)
        if np.isfinite(noise_floor)
        else float("nan")
    )

    high_mask = geometry.high_band_mask
    if high_mask.any() and high_power > epsilon:
        high_indices = np.flatnonzero(high_mask)
        peak_index = int(high_indices[np.argmax(wheel_power[high_mask])])
        peak_frequency = float(frequencies[peak_index])
        centroid = float(
            np.sum(frequencies[high_mask] * wheel_power[high_mask]) / high_power
        )
    else:
        peak_frequency = centroid = float("nan")

    axle_correlation = _safe_correlation(wheel_high, opposite_high)
    ratio_score = float(
        np.clip(
            np.log10(max(high_to_low, epsilon) / spec.min_high_to_low_power_ratio)
            + 1.0,
            0.0,
            2.0,
        )
        / 2.0
    )
    fraction_score = float(
        np.clip(
            (high_fraction - spec.min_high_band_power_fraction)
            / max(1.0 - spec.min_high_band_power_fraction, epsilon),
            0.0,
            1.0,
        )
    )
    pedal_score = float(
        np.clip(
            1.0 - control_high_fraction / spec.max_control_high_band_power_fraction,
            0.0,
            1.0,
        )
    )
    noise_score = (
        float(
            np.clip(
                np.log10(
                    max(noise_excess, epsilon) / spec.min_high_band_noise_excess_ratio
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
    if spec.kind == "abs":
        independence_score = float(np.clip(1.0 - abs(axle_correlation), 0.0, 1.0))
        activity_score = (
            0.35 * ratio_score
            + 0.25 * fraction_score
            + 0.20 * pedal_score
            + 0.10 * independence_score
            + 0.10 * noise_score
        )
    else:
        activity_score = (
            0.40 * ratio_score
            + 0.30 * fraction_score
            + 0.20 * pedal_score
            + 0.10 * noise_score
        )
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
        start,
        end,
        candidate,
        peak_frequency,
        centroid,
        high_fraction,
        high_to_low,
        noise_excess,
        control_high_fraction,
        axle_correlation,
        activity_score,
    )


def _build_window_signals(
    segment: pd.DataFrame,
    starts: list[int],
    window_samples: int,
    sample_rate_hz: float,
    high_max_hz: float,
    spec: _ActivitySpec,
    config: ProcessingConfig,
) -> list[_WindowSignals]:
    """Cache spectral values shared by all wheels in each analysis window."""
    wheel_columns = list(
        dict.fromkeys(
            [
                f"wheel_{wheel}_slip_ratio"
                for wheel in (
                    *spec.wheels,
                    *(_OPPOSITE_WHEEL[wheel] for wheel in spec.wheels),
                )
            ]
        )
    )
    wheel_columns = [column for column in wheel_columns if column in segment]
    signal_arrays = {
        column: segment[column].to_numpy(float) for column in wheel_columns
    }
    control_values = segment[spec.control_column].to_numpy(float)
    brake_values = segment["brake_n"].to_numpy(float) if spec.kind == "tc" else None
    gear_values = None
    shift_indices = np.empty(0, dtype=np.int64)
    if spec.kind == "tc" and "gear_physical" in segment:
        gear_values = segment["gear_physical"].fillna(0).to_numpy(int)
        shift_indices = np.flatnonzero(gear_values[1:] != gear_values[:-1]) + 1
    margin = (
        round(config.tc_shift_exclusion_s * sample_rate_hz) if spec.kind == "tc" else 0
    )
    geometry = _spectral_geometry(
        window_samples,
        sample_rate_hz,
        spec.low_frequency_min_hz,
        spec.low_frequency_max_hz,
        spec.high_frequency_min_hz,
        high_max_hz,
    )
    allowed_by_start = np.ones(len(starts), dtype=bool)
    if spec.kind == "tc":
        assert brake_values is not None
        allowed_by_start = np.asarray(
            [
                _tc_window_allowed(
                    brake_values,
                    gear_values,
                    shift_indices,
                    start,
                    start + window_samples,
                    margin,
                    config.brake_active_threshold,
                )
                for start in starts
            ],
            dtype=bool,
        )
    cached_windows = [
        _WindowSignals(geometry, {}, {}, {}, bool(allowed))
        for allowed in allowed_by_start
    ]
    valid_indexes = np.flatnonzero(allowed_by_start)
    if not len(valid_indexes) or not wheel_columns:
        return cached_windows

    signal_values = np.stack([signal_arrays[column] for column in wheel_columns])
    signal_windows = np.lib.stride_tricks.sliding_window_view(
        signal_values, window_samples, axis=1
    )[:, starts, :][:, valid_indexes, :]
    detrended = _detrend_many(signal_windows.reshape(-1, window_samples)).reshape(
        signal_windows.shape
    )
    power_values = np.abs(np.fft.rfft(detrended * geometry.hanning, axis=2)) ** 2
    high_transformed = np.fft.rfft(detrended, axis=2)
    high_transformed[:, :, geometry.signal_high_mask] = 0
    high_signals = np.fft.irfft(high_transformed, n=window_samples, axis=2)
    high_rms = np.sqrt(np.mean(high_signals**2, axis=2))

    control_windows = np.lib.stride_tricks.sliding_window_view(
        control_values, window_samples
    )[starts][valid_indexes]
    control_detrended = _detrend_many(control_windows)
    control_powers = (
        np.abs(np.fft.rfft(control_detrended * geometry.hanning, axis=1)) ** 2
    )

    for result_index, window_index in enumerate(valid_indexes):
        powers = {
            column: power_values[column_index, result_index]
            for column_index, column in enumerate(wheel_columns)
        }
        powers[spec.control_column] = control_powers[result_index]
        window_high_signals = {
            column: high_signals[column_index, result_index]
            for column_index, column in enumerate(wheel_columns)
        }
        window_high_rms = {
            column: float(high_rms[column_index, result_index])
            for column_index, column in enumerate(wheel_columns)
        }
        cached_windows[window_index] = _WindowSignals(
            geometry,
            powers,
            window_high_signals,
            window_high_rms,
            True,
        )
    return cached_windows


def _quality_flags(
    evidence: list[_WindowEvidence],
    sample_rate_hz: float,
    window_samples: int,
    event_samples: int,
    spec: _ActivitySpec,
    driven_status: str | None = None,
) -> str:
    flags: list[str] = []
    peaks = [
        item.peak_frequency_hz
        for item in evidence
        if np.isfinite(item.peak_frequency_hz)
    ]
    if peaks and float(np.median(peaks)) >= 0.80 * sample_rate_hz / 2.0:
        flags.append("near_nyquist_alias_risk")
    if event_samples <= window_samples:
        flags.append("single_analysis_window")
    if not any(np.isfinite(item.high_band_noise_excess_ratio) for item in evidence):
        flags.append(
            "no_non_braking_noise_baseline"
            if spec.kind == "abs"
            else "no_non_throttle_noise_baseline"
        )
    if spec.kind == "tc" and driven_status == "unknown":
        flags.append("unknown_drivetrain")
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
    driven_status: str | None = None,
) -> dict[str, Any]:
    event_segment = segment.iloc[start : end + 1]
    slip = event_segment[f"wheel_{wheel}_slip_ratio"].to_numpy(float)
    first, last = event_segment.iloc[0], event_segment.iloc[-1]
    mean_score = float(np.mean([item.activity_score for item in evidence]))
    alias_penalty = (
        0.85
        if any(
            item.peak_frequency_hz >= 0.80 * sample_rate_hz / 2.0
            for item in evidence
            if np.isfinite(item.peak_frequency_hz)
        )
        else 1.0
    )
    parent_id = str(parent_event_id)
    row: dict[str, Any] = {
        "event_id": stable_id(parent_id, wheel, int(first["sample_index"])),
        spec.parent_id_column: parent_id,
        "session_id": first["session_id"],
        "lap_id": first["lap_id"],
        "event_type": spec.event_type,
        "detection_method": spec.detection_method,
        "wheel": wheel,
        "start_sample": int(first["sample_index"]),
        "end_sample": int(last["sample_index"]),
        "start_time_s": float(first["lap_time_s"]),
        "end_time_s": float(last["lap_time_s"]),
        "duration_s": float(event_segment["dt_s"].sum()),
        "sample_count": len(event_segment),
        "start_track_s_m": float(first["track_s_m"]),
        "end_track_s_m": float(last["track_s_m"]),
        "start_track_progress": float(first["track_progress"]),
        "end_track_progress": float(last["track_progress"]),
        "entry_speed_kmh": float(first["speed_kmh"]),
        "exit_speed_kmh": float(last["speed_kmh"]),
        "mean_wheel_load": float(event_segment[f"wheel_{wheel}_load"].mean()),
        "mean_slip_ratio": float(np.mean(slip)),
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
        "same_axle_high_band_correlation": float(
            np.mean([item.same_axle_high_band_correlation for item in evidence])
        ),
        "activity_score": mean_score,
        "confidence": float(np.clip(mean_score * alias_penalty, 0.0, 1.0)),
        "quality_flags": _quality_flags(
            evidence,
            sample_rate_hz,
            window_samples,
            len(event_segment),
            spec,
            driven_status,
        ),
    }
    control_fraction = float(
        np.mean([item.control_high_band_power_fraction for item in evidence])
    )
    if spec.kind == "abs":
        row.update(
            {
                "mean_brake": float(event_segment["brake_n"].mean()),
                "peak_brake": float(event_segment["brake_n"].max()),
                "minimum_slip_ratio": float(np.min(slip)),
                "brake_high_band_power_fraction": control_fraction,
            }
        )
    else:
        row.update(
            {
                "driven_status": driven_status,
                "mean_throttle": float(event_segment["throttle"].mean()),
                "peak_throttle": float(event_segment["throttle"].max()),
                "mean_brake": float(event_segment["brake_n"].mean()),
                "maximum_slip_ratio": float(np.max(slip)),
                "throttle_high_band_power_fraction": control_fraction,
            }
        )
    return row


def _detect_activity(
    samples: pd.DataFrame,
    parent_events: pd.DataFrame,
    config: ProcessingConfig,
    spec: _ActivitySpec,
    driven_wheels_by_session: Mapping[str, frozenset[str] | None] | None = None,
) -> pd.DataFrame:
    if samples.empty or parent_events.empty:
        return pd.DataFrame(columns=spec.columns)
    rows: list[dict[str, Any]] = []
    noise_floors = _noise_floors(samples, spec, config)
    sample_groups: dict[tuple[str, str], pd.DataFrame] = {}
    for keys, group in samples.groupby(["session_id", "lap_id"], sort=False):
        session_id, lap_id = cast(tuple[Any, Any], keys)
        sample_groups[(str(session_id), str(lap_id))] = group.sort_values(
            "sample_index"
        ).reset_index(drop=True)
    for parent_event in parent_events.itertuples(index=False):
        session_id = str(parent_event.session_id)
        lap_samples = sample_groups.get(
            (str(parent_event.session_id), str(parent_event.lap_id))
        )
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
        cached_windows = _build_window_signals(
            segment,
            starts,
            window_samples,
            sample_rate_hz,
            high_max_hz,
            spec,
            config,
        )
        known_driven_wheels = (
            driven_wheels_by_session.get(session_id)
            if driven_wheels_by_session is not None
            else None
        )
        wheels = (
            tuple(wheel for wheel in spec.wheels if wheel in known_driven_wheels)
            if spec.kind == "tc" and known_driven_wheels is not None
            else spec.wheels
        )
        driven_status = (
            "driven"
            if spec.kind == "tc" and known_driven_wheels is not None
            else ("unknown" if spec.kind == "tc" else None)
        )
        for wheel in wheels:
            required = [
                f"wheel_{wheel}_slip_ratio",
                f"wheel_{_OPPOSITE_WHEEL[wheel]}_slip_ratio",
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
                        (str(parent_event.session_id), wheel), float("nan")
                    ),
                    spec,
                    cached,
                )
                for start, cached in zip(starts, cached_windows, strict=True)
            ]
            mask = close_short_false_gaps(
                np.asarray([item.candidate for item in windows], dtype=bool),
                max(
                    0,
                    round(spec.event_gap_close_s / max(spec.analysis_hop_s, median_dt)),
                ),
            )
            for first_window, last_window in contiguous_true_runs(mask):
                selected = [
                    item
                    for item in windows[first_window : last_window + 1]
                    if item.candidate
                ]
                if not selected:
                    continue
                start, end = windows[first_window].start, windows[last_window].end
                if (
                    float(segment.iloc[start : end + 1]["dt_s"].sum())
                    < spec.minimum_event_s
                ):
                    continue
                rows.append(
                    _event_row(
                        str(parent_event.event_id),
                        segment,
                        wheel,
                        start,
                        end,
                        selected,
                        sample_rate_hz,
                        window_samples,
                        spec,
                        driven_status,
                    )
                )
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
    driven_wheels_by_session: Mapping[str, frozenset[str] | None] | None = None,
) -> pd.DataFrame:
    """Detect spectral TC activity on known or potentially driven wheels."""
    return _detect_activity(
        samples,
        throttle_events,
        config,
        _tc_spec(config),
        driven_wheels_by_session,
    )
