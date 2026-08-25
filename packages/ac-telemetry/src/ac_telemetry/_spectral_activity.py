from __future__ import annotations

import numpy as np


def detrend(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return values - np.nanmean(values)
    x = np.linspace(-1.0, 1.0, len(values))
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return np.zeros_like(values)
    filled = np.interp(x, x[finite], values[finite])
    return filled - np.polyval(np.polyfit(x, filled, 2), x)


def spectrum(values: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    windowed = detrend(values) * np.hanning(len(values))
    frequencies = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate_hz)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    return frequencies, power


def band_power(frequencies: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies < high)
    return float(power[mask].sum())


def band_signal(values: np.ndarray, sample_rate_hz: float, low: float, high: float) -> np.ndarray:
    detrended = detrend(values)
    transformed = np.fft.rfft(detrended)
    frequencies = np.fft.rfftfreq(len(detrended), d=1.0 / sample_rate_hz)
    transformed[(frequencies < low) | (frequencies > high)] = 0
    return np.fft.irfft(transformed, n=len(detrended))


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= np.finfo(float).eps or np.std(right) <= np.finfo(float).eps:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def high_band_rms(
    values: np.ndarray,
    sample_rate_hz: float,
    high_min_hz: float,
    high_max_hz: float,
) -> float:
    filtered = band_signal(values, sample_rate_hz, high_min_hz, high_max_hz)
    return float(np.sqrt(np.mean(filtered**2)))


def window_starts(length: int, window_samples: int, hop_samples: int) -> list[int]:
    if length < window_samples:
        return []
    starts = list(range(0, length - window_samples + 1, hop_samples))
    last = length - window_samples
    if starts[-1] != last:
        starts.append(last)
    return starts
