"""HRV feature helpers for the staged Biosigpy migration."""

from dataclasses import dataclass

import numpy as np
from biosigpy.hrv import removefp, tdmetrics
from scipy.integrate import trapezoid
from scipy.signal import lombscargle


@dataclass(frozen=True)
class TimeDomainHrvResult:
    raw_event_times: np.ndarray
    cleaned_event_times: np.ndarray
    removed_detection_mask: np.ndarray
    intervals: np.ndarray
    metrics: dict[str, float]

    @property
    def removed_count(self):
        return int(np.count_nonzero(self.removed_detection_mask))

    @property
    def removed_percentage(self):
        if self.raw_event_times.size == 0:
            return np.nan
        return 100.0 * self.removed_count / self.raw_event_times.size


def _custom_time_metrics(intervals, sampling_frequency):
    """Preserve PIP, PNNLS, PNNSS and AVNN on the cleaned TD series."""

    intervals = np.asarray(intervals, dtype=float)
    delta_nn = np.diff(intervals)
    threshold = 1.0 / float(sampling_frequency)
    acceleration = delta_nn <= -threshold
    deceleration = delta_nn >= threshold

    signs = np.zeros_like(delta_nn)
    signs[acceleration] = -1
    signs[deceleration] = 1

    num_deltas = len(delta_nn)
    inflections = sum(
        delta_nn[index + 1] * delta_nn[index] <= 0
        and delta_nn[index + 1] != delta_nn[index]
        for index in range(max(0, num_deltas - 1))
    )
    pip = inflections / (num_deltas - 1) * 100 if num_deltas > 1 else np.nan

    segments = []
    if num_deltas > 0:
        current_sign = signs[0]
        segment_length = 1
        for sign in signs[1:]:
            if sign == current_sign and sign != 0:
                segment_length += 1
            else:
                if current_sign != 0:
                    segments.append(segment_length)
                current_sign = sign
                segment_length = 1
        if current_sign != 0:
            segments.append(segment_length)

    segments = np.asarray(segments)
    if segments.size:
        pnnls = np.sum(segments[segments >= 3]) / num_deltas * 100
        pnnss = np.sum(segments[segments < 3]) / np.sum(segments) * 100
    else:
        pnnls = np.nan
        pnnss = np.nan

    return {
        "PIP": float(pip),
        "PNNLS": float(pnnls),
        "PNNSS": float(pnnss),
        "AVNN": float(np.mean(intervals)) if intervals.size else np.nan,
    }


def compute_time_domain_hrv(event_times, sampling_frequency):
    """Run removefp and TD metrics without gap filling."""

    raw_event_times = np.asarray(event_times, dtype=float).flatten()
    cleaned_event_times = (
        removefp(raw_event_times)
        if raw_event_times.size
        else raw_event_times.copy()
    )
    removed_mask = ~np.isin(raw_event_times, cleaned_event_times)
    intervals = np.diff(cleaned_event_times)

    if intervals.size:
        biosigpy_metrics = tdmetrics(intervals)
    else:
        biosigpy_metrics = {
            name: np.nan
            for name in ("mhr", "sdnn", "sdsd", "rmssd", "pnn50")
        }

    metrics = _custom_time_metrics(intervals, sampling_frequency)
    metrics.update(
        {
            "MHR": biosigpy_metrics["mhr"],
            "SDNN": biosigpy_metrics["sdnn"],
            "SDSD": biosigpy_metrics["sdsd"],
            "RMSSD": biosigpy_metrics["rmssd"],
            "PNN50": biosigpy_metrics["pnn50"],
        }
    )
    return TimeDomainHrvResult(
        raw_event_times=raw_event_times,
        cleaned_event_times=cleaned_event_times,
        removed_detection_mask=removed_mask,
        intervals=intervals,
        metrics=metrics,
    )


def compute_legacy_hf(intervals):
    """Keep the current HF feature isolated until the FD path is replaced."""

    intervals = np.asarray(intervals, dtype=float).flatten()
    if intervals.size <= 1:
        return np.nan
    elapsed_time = np.cumsum(intervals)
    frequencies = np.linspace(0.01, 0.5, 1000)
    spectrum = lombscargle(
        elapsed_time,
        intervals - np.mean(intervals),
        2 * np.pi * frequencies,
        normalize=True,
    )
    hf_band = (frequencies >= 0.15) & (frequencies <= 0.4)
    return float(trapezoid(spectrum[hf_band], frequencies[hf_band]))
