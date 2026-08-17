"""HRV feature helpers for the staged Biosigpy migration."""

from dataclasses import dataclass

import numpy as np
from biosigpy.hrv import removefp, tdmetrics
from biosigpy.tools import medfilt_threshold


@dataclass(frozen=True)
class TimeDomainHrvResult:
    raw_event_times: np.ndarray
    cleaned_event_times: np.ndarray
    removed_detection_mask: np.ndarray
    intervals_after_removefp: np.ndarray
    interval_outlier_mask: np.ndarray
    intervals: np.ndarray
    metrics: dict[str, float]

    @property
    def removed_fp_count(self):
        return int(np.count_nonzero(self.removed_detection_mask))

    @property
    def interval_outlier_count(self):
        return int(np.count_nonzero(self.interval_outlier_mask))

    @property
    def removed_rr_count(self):
        return self.removed_fp_count + self.interval_outlier_count

    @property
    def removed_rr_percentage(self):
        raw_interval_count = max(0, self.raw_event_times.size - 1)
        if raw_interval_count == 0:
            return np.nan
        return 100.0 * self.removed_rr_count / raw_interval_count


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
    """Remove false positives and RR outliers without filling gaps."""

    raw_event_times = np.asarray(event_times, dtype=float).flatten()
    cleaned_event_times = (
        removefp(raw_event_times)
        if raw_event_times.size
        else raw_event_times.copy()
    )
    removed_mask = ~np.isin(raw_event_times, cleaned_event_times)
    intervals_after_removefp = np.diff(cleaned_event_times)

    if intervals_after_removefp.size >= 2:
        interval_threshold = medfilt_threshold(
            intervals_after_removefp,
            50,
            1.5,
            1.5,
        )
        interval_outlier_mask = (
            intervals_after_removefp > interval_threshold
        )
    else:
        interval_outlier_mask = np.zeros(
            intervals_after_removefp.size,
            dtype=bool,
        )
    intervals = intervals_after_removefp[~interval_outlier_mask]

    if intervals.size:
        biosigpy_metrics = tdmetrics(intervals)
    else:
        biosigpy_metrics = {
            name: np.nan
            for name in ("mhr", "sdnn", "rmssd", "pnn50")
        }

    metrics = _custom_time_metrics(intervals, sampling_frequency)
    metrics.update(
        {
            "MHR": biosigpy_metrics["mhr"],
            "SDNN": biosigpy_metrics["sdnn"],
            "RMSSD": biosigpy_metrics["rmssd"],
            "PNN50": biosigpy_metrics["pnn50"],
        }
    )
    return TimeDomainHrvResult(
        raw_event_times=raw_event_times,
        cleaned_event_times=cleaned_event_times,
        removed_detection_mask=removed_mask,
        intervals_after_removefp=intervals_after_removefp,
        interval_outlier_mask=interval_outlier_mask,
        intervals=intervals,
        metrics=metrics,
    )
