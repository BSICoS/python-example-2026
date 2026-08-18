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

    metrics = {
        "MHR": biosigpy_metrics["mhr"],
        "SDNN": biosigpy_metrics["sdnn"],
        "RMSSD": biosigpy_metrics["rmssd"],
        "PNN50": biosigpy_metrics["pnn50"],
    }
    return TimeDomainHrvResult(
        raw_event_times=raw_event_times,
        cleaned_event_times=cleaned_event_times,
        removed_detection_mask=removed_mask,
        intervals_after_removefp=intervals_after_removefp,
        interval_outlier_mask=interval_outlier_mask,
        intervals=intervals,
        metrics=metrics,
    )
