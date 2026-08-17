"""Signal-quality gates for ECG windows used by HRV processing."""

from dataclasses import dataclass
import math


DEFAULT_MIN_HEART_RATE_BPM = 30.0
DEFAULT_MAX_HEART_RATE_BPM = 220.0
DEFAULT_MAX_ALTERED_FRACTION = 0.25


@dataclass(frozen=True)
class EcgSegmentQuality:
    """Quality decision and evidence for one ECG processing window."""

    is_valid: bool
    detection_count: int
    interval_count: int
    altered_count: int
    altered_fraction: float
    minimum_detections: int
    maximum_detections: int
    maximum_altered_fraction: float
    reasons: tuple[str, ...]


def evaluate_ecg_segment_quality(
    detection_count,
    altered_count,
    duration_seconds,
    *,
    min_heart_rate_bpm=DEFAULT_MIN_HEART_RATE_BPM,
    max_heart_rate_bpm=DEFAULT_MAX_HEART_RATE_BPM,
    max_altered_fraction=DEFAULT_MAX_ALTERED_FRACTION,
):
    """Reject windows with implausible counts or excessive RR alteration.

    Counts are evaluated over the actual segment duration.  The altered
    fraction uses the number of R-R intervals as its denominator, matching the
    output of the interval-cleaning stage.
    """

    detection_count = int(detection_count)
    altered_count = int(altered_count)
    duration_seconds = float(duration_seconds)
    min_heart_rate_bpm = float(min_heart_rate_bpm)
    max_heart_rate_bpm = float(max_heart_rate_bpm)
    max_altered_fraction = float(max_altered_fraction)

    if detection_count < 0 or altered_count < 0:
        raise ValueError("Detection and altered counts must be non-negative.")
    if duration_seconds <= 0:
        raise ValueError("Segment duration must be positive.")
    if min_heart_rate_bpm <= 0 or max_heart_rate_bpm <= min_heart_rate_bpm:
        raise ValueError("Heart-rate limits must be positive and ordered.")
    if not 0 <= max_altered_fraction <= 1:
        raise ValueError("Maximum altered fraction must be between 0 and 1.")

    interval_count = max(0, detection_count - 1)
    minimum_intervals = math.ceil(
        duration_seconds * min_heart_rate_bpm / 60.0
    )
    maximum_intervals = math.floor(
        duration_seconds * max_heart_rate_bpm / 60.0
    )
    minimum_detections = minimum_intervals + 1
    maximum_detections = maximum_intervals + 1
    altered_fraction = (
        altered_count / interval_count if interval_count else 0.0
    )

    reasons = []
    if detection_count < minimum_detections:
        reasons.append(
            f"Too few R-wave detections: {detection_count} < "
            f"{minimum_detections}."
        )
    if detection_count > maximum_detections:
        reasons.append(
            f"Too many R-wave detections: {detection_count} > "
            f"{maximum_detections}."
        )
    if altered_fraction > max_altered_fraction:
        reasons.append(
            f"Too many altered R-R intervals: {altered_fraction:.1%} > "
            f"{max_altered_fraction:.1%}."
        )

    return EcgSegmentQuality(
        is_valid=not reasons,
        detection_count=detection_count,
        interval_count=interval_count,
        altered_count=altered_count,
        altered_fraction=altered_fraction,
        minimum_detections=minimum_detections,
        maximum_detections=maximum_detections,
        maximum_altered_fraction=max_altered_fraction,
        reasons=tuple(reasons),
    )
