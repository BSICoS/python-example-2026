"""Signal-quality gates for ECG windows used by HRV processing."""

from dataclasses import dataclass
import math

import numpy as np


DEFAULT_MIN_HEART_RATE_BPM = 30.0
DEFAULT_MAX_HEART_RATE_BPM = 220.0
DEFAULT_MAX_REMOVED_FRACTION = 0.25
DEFAULT_AMPLITUDE_BLOCK_SECONDS = 10.0
DEFAULT_MAX_AMPLITUDE_SPREAD_RATIO = 10.0


@dataclass(frozen=True)
class EcgSegmentQuality:
    """Quality decision and evidence for one ECG processing window."""

    is_valid: bool
    raw_detection_count: int
    cleaned_detection_count: int
    removed_detection_count: int
    removed_fraction: float
    amplitude_spread_ratio: float
    minimum_detections: int
    maximum_detections: int
    maximum_removed_fraction: float
    maximum_amplitude_spread_ratio: float
    reasons: tuple[str, ...]


def compute_ecg_amplitude_spread_ratio(
    ecg_signal,
    sampling_frequency,
    *,
    block_seconds=DEFAULT_AMPLITUDE_BLOCK_SECONDS,
):
    """Measure robust block-amplitude instability without assuming ECG units."""

    signal = np.asarray(ecg_signal, dtype=float).flatten()
    sampling_frequency = float(sampling_frequency)
    block_seconds = float(block_seconds)
    if signal.size == 0:
        return np.nan
    if sampling_frequency <= 0 or block_seconds <= 0:
        raise ValueError("Sampling frequency and block duration must be positive.")

    block_samples = max(1, int(round(sampling_frequency * block_seconds)))
    amplitudes = []
    for start in range(0, signal.size, block_samples):
        block = signal[start : start + block_samples]
        finite = block[np.isfinite(block)]
        if finite.size:
            amplitudes.append(
                np.percentile(finite, 95) - np.percentile(finite, 5)
            )

    amplitudes = np.asarray(amplitudes, dtype=float)
    amplitudes = amplitudes[amplitudes > np.finfo(float).eps]
    if amplitudes.size == 0:
        return np.inf

    median_amplitude = np.median(amplitudes)
    return float(np.percentile(amplitudes, 90) / median_amplitude)


def evaluate_ecg_segment_quality(
    raw_detection_count,
    cleaned_detection_count,
    duration_seconds,
    *,
    amplitude_spread_ratio=np.nan,
    min_heart_rate_bpm=DEFAULT_MIN_HEART_RATE_BPM,
    max_heart_rate_bpm=DEFAULT_MAX_HEART_RATE_BPM,
    max_removed_fraction=DEFAULT_MAX_REMOVED_FRACTION,
    max_amplitude_spread_ratio=DEFAULT_MAX_AMPLITUDE_SPREAD_RATIO,
):
    """Reject implausible counts, excessive removal, or unstable ECG amplitude."""

    raw_detection_count = int(raw_detection_count)
    cleaned_detection_count = int(cleaned_detection_count)
    duration_seconds = float(duration_seconds)
    amplitude_spread_ratio = float(amplitude_spread_ratio)
    min_heart_rate_bpm = float(min_heart_rate_bpm)
    max_heart_rate_bpm = float(max_heart_rate_bpm)
    max_removed_fraction = float(max_removed_fraction)
    max_amplitude_spread_ratio = float(max_amplitude_spread_ratio)

    if raw_detection_count < 0 or cleaned_detection_count < 0:
        raise ValueError("Detection counts must be non-negative.")
    if cleaned_detection_count > raw_detection_count:
        raise ValueError("Cleaned detections cannot exceed raw detections.")
    if duration_seconds <= 0:
        raise ValueError("Segment duration must be positive.")
    if min_heart_rate_bpm <= 0 or max_heart_rate_bpm <= min_heart_rate_bpm:
        raise ValueError("Heart-rate limits must be positive and ordered.")
    if not 0 <= max_removed_fraction <= 1:
        raise ValueError("Maximum removed fraction must be between 0 and 1.")
    if max_amplitude_spread_ratio <= 0:
        raise ValueError("Maximum amplitude spread ratio must be positive.")

    minimum_intervals = math.ceil(
        duration_seconds * min_heart_rate_bpm / 60.0
    )
    maximum_intervals = math.floor(
        duration_seconds * max_heart_rate_bpm / 60.0
    )
    minimum_detections = minimum_intervals + 1
    maximum_detections = maximum_intervals + 1
    removed_detection_count = raw_detection_count - cleaned_detection_count
    removed_fraction = (
        removed_detection_count / raw_detection_count
        if raw_detection_count
        else 0.0
    )

    reasons = []
    if raw_detection_count < minimum_detections:
        reasons.append(
            f"Too few raw R-wave detections: {raw_detection_count} < "
            f"{minimum_detections}."
        )
    elif cleaned_detection_count < minimum_detections:
        reasons.append(
            f"Too few R-wave detections after removefp: "
            f"{cleaned_detection_count} < {minimum_detections}."
        )

    if raw_detection_count > maximum_detections:
        reasons.append(
            f"Too many raw R-wave detections: {raw_detection_count} > "
            f"{maximum_detections}."
        )
    elif cleaned_detection_count > maximum_detections:
        reasons.append(
            f"Too many R-wave detections after removefp: "
            f"{cleaned_detection_count} > {maximum_detections}."
        )

    if removed_fraction > max_removed_fraction:
        reasons.append(
            f"Too many detections removed by removefp: "
            f"{removed_fraction:.1%} > {max_removed_fraction:.1%}."
        )

    if (
        not np.isnan(amplitude_spread_ratio)
        and amplitude_spread_ratio > max_amplitude_spread_ratio
    ):
        reasons.append(
            f"ECG block amplitude is too unstable: "
            f"{amplitude_spread_ratio:.2f} > "
            f"{max_amplitude_spread_ratio:.2f}."
        )

    return EcgSegmentQuality(
        is_valid=not reasons,
        raw_detection_count=raw_detection_count,
        cleaned_detection_count=cleaned_detection_count,
        removed_detection_count=removed_detection_count,
        removed_fraction=removed_fraction,
        amplitude_spread_ratio=amplitude_spread_ratio,
        minimum_detections=minimum_detections,
        maximum_detections=maximum_detections,
        maximum_removed_fraction=max_removed_fraction,
        maximum_amplitude_spread_ratio=max_amplitude_spread_ratio,
        reasons=tuple(reasons),
    )
