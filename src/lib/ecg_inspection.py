"""Diagnostic trace of the current ECG feature pipeline.

This module intentionally mirrors the production path without changing the
feature extractor. Equivalence is covered by tests while the migration viewer
is in use; the module can be removed with the visualization layer.
"""

from dataclasses import dataclass, field
from typing import cast

import numpy as np
from scipy.signal import butter, filtfilt, resample

from .ecg_age import compute_ecgage
from .ecg_features import FD_METRIC_NAMES
from .ecg_frequency_features import (
    FrequencyDomainHrvResult,
    compute_frequency_domain_hrv,
)

from .ecg_hrv_features import compute_time_domain_hrv
from .ecg_peak_detection import PanTompkinsTrace, pan_tompkins
from .ecg_quality import (
    EcgSegmentQuality,
    compute_ecg_amplitude_spread_ratio,
    evaluate_ecg_segment_quality,
)


def _empty_float_array():
    return np.array([], dtype=float)


def _empty_bool_array():
    return np.array([], dtype=bool)


@dataclass
class CurrentEcgTrace:
    """Observable state of the legacy ECG feature pipeline for one window."""

    raw_signal: np.ndarray
    original_fs: int
    processed_fs: int | None = None
    signal_duration_seconds: float = 0.0

    centered_signal: np.ndarray = field(default_factory=_empty_float_array)
    resampled_signal: np.ndarray = field(default_factory=_empty_float_array)
    notch_filtered_signal: np.ndarray = field(default_factory=_empty_float_array)
    highpass_filtered_signal: np.ndarray = field(default_factory=_empty_float_array)
    lowpass_filtered_signal: np.ndarray = field(default_factory=_empty_float_array)
    detector: PanTompkinsTrace | None = None
    cleaned_event_times: np.ndarray = field(default_factory=_empty_float_array)
    removed_detection_mask: np.ndarray = field(default_factory=_empty_bool_array)
    raw_intervals: np.ndarray = field(default_factory=_empty_float_array)
    intervals_after_removefp: np.ndarray = field(default_factory=_empty_float_array)
    interval_outlier_mask: np.ndarray = field(default_factory=_empty_bool_array)
    cleaned_intervals: np.ndarray = field(default_factory=_empty_float_array)
    removed_rr_percentage: float = np.nan
    quality: EcgSegmentQuality | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    frequency_domain: FrequencyDomainHrvResult | None = None
    frequency_failure_reason: str | None = None
    features: np.ndarray | None = None
    failure_reason: str | None = None


def inspect_current_ecg_features(
    ecg_signal,
    fs,
    ecg_feature_length,
    *,
    respiration_signal=None,
    respiration_sampling_frequency=None,
) -> CurrentEcgTrace:
    """Run the current algorithm while retaining each intermediate array."""

    raw_signal = np.asarray(ecg_signal, dtype=float).flatten()
    fs = int(round(float(fs)))
    trace = CurrentEcgTrace(raw_signal=raw_signal.copy(), original_fs=fs)

    def reject(reason):
        trace.features = np.full(
            ecg_feature_length,
            np.nan,
            dtype=np.float32,
        )
        trace.failure_reason = reason
        return trace

    if fs <= 0:
        trace.failure_reason = "Sampling frequency must be positive."
        return trace

    trace.signal_duration_seconds = len(raw_signal) / fs
    centered_signal = raw_signal - np.mean(raw_signal)
    trace.centered_signal = centered_signal.copy()

    target_fs = 200
    if fs != target_fs:
        num_samples = int(len(centered_signal) * target_fs / fs)
        processed_signal = resample(centered_signal, num_samples)
        fs = target_fs
    else:
        processed_signal = centered_signal

    trace.processed_fs = fs
    trace.resampled_signal = np.asarray(processed_signal, dtype=float).copy()

    if (
        np.sum(np.isnan(processed_signal)) != 0
        or np.sum(processed_signal == 0) > 0.2 * len(processed_signal)
    ):
        return reject("The ECG contains NaNs or more than 20% zero samples.")

    b, a = cast(
        tuple[np.ndarray, np.ndarray],
        butter(
            3,
            [59.5 / (fs / 2), 60.5 / (fs / 2)],
            btype="bandstop",
            output="ba",
        ),
    )
    trace.notch_filtered_signal = filtfilt(b, a, processed_signal)

    b, a = cast(
        tuple[np.ndarray, np.ndarray],
        butter(3, 0.5 / (fs / 2), btype="high", output="ba"),
    )
    trace.highpass_filtered_signal = filtfilt(
        b,
        a,
        trace.notch_filtered_signal,
    )

    b, a = cast(
        tuple[np.ndarray, np.ndarray],
        butter(3, 50 / (fs / 2), btype="low", output="ba"),
    )
    trace.lowpass_filtered_signal = filtfilt(
        b,
        a,
        trace.highpass_filtered_signal,
    )

    try:
        trace.detector = pan_tompkins(
            trace.lowpass_filtered_signal,
            fs,
            0,
            return_trace=True,
        )
    except Exception as error:
        return reject(f"Pan-Tompkins failed: {error}")

    time_domain = compute_time_domain_hrv(
        np.asarray(trace.detector.r_locations, dtype=float) / fs,
        fs,
    )
    trace.cleaned_event_times = time_domain.cleaned_event_times
    trace.removed_detection_mask = time_domain.removed_detection_mask
    trace.raw_intervals = np.diff(time_domain.raw_event_times)
    trace.intervals_after_removefp = time_domain.intervals_after_removefp
    trace.interval_outlier_mask = time_domain.interval_outlier_mask
    trace.cleaned_intervals = time_domain.intervals
    trace.removed_rr_percentage = time_domain.removed_rr_percentage
    amplitude_spread_ratio = compute_ecg_amplitude_spread_ratio(
        trace.lowpass_filtered_signal,
        fs,
    )
    trace.quality = evaluate_ecg_segment_quality(
        raw_detection_count=len(time_domain.raw_event_times),
        cleaned_detection_count=len(time_domain.cleaned_event_times),
        duration_seconds=trace.signal_duration_seconds,
        amplitude_spread_ratio=amplitude_spread_ratio,
    )
    if not trace.quality.is_valid:
        return reject(
            "ECG quality rejected: " + "; ".join(trace.quality.reasons)
        )

    trace.metrics = time_domain.metrics.copy()
    trace.metrics.update({name: np.nan for name in FD_METRIC_NAMES})
    try:
        trace.frequency_domain = compute_frequency_domain_hrv(
            time_domain.cleaned_event_times,
            trace.lowpass_filtered_signal,
            fs,
            respiration_signal=respiration_signal,
            respiration_sampling_frequency=respiration_sampling_frequency,
        )
        trace.metrics.update(trace.frequency_domain.metrics)
    except (TypeError, ValueError, np.linalg.LinAlgError) as error:
        trace.frequency_failure_reason = str(error)
    try:
        ecg_age = compute_ecgage(trace.lowpass_filtered_signal)
    except Exception as error:
        ecg_age = np.nan
        trace.failure_reason = f"ECGage failed: {error}"

    trace.features = np.array(
        [
            trace.metrics["PIP"],
            trace.metrics["PNNLS"],
            trace.metrics["PNNSS"],
            trace.metrics["AVNN"],
            trace.metrics["MHR"],
            trace.metrics["SDNN"],
            trace.metrics["RMSSD"],
            trace.metrics["PNN50"],
            trace.metrics["LF"],
            trace.metrics["HF_RESP"],
            trace.metrics["LFN_RESP"],
            trace.metrics["LFHF_RESP"],
            trace.metrics["URLF"],
            trace.metrics["RE"],
            trace.metrics["R"],
            trace.removed_rr_percentage,
            ecg_age,
        ],
        dtype=np.float32,
    )
    return trace
