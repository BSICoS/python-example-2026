"""Diagnostic trace of the current ECG feature pipeline.

This module intentionally mirrors the production path without changing the
feature extractor. Equivalence is covered by tests while the migration viewer
is in use; the module can be removed once the Biosigpy pipeline replaces it.
"""

from dataclasses import dataclass, field
from typing import cast

import numpy as np
from scipy.signal import butter, filtfilt, resample

from .ecg_age import compute_ecgage
from .ecg_hrv_features import compute_hrv_hrf
from .ecg_nn_interpolation import interpolate_nn_pchip
from .ecg_peak_detection import PanTompkinsTrace, pan_tompkins
from .ecg_quality import EcgSegmentQuality, evaluate_ecg_segment_quality
from .ecg_rr_cleaning import remove_ectopic_beats_with_mask


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
    minimum_intervals_per_window: int = 0
    centered_signal: np.ndarray = field(default_factory=_empty_float_array)
    resampled_signal: np.ndarray = field(default_factory=_empty_float_array)
    notch_filtered_signal: np.ndarray = field(default_factory=_empty_float_array)
    highpass_filtered_signal: np.ndarray = field(default_factory=_empty_float_array)
    lowpass_filtered_signal: np.ndarray = field(default_factory=_empty_float_array)
    detector: PanTompkinsTrace | None = None
    raw_intervals: np.ndarray = field(default_factory=_empty_float_array)
    corrected_intervals: np.ndarray = field(default_factory=_empty_float_array)
    interpolated_intervals: np.ndarray = field(default_factory=_empty_float_array)
    ectopic_mask: np.ndarray = field(default_factory=_empty_bool_array)
    ectopic_percentage: float = np.nan
    valid_ratio: float = np.nan
    quality: EcgSegmentQuality | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    features: np.ndarray | None = None
    failure_reason: str | None = None


def inspect_current_ecg_features(
    ecg_signal,
    fs,
    ecg_feature_length,
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
    window_length_seconds = max(1, int(trace.signal_duration_seconds))
    trace.minimum_intervals_per_window = max(
        1,
        int(np.ceil(window_length_seconds / 2)),
    )

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
        trace.failure_reason = f"Pan-Tompkins failed: {error}"
        return trace

    r_locations = trace.detector.r_locations
    trace.raw_intervals = np.diff(r_locations) / fs
    (
        trace.corrected_intervals,
        trace.ectopic_percentage,
        trace.ectopic_mask,
    ) = remove_ectopic_beats_with_mask(
        trace.raw_intervals,
        40,
        0.10,
    )
    trace.quality = evaluate_ecg_segment_quality(
        detection_count=len(r_locations),
        altered_count=np.count_nonzero(trace.ectopic_mask),
        duration_seconds=trace.signal_duration_seconds,
    )
    if not trace.quality.is_valid:
        return reject(
            "ECG quality rejected: " + "; ".join(trace.quality.reasons)
        )

    trace.interpolated_intervals = interpolate_nn_pchip(
        trace.corrected_intervals,
        2,
    )

    if len(trace.interpolated_intervals) == 0:
        return reject("No intervals remain after legacy cleaning.")

    trace.valid_ratio = float(
        np.sum(~np.isnan(trace.interpolated_intervals))
        / len(trace.interpolated_intervals)
    )
    valid_intervals = trace.interpolated_intervals[
        ~np.isnan(trace.interpolated_intervals)
    ]
    if (
        trace.valid_ratio < 0.75
        or len(valid_intervals) < trace.minimum_intervals_per_window
    ):
        return reject("Fewer than 75% of intervals remain valid.")

    trace.metrics = compute_hrv_hrf(valid_intervals, fs)
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
            trace.metrics["SDNN"],
            trace.metrics["RMSSD"],
            trace.metrics["HF"],
            trace.ectopic_percentage,
            ecg_age,
        ],
        dtype=np.float32,
    )
    return trace
