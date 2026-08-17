import numpy as np
from typing import cast

from scipy.signal import butter, filtfilt, resample

from .ecg_age import compute_ecgage
from .ecg_peak_detection import pan_tompkins
from .ecg_hrv_features import compute_legacy_hf, compute_time_domain_hrv
from .ecg_quality import (
    compute_ecg_amplitude_spread_ratio,
    evaluate_ecg_segment_quality,
)


def compute_ecg_features(ecg_signal, fs, ecg_feature_length):
    fs = int(round(float(fs)))
    if fs <= 0:
        return None

    signal_duration_seconds = len(ecg_signal) / fs
    ecg_signal = ecg_signal - np.mean(ecg_signal)

    target_fs = 200
    length_ecg = len(ecg_signal)

    if fs != target_fs:
        num_samples = int(length_ecg * target_fs / fs)
        ecg_signal = resample(ecg_signal, num_samples)
        fs = target_fs

    length_ecg = len(ecg_signal)
    if (
        np.sum(np.isnan(ecg_signal)) != 0
        or np.sum(ecg_signal == 0) > 0.2 * length_ecg
    ):
        return np.full(ecg_feature_length, np.nan, dtype=np.float32)

    b, a = cast(
        tuple[np.ndarray, np.ndarray],
        butter(
            3,
            [59.5 / (fs / 2), 60.5 / (fs / 2)],
            btype="bandstop",
            output="ba",
        ),
    )
    ecg_signal = filtfilt(b, a, ecg_signal)

    b, a = cast(
        tuple[np.ndarray, np.ndarray],
        butter(3, 0.5 / (fs / 2), btype="high", output="ba"),
    )
    ecg_signal = filtfilt(b, a, ecg_signal)

    b, a = cast(
        tuple[np.ndarray, np.ndarray],
        butter(3, 50 / (fs / 2), btype="low", output="ba"),
    )
    ecg_signal = filtfilt(b, a, ecg_signal)

    _, r_locations, _ = pan_tompkins(ecg_signal, fs, 0)
    r_wave_times = np.asarray(r_locations, dtype=float) / fs
    time_domain = compute_time_domain_hrv(
        r_wave_times,
        fs,
    )
    amplitude_spread_ratio = compute_ecg_amplitude_spread_ratio(
        ecg_signal,
        fs,
    )
    quality = evaluate_ecg_segment_quality(
        raw_detection_count=len(time_domain.raw_event_times),
        cleaned_detection_count=len(time_domain.cleaned_event_times),
        duration_seconds=signal_duration_seconds,
        amplitude_spread_ratio=amplitude_spread_ratio,
    )
    if not quality.is_valid:
        return np.full(ecg_feature_length, np.nan, dtype=np.float32)

    metrics = time_domain.metrics
    hf = compute_legacy_hf(time_domain.intervals)
    ecg_age = compute_ecgage(ecg_signal)

    return np.array(
        [
            metrics["PIP"],
            metrics["PNNLS"],
            metrics["PNNSS"],
            metrics["AVNN"],
            metrics["SDNN"],
            metrics["RMSSD"],
            hf,
            time_domain.removed_percentage,
            ecg_age,
        ],
        dtype=np.float32,
    )
