from unittest.mock import patch

import numpy as np
from biosigpy.hrv import removefp, tdmetrics
from biosigpy.tools import medfilt_threshold, snap_to_peak

from src.lib.ecg_peak_detection import pan_tompkins

from src.ecg_processing import (
    ECG_SEGMENT_FEATURE_LENGTH,
    ECG_SEGMENT_FEATURE_NAMES,
)
from src.lib.ecg_hrv_features import compute_time_domain_hrv


def _synthetic_ecg(duration_seconds=300, fs=200):
    time = np.arange(duration_seconds * fs, dtype=float) / fs
    ecg = 0.02 * np.sin(2 * np.pi * 0.2 * time)
    width = max(1, int(0.015 * fs))
    offsets = np.arange(-4 * width, 4 * width + 1)
    pulse = np.exp(-0.5 * (offsets / width) ** 2)
    beat_times = np.arange(1.0, duration_seconds - 1.0, 1.0)
    beat_times += 0.03 * np.sin(2 * np.pi * 0.1 * beat_times)
    for beat_time in beat_times:
        center = int(round(beat_time * fs))
        indices = center + offsets
        ecg[indices] += pulse
    return ecg


def test_ecg_feature_schema_excludes_removed_rr_percentage():
    assert "REMOVED_RR_PERCENTAGE" not in ECG_SEGMENT_FEATURE_NAMES
    assert "ECTOPIC" not in ECG_SEGMENT_FEATURE_NAMES
    assert "REMOVED_FP" not in ECG_SEGMENT_FEATURE_NAMES
    assert {"PIP", "PNNLS", "PNNSS", "AVNN"}.isdisjoint(
        ECG_SEGMENT_FEATURE_NAMES
    )
    assert {"MHR", "PNN50"}.issubset(
        ECG_SEGMENT_FEATURE_NAMES
    )
    assert "SDSD" not in ECG_SEGMENT_FEATURE_NAMES
    assert "HF" not in ECG_SEGMENT_FEATURE_NAMES
    assert "ECGage" in ECG_SEGMENT_FEATURE_NAMES
    assert {
        "LF",
        "HF_RESP",
        "LFN_RESP",
        "URLF",
        "RE",
        "R",
    }.issubset(ECG_SEGMENT_FEATURE_NAMES)
    assert ECG_SEGMENT_FEATURE_LENGTH == 11


def test_pan_tompkins_returns_refined_outputs():
    ecg = _synthetic_ecg(duration_seconds=20)
    with patch(
        "src.lib.ecg_peak_detection.snap_to_peak",
        wraps=snap_to_peak,
    ) as refine:
        locations = pan_tompkins(ecg, 200)

    refine.assert_called_once()
    snap_signal, approximate_locations, window_size = refine.call_args.args
    expected_locations = snap_to_peak(
        snap_signal,
        approximate_locations,
        window_size,
    ) - 1.0
    expected_locations = np.unique(np.sort(expected_locations.astype(int)))
    np.testing.assert_array_equal(locations, expected_locations)
    assert np.all(np.diff(locations) > 0)


def test_time_domain_flow_matches_biosigpy_without_fillgaps():
    raw_events = np.array([0.0, 1.0, 2.0, 2.2, 3.0, 4.0, 5.0])
    expected_events = removefp(raw_events)
    intervals_after_removefp = np.diff(expected_events)
    threshold = medfilt_threshold(
        intervals_after_removefp,
        50,
        1.5,
        1.5,
    )
    expected_outliers = intervals_after_removefp > threshold
    expected_intervals = intervals_after_removefp[~expected_outliers]
    expected_metrics = tdmetrics(expected_intervals)

    actual = compute_time_domain_hrv(raw_events, 200)

    np.testing.assert_array_equal(
        actual.cleaned_event_times,
        expected_events,
    )
    np.testing.assert_array_equal(
        actual.intervals_after_removefp,
        intervals_after_removefp,
    )
    np.testing.assert_array_equal(
        actual.interval_outlier_mask,
        expected_outliers,
    )
    np.testing.assert_array_equal(actual.intervals, expected_intervals)
    assert actual.removed_fp_count == 1
    assert actual.interval_outlier_count == 0
    assert actual.removed_rr_count == 1
    assert actual.removed_rr_percentage == 100.0 / 6.0
    assert actual.removed_detection_mask.tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert actual.metrics["MHR"] == expected_metrics["mhr"]
    assert actual.metrics["SDNN"] == expected_metrics["sdnn"]
    assert actual.metrics["RMSSD"] == expected_metrics["rmssd"]
    assert actual.metrics["PNN50"] == expected_metrics["pnn50"]


def test_time_domain_flow_excludes_median_threshold_outliers():
    intervals = np.ones(25)
    intervals[12] = 2.0
    event_times = np.concatenate(([0.0], np.cumsum(intervals)))
    threshold = medfilt_threshold(intervals, 50, 1.5, 1.5)
    expected_outliers = intervals > threshold
    expected_intervals = intervals[~expected_outliers]

    with patch(
        "src.lib.ecg_hrv_features.removefp",
        return_value=event_times,
    ):
        actual = compute_time_domain_hrv(event_times, 200)

    np.testing.assert_array_equal(
        actual.interval_outlier_mask,
        expected_outliers,
    )
    np.testing.assert_array_equal(actual.intervals, expected_intervals)
    assert actual.removed_fp_count == 0
    assert actual.interval_outlier_count == 1
    assert actual.removed_rr_count == 1
    assert actual.removed_rr_percentage == 4.0
    assert actual.metrics["SDNN"] == tdmetrics(expected_intervals)["sdnn"]
