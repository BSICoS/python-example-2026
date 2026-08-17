from unittest.mock import patch

import numpy as np

from src.lib import ecg_features, ecg_inspection
from src.lib.ecg_peak_detection import pan_tompkins
from src.lib.ecg_rr_cleaning import (
    remove_ectopic_beats,
    remove_ectopic_beats_with_mask,
)


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


def test_pan_tompkins_trace_preserves_legacy_outputs():
    ecg = _synthetic_ecg(duration_seconds=20)
    amplitudes, locations, delay = pan_tompkins(ecg, 200)
    trace = pan_tompkins(ecg, 200, return_trace=True)

    np.testing.assert_array_equal(trace.r_amplitudes, amplitudes)
    np.testing.assert_array_equal(trace.r_locations, locations)
    assert trace.delay == delay
    assert trace.ecg_bandpassed.shape == ecg.shape
    assert trace.derivative.shape == ecg.shape
    assert trace.envelope.shape == ecg.shape


def test_ectopic_mask_preserves_legacy_cleaning_outputs():
    intervals = np.ones(80)
    intervals[40] = 1.5

    expected_intervals, expected_percentage = remove_ectopic_beats(
        intervals,
        40,
        0.10,
    )
    actual_intervals, actual_percentage, mask = (
        remove_ectopic_beats_with_mask(intervals, 40, 0.10)
    )

    np.testing.assert_array_equal(actual_intervals, expected_intervals)
    assert actual_percentage == expected_percentage
    assert np.flatnonzero(mask).tolist() == [40]


def test_inspection_trace_matches_current_feature_vector():
    ecg = _synthetic_ecg()
    with (
        patch.object(
            ecg_features,
            "compute_ecgage",
            return_value=42.0,
        ),
        patch.object(
            ecg_inspection,
            "compute_ecgage",
            return_value=42.0,
        ),
    ):
        expected = ecg_features.compute_ecg_features(ecg, 200, 9)
        trace = ecg_inspection.inspect_current_ecg_features(ecg, 200, 9)

    assert trace.failure_reason is None
    assert trace.features is not None
    np.testing.assert_allclose(trace.features, expected, rtol=0, atol=0)
    assert trace.detector is not None
    assert trace.raw_intervals.size == trace.corrected_intervals.size
    assert trace.ectopic_mask.size == trace.raw_intervals.size
