from unittest.mock import patch

import numpy as np
from biosigpy.hrv import removefp, tdmetrics
from biosigpy.tools import snap_to_peak

from src.lib import ecg_features, ecg_inspection
from src.lib.ecg_peak_detection import pan_tompkins

from src.ecg_processing import ECG_SEGMENT_FEATURE_NAMES
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


def test_removed_fp_feature_uses_its_actual_semantics():
    assert ECG_SEGMENT_FEATURE_NAMES[7] == "REMOVED_FP"
    assert "ECTOPIC" not in ECG_SEGMENT_FEATURE_NAMES


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
    expected_locations = snap_to_peak(
        trace.ecg_centered,
        trace.unrefined_r_locations.astype(float) + 1.0,
        20.0,
    ) - 1.0
    expected_locations = np.unique(np.sort(expected_locations.astype(int)))
    np.testing.assert_array_equal(trace.r_locations, expected_locations)
    assert np.all(np.diff(trace.r_locations) > 0)
    np.testing.assert_array_equal(
        trace.r_amplitudes,
        trace.ecg_centered[trace.r_locations],
    )


def test_time_domain_flow_matches_biosigpy_without_fillgaps():
    raw_events = np.array([0.0, 1.0, 2.0, 2.2, 3.0, 4.0, 5.0])
    expected_events = removefp(raw_events)
    expected_metrics = tdmetrics(np.diff(expected_events))

    actual = compute_time_domain_hrv(raw_events, 200)

    np.testing.assert_array_equal(
        actual.cleaned_event_times,
        expected_events,
    )
    np.testing.assert_array_equal(
        actual.intervals,
        np.diff(expected_events),
    )
    assert actual.removed_count == 1
    assert actual.removed_detection_mask.tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert actual.metrics["SDNN"] == expected_metrics["sdnn"]
    assert actual.metrics["RMSSD"] == expected_metrics["rmssd"]


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
    assert (
        trace.removed_detection_mask.size
        == trace.detector.r_locations.size
    )
    assert (
        trace.cleaned_intervals.size
        == max(0, trace.cleaned_event_times.size - 1)
    )
