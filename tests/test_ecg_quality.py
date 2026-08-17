from unittest.mock import patch

import numpy as np

from src.lib.ecg_quality import (
    compute_ecg_amplitude_spread_ratio,
    evaluate_ecg_segment_quality,
)
from src.pipeline.config import DEFAULT_CSV_PATH
from src.resp_processing import (
    processResp,
    select_best_respiration_signal,
)


def test_ecg_quality_accepts_plausible_clean_window():
    quality = evaluate_ecg_segment_quality(
        raw_detection_count=321,
        cleaned_detection_count=301,
        duration_seconds=300,
    )

    assert quality.is_valid
    assert quality.minimum_detections == 151
    assert quality.maximum_detections == 1101
    assert quality.removed_detection_count == 20
    assert np.isclose(quality.removed_fraction, 20 / 321)


def test_ecg_quality_rejects_too_few_and_too_many_detections():
    too_few = evaluate_ecg_segment_quality(150, 150, 300)
    too_many = evaluate_ecg_segment_quality(1102, 1102, 300)

    assert not too_few.is_valid
    assert "Too few" in too_few.reasons[0]
    assert not too_many.is_valid
    assert "Too many raw R-wave" in too_many.reasons[0]


def test_ecg_quality_rejects_excessive_removefp_removal():
    quality = evaluate_ecg_segment_quality(
        raw_detection_count=400,
        cleaned_detection_count=299,
        duration_seconds=300,
    )

    assert not quality.is_valid
    assert np.isclose(quality.removed_fraction, 101 / 400)
    assert "removed by removefp" in quality.reasons[0]

def test_ecg_quality_rejects_large_block_amplitude_instability():
    fs = 100
    time = np.arange(300 * fs) / fs
    ecg = np.sin(2 * np.pi * time)
    ecg[200 * fs :] *= 20
    amplitude_spread_ratio = compute_ecg_amplitude_spread_ratio(ecg, fs)

    quality = evaluate_ecg_segment_quality(
        raw_detection_count=301,
        cleaned_detection_count=295,
        duration_seconds=300,
        amplitude_spread_ratio=amplitude_spread_ratio,
    )

    assert amplitude_spread_ratio > 10
    assert not quality.is_valid
    assert "amplitude is too unstable" in quality.reasons[0]

def test_cflow_is_not_an_eligible_direct_respiration_signal():
    selection = select_best_respiration_signal(
        {"C-FLOW": np.ones(3000)},
        {"C-FLOW": 10},
        DEFAULT_CSV_PATH,
    )

    assert selection is None


def test_recognized_respiration_uses_feature_pipeline_quality():
    peakedness_output = (
        np.array([1.0, 2.0]),
        None,
        None,
        np.array([0.7, 0.9]),
    )
    with patch(
        "src.resp_processing.resp_features.peakedness_application",
        return_value=peakedness_output,
    ):
        data = {"CHEST": np.sin(np.linspace(0, 20, 3000))}
        sampling_frequencies = {"CHEST": 10}
        selection = select_best_respiration_signal(
            data,
            sampling_frequencies,
            DEFAULT_CSV_PATH,
        )
        features = processResp(
            data,
            sampling_frequencies,
            DEFAULT_CSV_PATH,
        )

    assert selection is not None
    assert selection.label == "CHEST"
    assert selection.group == "Chest"
    assert np.isclose(selection.quality, 0.8)
    assert np.isclose(selection.peakedness, 1.5)
    assert np.isclose(features[1], 1.5)
