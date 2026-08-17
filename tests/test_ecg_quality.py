from unittest.mock import patch

import numpy as np

from src.lib.ecg_quality import evaluate_ecg_segment_quality
from src.pipeline.config import DEFAULT_CSV_PATH
from src.resp_processing import (
    processResp,
    select_best_respiration_signal,
)


def test_ecg_quality_accepts_plausible_clean_window():
    quality = evaluate_ecg_segment_quality(
        detection_count=301,
        altered_count=20,
        duration_seconds=300,
    )

    assert quality.is_valid
    assert quality.minimum_detections == 151
    assert quality.maximum_detections == 1101
    assert np.isclose(quality.altered_fraction, 20 / 300)


def test_ecg_quality_rejects_too_few_and_too_many_detections():
    too_few = evaluate_ecg_segment_quality(150, 0, 300)
    too_many = evaluate_ecg_segment_quality(1102, 0, 300)

    assert not too_few.is_valid
    assert "Too few" in too_few.reasons[0]
    assert not too_many.is_valid
    assert "Too many R-wave" in too_many.reasons[0]


def test_ecg_quality_rejects_excessive_interval_alteration():
    quality = evaluate_ecg_segment_quality(
        detection_count=558,
        altered_count=450,
        duration_seconds=300,
    )

    assert not quality.is_valid
    assert np.isclose(quality.altered_fraction, 450 / 557)
    assert "altered R-R intervals" in quality.reasons[0]


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
