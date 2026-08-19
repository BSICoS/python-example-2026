import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import numpy as np

from src.eeg_processing import (
    EEG_BACKGROUND_AGGREGATED_FEATURE_NAMES,
    EEG_SLOW_WAVE_FEATURE_NAMES,
)
from src.pipeline import features
from src.pipeline.features import get_feature_group_indices, get_feature_names
from src.pipeline.training import (
    COMPETITIVE_CAISR_FEATURE_NAMES,
    _get_combined_model_indices,
)
from src.common.caisr import get_sleep_architecture_features


class CompetitiveEegSlowWaveIndicesTests(unittest.TestCase):
    def test_caisr_sleep_architecture_features_follow_sleep_onset(self):
        annotation = {
            'available': True, 'fs': 1 / 30,
            'stage': np.array([5, 5, 2, 2, 9, 3, 4, 5], dtype=float),
            'respiratory': np.array([0, 1, 1, 0, 1, 0, 0, 0], dtype=float),
            'arousal': np.array([0, 1, 0, 1], dtype=float),
            'limb_movement': np.array([1, 0, 1], dtype=float),
        }
        values = get_sleep_architecture_features(annotation)
        expected = np.array([
            3 / 7, 1 / 7, 2 / 7, 0, 1 / 7, 4 / 7, 3 / (7 / 120), .5, 2.,
            60., 60., 60.,
        ], dtype=np.float32)
        np.testing.assert_allclose(values, expected)

    def test_legacy_cache_is_extended_without_physiological_extraction(self):
        legacy = np.arange(features.LEGACY_FEATURE_VECTOR_LENGTH, dtype=np.float32)
        cheap = np.arange(len(features.CHEAP_FEATURE_NAMES), dtype=np.float32) + 1000
        record = {'BidsFolder': 'sub-I0002000000001', 'SiteID': 'I0002', 'SessionID': 1}
        with tempfile.TemporaryDirectory() as directory:
            cache_file = Path(directory, 'legacy.sav')
            features.joblib.dump(legacy, cache_file, protocol=0)
            with patch.object(features, '_get_feature_cache_file', return_value=str(cache_file)), \
                 patch.object(features, '_compute_record_feature_vector', side_effect=AssertionError('extraction ran')), \
                 patch.object(features, '_extract_cheap_features', return_value=cheap):
                result, cache_hit = features.get_or_create_record_feature_vector(
                    record, directory, {'BMI': 25.0}, return_cache_hit=True)
        self.assertTrue(cache_hit)
        np.testing.assert_array_equal(result, np.hstack([legacy, cheap]))
        self.assertEqual(result.size, 213)

    def test_fresh_vector_has_the_current_schema_length(self):
        physiological = np.arange(features.LEGACY_FEATURE_VECTOR_LENGTH - 2, dtype=np.float32)
        cheap = np.arange(len(features.CHEAP_FEATURE_NAMES), dtype=np.float32)
        with patch.object(features.os.path, 'exists', return_value=True), \
             patch.object(features, '_load_required_signal_data', return_value=({}, {})), \
             patch.object(features, 'extract_extended_physiological_features', return_value=physiological), \
             patch.object(features, '_extract_cheap_features', return_value=cheap):
            vector = features._compute_record_feature_vector(
                {'Age': 70, 'Sex': 'Male'}, 'data', 'I0002', 'sub-I0002000000001', 1,
                'channel_table.csv', True)
        self.assertEqual(vector.size, 213)
    def test_production_features_keep_slow_wave_positions_as_nan(self):
        background = np.arange(len(EEG_BACKGROUND_AGGREGATED_FEATURE_NAMES), dtype=np.float32)
        resp = np.full(len(features.FEATURE_NAME_GROUPS['resp']), np.nan, dtype=np.float32)
        ecg = np.arange(len(features.FEATURE_NAME_GROUPS['ecg']), dtype=np.float32)
        with patch.object(features, '_extract_respiration_and_ecg_features', return_value=(
            resp, ecg,
        )), patch.object(features, '_extract_segmented_features', return_value=background):
            extracted = features.extract_extended_physiological_features({}, {})

        slow_wave_start = len(resp) + len(background)
        slow_wave_end = slow_wave_start + len(EEG_SLOW_WAVE_FEATURE_NAMES)
        self.assertEqual(extracted.size, features.LEGACY_FEATURE_VECTOR_LENGTH - 2)
        np.testing.assert_array_equal(extracted[slow_wave_start:slow_wave_end], np.full(28, np.nan))
        np.testing.assert_array_equal(extracted[:slow_wave_start], np.hstack([
            resp, background,
        ]))
        np.testing.assert_array_equal(extracted[slow_wave_end:], ecg)

    def test_competitive_eeg_routes_exclude_slow_wave_features(self):
        feature_names = get_feature_names()
        feature_indices = get_feature_group_indices(include_demographics=True)
        combined_indices = _get_combined_model_indices(feature_indices)

        eeg_route_names = [feature_names[index] for index in combined_indices['eeg']]
        ecg_eeg_names = [feature_names[index] for index in combined_indices['ecg_eeg']]
        self.assertFalse(any('NREM_SW_' in name for name in eeg_route_names))
        self.assertFalse(any('NREM_SW_' in name for name in ecg_eeg_names))
        self.assertTrue(set(COMPETITIVE_CAISR_FEATURE_NAMES).issubset(eeg_route_names))
        self.assertTrue(set(COMPETITIVE_CAISR_FEATURE_NAMES).issubset(ecg_eeg_names))
        excluded_cheap_features = set(features.CHEAP_FEATURE_NAMES) - set(COMPETITIVE_CAISR_FEATURE_NAMES)
        self.assertFalse(set(eeg_route_names) & excluded_cheap_features)
        self.assertFalse(set(ecg_eeg_names) & excluded_cheap_features)
        self.assertEqual(len(combined_indices['ecg_eeg']), 142)

    def test_cheap_features_do_not_affect_modality_presence(self):
        feature_indices = get_feature_group_indices(include_demographics=True)
        combined_indices = _get_combined_model_indices(feature_indices)
        competitive_indices = [
            get_feature_names().index(name) for name in COMPETITIVE_CAISR_FEATURE_NAMES
        ]
        self.assertFalse(any(index in feature_indices['eeg'] for index in competitive_indices))
        self.assertFalse(any(index in feature_indices['ecg'] for index in competitive_indices))
        self.assertTrue(all(index in combined_indices['ecg_eeg'] for index in competitive_indices))


if __name__ == '__main__':
    unittest.main()