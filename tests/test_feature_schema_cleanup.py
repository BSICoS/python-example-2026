import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from src.pipeline import features
from src.pipeline.config import TOTAL_PHYSIOLOGICAL_FEATURE_LENGTH
from src.pipeline.training import (
    _fit_ensemble, _get_combined_model_indices, _get_ecg_eeg_search_data,
    _get_eeg_search_data,
)
from src.resp_processing import SelectedRespiration


class FeatureSchemaCleanupTests(unittest.TestCase):
    def test_final_schema_and_model_route_sizes(self):
        names = features.get_feature_names()
        groups = features.FEATURE_NAME_GROUPS
        self.assertEqual(len(names), 140)
        self.assertEqual(len(groups['demographics']), 2)
        self.assertEqual(len(groups['eeg']), 96)
        self.assertEqual(len(groups['ecg']), 42)
        self.assertEqual(TOTAL_PHYSIOLOGICAL_FEATURE_LENGTH, 138)
        self.assertEqual(names[:2], ('Age', 'Sex'))
        self.assertNotIn('resp', groups)
        self.assertFalse(any(token in name for token in ('NREM_SW', 'CAISR', 'BMI') for name in names))
        routes = _get_combined_model_indices(features.get_feature_group_indices(True))
        self.assertEqual(tuple(routes), ('ecg', 'eeg', 'ecg_eeg'))
        self.assertEqual({name: len(indices) for name, indices in routes.items()},
                         {'ecg': 44, 'eeg': 98, 'ecg_eeg': 140})

    def test_fresh_extraction_has_final_length_and_keeps_ecg_respiration_auxiliary(self):
        selected = SelectedRespiration('CHEST', 'Chest', np.ones(3000), 10.0,
                                      np.arange(7500), 25.0, .9, 1.2)
        captured = {}
        def extract_ecg(data, frequencies, csv_path, selected_respiration):
            captured['selected'] = selected_respiration
            return np.zeros(11, dtype=np.float32)
        with patch.object(features, 'select_best_respiration_signal', return_value=selected), \
             patch.object(features, 'processECG', side_effect=extract_ecg), \
             patch.object(features, 'processEEG', return_value=np.zeros(24, dtype=np.float32)):
            vector = features.extract_extended_physiological_features(
                {'ECG': np.ones(60000), 'CHEST': np.ones(3000)}, {'ECG': 200, 'CHEST': 10},
                'channel_table.csv')
        self.assertEqual(vector.size, 138)
        self.assertIs(captured['selected'], selected)

    def test_legacy_cache_lengths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'legacy.sav')
            joblib.dump(np.zeros(200, dtype=np.float32), path, protocol=0)
            self.assertIsNone(features._load_cached_feature_vector(path))
            joblib.dump(np.zeros(140, dtype=np.float32), path, protocol=0)
            self.assertEqual(features._load_cached_feature_vector(path).size, 140)

    def test_route_search_data_and_route_specific_parameters(self):
        feature_indices = features.get_feature_group_indices(True)
        modality_indices = features.get_feature_group_indices(False)
        matrix = np.ones((3, 140), dtype=np.float32)
        matrix[2, modality_indices['eeg']] = np.nan
        labels = np.array([0, 1, 0], dtype=np.int32)
        groups = np.array(['I0002', 'I0006', 'S0001'])
        common = _get_ecg_eeg_search_data(
            matrix, labels, feature_indices, modality_indices,
            categorical_indices=[1], site_groups=groups)
        eeg = _get_eeg_search_data(
            matrix, labels, feature_indices, modality_indices,
            categorical_indices=[1], site_groups=groups)
        self.assertEqual(common['features'].shape[1], 140)
        self.assertEqual(eeg['features'].shape[1], 98)
        self.assertEqual(eeg['features'].shape[0], 2)
        self.assertEqual(eeg['age_feature_index'], 0)
        self.assertEqual(eeg['categorical_indices'], [1])
        self.assertEqual(eeg['site_groups'].tolist(), ['I0002', 'I0006'])

        captured = {}
        def fake_route(features_, labels_, indices_, categorical_, params_):
            captured[tuple(indices_.tolist())] = dict(params_)
            return {'model': object(), 'raw_indices': indices_}
        with patch('src.pipeline.training._fit_route_model', side_effect=fake_route):
            _fit_ensemble(
                matrix, labels, feature_indices,
                final_params_by_route={
                    'ecg_eeg': {'max_depth': 3},
                    'eeg': {'max_depth': 5},
                    'ecg': {'max_depth': 7},
                },
                modality_presence_indices=modality_indices,
                categorical_indices=[1],
            )
        expected = _get_combined_model_indices(feature_indices)
        self.assertEqual(captured[tuple(expected['ecg_eeg'].tolist())], {'max_depth': 3})
        self.assertEqual(captured[tuple(expected['eeg'].tolist())], {'max_depth': 5})
        self.assertEqual(captured[tuple(expected['ecg'].tolist())], {'max_depth': 7})


if __name__ == '__main__':
    unittest.main()
