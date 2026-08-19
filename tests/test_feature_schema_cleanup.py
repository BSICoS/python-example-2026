import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from src.pipeline import features
from src.pipeline.config import TOTAL_PHYSIOLOGICAL_FEATURE_LENGTH
from src.pipeline.training import _get_combined_model_indices
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


if __name__ == '__main__':
    unittest.main()
