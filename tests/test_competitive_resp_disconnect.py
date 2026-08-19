import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from src.pipeline import features
from src.pipeline.training import _get_combined_model_indices, _select_ensemble_model_name
from src.resp_processing import SelectedRespiration


class CompetitiveRespDisconnectTests(unittest.TestCase):
    def test_new_resp_block_is_nan_and_ecg_receives_selected_signal(self):
        selected = SelectedRespiration('CHEST', 'Chest', np.ones(3000), 10.0,
                                      np.arange(7500), 25.0, .9, 1.2)
        captured = {}
        def process_ecg(data, frequencies, csv_path, selected_respiration):
            captured['selected'] = selected_respiration
            return np.zeros(10, dtype=np.float32)
        data, frequencies = {'ECG': np.ones(60000), 'CHEST': np.ones(3000)}, {'ECG': 200, 'CHEST': 10}
        with patch.object(features, 'select_best_respiration_signal', return_value=selected), \
             patch.object(features, 'processECG', side_effect=process_ecg):
            resp, _ = features._extract_respiration_and_ecg_features(data, frequencies, 'channel_table.csv')
        self.assertTrue(np.isnan(resp).all())
        self.assertIs(captured['selected'], selected)

    def test_model_routes_exclude_resp_and_ignore_finite_cached_resp_columns(self):
        indices = {'demographics': np.array([0]), 'resp': np.array([1]),
                   'eeg': np.array([2]), 'ecg': np.array([3])}
        self.assertEqual(tuple(_get_combined_model_indices(indices)), ('ecg', 'eeg', 'ecg_eeg'))
        models = {name: object() for name in ('ecg', 'eeg', 'ecg_eeg')}
        self.assertEqual(_select_ensemble_model_name(
            np.array([1., 99., np.nan, np.nan]), models, indices), 'ecg_eeg')

        cached = np.zeros(len(features.FEATURE_NAMES), dtype=np.float32)
        cached[2] = 42.0  # A finite historic RESP column remains schema-compatible.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'historic.sav')
            joblib.dump(cached, path, protocol=0)
            loaded = features._load_cached_feature_vector(path)
        np.testing.assert_array_equal(loaded, cached)


if __name__ == '__main__':
    unittest.main()
