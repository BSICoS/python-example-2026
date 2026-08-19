import unittest

from src.pipeline.features import get_feature_group_indices, get_feature_names
from src.pipeline.training import (
    COMPETITIVE_EEG_SLOW_WAVE_FEATURE_NAMES,
    _get_combined_model_indices,
)


class CompetitiveEegSlowWaveIndicesTests(unittest.TestCase):
    def test_competitive_eeg_routes_retain_only_requested_slow_wave_features(self):
        feature_names = get_feature_names()
        feature_indices = get_feature_group_indices(include_demographics=True)
        combined_indices = _get_combined_model_indices(feature_indices)

        eeg_route_names = [feature_names[index] for index in combined_indices['eeg']]
        ecg_eeg_names = [feature_names[index] for index in combined_indices['ecg_eeg']]
        retained_slow_wave_names = {
            name for name in eeg_route_names if 'NREM_SW_' in name
        }

        self.assertEqual(retained_slow_wave_names, COMPETITIVE_EEG_SLOW_WAVE_FEATURE_NAMES)
        self.assertEqual(
            {name for name in ecg_eeg_names if 'NREM_SW_' in name},
            COMPETITIVE_EEG_SLOW_WAVE_FEATURE_NAMES,
        )
        self.assertEqual(len(combined_indices['ecg_eeg']), 148)


if __name__ == '__main__':
    unittest.main()