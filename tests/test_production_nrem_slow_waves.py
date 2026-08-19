import unittest
from unittest.mock import patch

import numpy as np

from src import eeg_processing
from src.common.caisr import unavailable_annotation


class ProductionNremSlowWaveTests(unittest.TestCase):
    @staticmethod
    def _annotation(probability=.7):
        return {'available': True, 'fs': 1 / 30,
                'stage': np.full(10, 2.0),
                'p_n2': np.full(10, probability - .3),
                'p_n3': np.full(10, .3)}

    @staticmethod
    def _signals():
        fs = 200
        signal = np.sin(2 * np.pi * np.arange(fs * 300) / fs)
        return {'c3-m2': signal}, {'c3-m2': fs}

    def test_unavailable_caisr_keeps_background_and_returns_nan_sw(self):
        data, frequencies = self._signals()
        background = eeg_processing.processEEG(data, frequencies, 'channel_table.csv')
        slow_waves = eeg_processing.extract_record_slow_wave_features(
            data, frequencies, 'channel_table.csv', unavailable_annotation())
        self.assertTrue(np.isfinite(background).any())
        self.assertTrue(np.isnan(slow_waves).all())

    def test_weighted_density_and_morphology_are_record_level(self):
        data, frequencies = self._signals()
        event = {'Ref_PeakInd': 100, 'Ref_P2PAmp': 10., 'Ref_NegSlope': -2.,
                 'Ref_DownInd': 50, 'Ref_UpInd': 150}
        captured = {}
        def detect(signal, fs, **kwargs):
            captured.update(kwargs)
            return {'events': [event]}
        with patch.object(eeg_processing.eeg_features, 'detect_slow_waves', side_effect=detect):
            values = eeg_processing.extract_record_slow_wave_features(
                data, frequencies, 'channel_table.csv', self._annotation())
        self.assertEqual(values.size, 28)
        self.assertEqual(captured['allowed_stages'], (1, 2))
        self.assertAlmostEqual(values[0], .2)  # .7 weighted event / (10 * .5 * .7) minutes
        np.testing.assert_allclose(values[1:7], [10., 0., -2., 0., .5, 0.], equal_nan=True)
        self.assertTrue(np.isnan(values[7:]).all())

    def test_weighted_quantiles_follow_event_probabilities(self):
        values = np.array([1.0, 10.0, 100.0])
        weights = np.array([.2, .3, .5])
        self.assertEqual(eeg_processing._weighted_quantile(values, weights, .5), 10.0)
        self.assertEqual(eeg_processing._weighted_quantile(values, weights, .75), 100.0)

    def test_final_eeg_schema_is_124_and_excludes_old_slow_wave_features(self):
        self.assertEqual(eeg_processing.EEG_FEATURE_LENGTH, 124)
        self.assertEqual(len(eeg_processing.EEG_FEATURE_NAMES), 124)
        for old_name in ('TotalSW', 'SWdensity', 'SWp2p_mean', 'SWnegSlope_mean', 'SWduration_mean'):
            self.assertFalse(any(old_name in name for name in eeg_processing.EEG_FEATURE_NAMES))


if __name__ == '__main__':
    unittest.main()
