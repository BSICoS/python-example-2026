import unittest
from unittest.mock import patch

import numpy as np

from src.lib import eeg_features
from src.slow_wave_audit import build_segment_intervals
from src.slow_wave_methods import METHODS, caisr_stages_to_detector_samples


class SlowWaveMethodTests(unittest.TestCase):
    def test_epoch_stages_expand_by_floor_without_interpolation(self):
        annotation = {'available': True, 'fs': 1 / 30,
                      'stage': np.array([5, 2, 1], dtype=float), 'probabilities': {}}
        stages = caisr_stages_to_detector_samples(annotation, 15, 90, 2)
        self.assertTrue(np.all(stages[:30] == 5))
        self.assertTrue(np.all(stages[30:90] == 2))

    def test_variants_have_the_requested_interval_schedules(self):
        current = build_segment_intervals(900, METHODS['current']['stride_seconds'])
        sampled = build_segment_intervals(900, METHODS['nrem_sampled']['stride_seconds'])
        full = build_segment_intervals(900, METHODS['nrem_full']['stride_seconds'])
        self.assertEqual(current, sampled)
        self.assertEqual(full, [(float(x), float(x + 300)) for x in range(0, 601, 300)])

    def test_default_detector_does_not_activate_stage_gate(self):
        signal = np.sin(2 * np.pi * np.arange(2000) / 200)
        captured = {}

        def find_ref(data, info):
            captured['parameters'] = dict(info['Parameters'])
            captured['has_stages'] = 'sleep_stages' in data
            return data, info, []

        with patch.object(eeg_features.swa_FindSWRef, 'swa_FindSWRef', side_effect=find_ref):
            eeg_features.detect_slow_waves(signal, 200)
        self.assertIsNone(captured['parameters']['Ref_UseStages'])
        self.assertFalse(captured['has_stages'])

    def test_stage_aware_detector_forwards_only_n2_n3_gate(self):
        signal = np.sin(2 * np.pi * np.arange(2000) / 200)
        captured = {}

        def find_ref(data, info):
            captured['stages'] = data['sleep_stages'].copy()
            captured['allowed'] = info['Parameters']['Ref_UseStages']
            return data, info, []

        with patch.object(eeg_features.swa_FindSWRef, 'swa_FindSWRef', side_effect=find_ref):
            eeg_features.detect_slow_waves(signal, 200, sleep_stages=np.full(len(signal), 2), allowed_stages=(1, 2))
        self.assertEqual(captured['allowed'], [1, 2])
        self.assertTrue(np.all(captured['stages'] == 2))

    def test_stage_vector_must_match_detector_input(self):
        with self.assertRaises(ValueError):
            eeg_features.detect_slow_waves(np.zeros(100), 200, sleep_stages=np.zeros(99), allowed_stages=(1, 2))


if __name__ == '__main__':
    unittest.main()
