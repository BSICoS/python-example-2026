import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import eeg_processing
from src.lib import eeg_features
from src.pipeline.features import _iter_signal_segments
from src.slow_wave_audit import (
    TriggeredWaveforms,
    _aggregate_summary,
    _subject_rows_for_channel,
    annotation_at_time,
    build_segment_intervals,
    stage_minutes_in_interval,
    translate_stage_code,
)


class SlowWaveAuditTests(unittest.TestCase):
    def test_caisr_stage_codes_are_translated(self):
        expected = {1: 'N3', 2: 'N2', 3: 'N1', 4: 'REM', 5: 'Wake', 9: 'unavailable'}
        self.assertEqual({code: translate_stage_code(code) for code in expected}, expected)
        self.assertEqual(translate_stage_code(42), 'unavailable')

    def test_event_time_maps_to_epoch_containing_trough(self):
        annotation = {
            'available': True,
            'fs': 1 / 30,
            'stage': np.array([5, 2, 1, 4], dtype=float),
            'probabilities': {
                'caisr_prob_n3': np.array([0, .1, .8, 0]),
                'caisr_prob_n2': np.array([0, .7, .1, 0]),
                'caisr_prob_n1': np.zeros(4),
                'caisr_prob_r': np.array([0, 0, 0, .9]),
                'caisr_prob_w': np.array([1, .2, .1, .1]),
            },
        }
        self.assertEqual(annotation_at_time(annotation, 89.999)['stage_at_trough'], 'N3')
        self.assertEqual(annotation_at_time(annotation, 60.0)['stage_at_trough'], 'N3')
        self.assertAlmostEqual(annotation_at_time(annotation, 60.0)['caisr_prob_n3'], .8)

    def test_annotation_unavailable_is_non_fatal(self):
        annotation = {'available': False, 'fs': 1 / 30, 'stage': np.array([]),
                      'probabilities': {}}
        self.assertEqual(annotation_at_time(annotation, 10)['stage_at_trough'], 'unavailable')
        minutes = stage_minutes_in_interval(annotation, 0, 300)
        self.assertEqual(minutes['unavailable'], 5.0)

    def test_density_uses_actual_stage_minutes(self):
        annotation = {
            'available': True, 'fs': 1 / 30,
            'stage': np.array([1, 1, 2, 5], dtype=float), 'probabilities': {},
        }
        record = {'patient_id': 'p', 'bids_folder': 'sub-p', 'site_id': 'A', 'session_id': '1'}
        events = [{'stage_at_trough': 'N3'}, {'stage_at_trough': 'N3'},
                  {'stage_at_trough': 'N2'}]
        row = _subject_rows_for_channel(record, 'C3-M2', annotation, 120, [(0, 120)], events)
        self.assertEqual(row['analyzed_N3_minutes'], 1.0)
        self.assertEqual(row['analyzed_N2_minutes'], .5)
        self.assertEqual(row['SW_per_min_N3'], 2.0)
        self.assertEqual(row['SW_per_min_N2'], 2.0)
        self.assertEqual(row['SW_per_min_N2_N3'], 2.0)

    def test_audit_segmentation_matches_production(self):
        fs = 2
        duration = 2100
        data = {'x': np.zeros(fs * duration)}
        production = _iter_signal_segments(data, {'x': fs})
        audit = build_segment_intervals(duration)
        self.assertEqual(len(production), len(audit))
        self.assertEqual(audit, [(0.0, 300.0), (900.0, 1200.0), (1800.0, 2100.0)])

    def test_channel_source_selection_is_shared_with_production(self):
        aliases = {
            'c3 m2': {'c3 m2'}, 'c3': {'c3'}, 'm2': {'m2'},
        }
        fs = {'C3-M2': 200, 'C3': 200, 'M2': 200}
        direct = eeg_processing.get_eeg_channel_source_labels(
            'C3-M2', fs.keys(), fs, aliases)
        self.assertEqual(direct, ('C3-M2',))
        data = {'C3-M2': np.arange(4), 'C3': np.ones(4), 'M2': np.zeros(4)}
        signal, selected_fs = eeg_processing._get_channel_signal('C3-M2', data, fs, aliases)
        np.testing.assert_array_equal(signal, data['C3-M2'])
        self.assertEqual(selected_fs, 200)
        derived = eeg_processing.get_eeg_channel_source_labels(
            'C3-M2', ['C3', 'M2'], fs, aliases)
        self.assertEqual(derived, ('C3', 'M2'))

    def test_site_summaries_do_not_mix_hospitals(self):
        subjects = pd.DataFrame([
            {'patient_id': 'a', 'session_id': '1', 'site_id': 'A', 'channel': 'C3-M2',
             'annotation_available': True, 'number_in_N2': 10, 'number_in_N3': 0,
             'number_in_REM': 0, 'number_in_Wake': 0, 'analyzed_minutes_N2': 5,
             'analyzed_minutes_N3': 0, 'analyzed_minutes_REM': 0, 'analyzed_minutes_Wake': 0},
            {'patient_id': 'b', 'session_id': '1', 'site_id': 'B', 'channel': 'C3-M2',
             'annotation_available': True, 'number_in_N2': 1, 'number_in_N3': 0,
             'number_in_REM': 0, 'number_in_Wake': 0, 'analyzed_minutes_N2': 5,
             'analyzed_minutes_N3': 0, 'analyzed_minutes_REM': 0, 'analyzed_minutes_Wake': 0},
        ])
        events = pd.DataFrame(columns=['site_id', 'channel', 'peak_to_peak_amplitude',
                                      'negative_half_wave_duration_seconds',
                                      'detector_amplitude_threshold'])
        segments = pd.DataFrame(columns=['site_id', 'channel', 'TotalSW', 'SWdensity'])
        summary = _aggregate_summary(subjects, events, segments, ['site_id']).set_index('site_id')
        self.assertEqual(summary.loc['A', 'SW_per_min_N2'], 2.0)
        self.assertEqual(summary.loc['B', 'SW_per_min_N2'], .2)

    def test_event_triggered_average_is_aligned_by_trough(self):
        accumulator = TriggeredWaveforms(max_quantile_samples=10)
        accumulator.add(('A', 'C3-M2', 'N3'), np.array([0.0, -2.0, 0.0]))
        accumulator.add(('A', 'C3-M2', 'N3'), np.array([1.0, -4.0, 1.0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'eta.npz'
            accumulator.save(path, fs=.5)
            with np.load(path) as payload:
                np.testing.assert_array_equal(payload['A__C3_M2__N3__mean'], [.5, -3, .5])
                self.assertEqual(int(payload['A__C3_M2__N3__n_events']), 2)

    def test_get_sw_features_uses_the_same_exposed_events(self):
        fs = 100
        time = np.arange(fs * 20) / fs
        signal = 100 * np.sin(2 * np.pi * time)
        detection = eeg_features.detect_slow_waves(signal, fs)
        expected = eeg_features.summarize_slow_waves(
            detection['events'], fs, detection['signal_duration_seconds'])
        actual = eeg_features.get_SW_features(signal, fs)
        self.assertEqual(tuple(actual), eeg_features.SLOW_WAVE_FEATURE_NAMES)
        for name in actual:
            self.assertAlmostEqual(actual[name], expected[name], places=12)


if __name__ == '__main__':
    unittest.main()
