import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import eeg_processing, slow_wave_audit
from src.lib import eeg_features
from src.pipeline.features import _iter_signal_segments
from src.slow_wave_audit import (
    TriggeredWaveforms,
    EVENT_COLUMNS,
    _aggregate_summary,
    _build_stage_summary,
    _subject_rows_for_channel,
    annotation_at_time,
    build_segment_intervals,
    caisr_uncertainty_metrics,
    detect_audited_slow_waves,
    physical_signal_statistics,
    prepare_audit_slow_wave_detector_input,
    stage_minutes_in_interval,
    translate_stage_code,
    weighted_event_metrics,
    weighted_stage_minutes_in_interval,
)


class SlowWaveAuditTests(unittest.TestCase):
    @staticmethod
    def _annotation(stages, n3, n2, n1=None, rem=None, wake=None):
        length = len(stages)
        zeros = np.zeros(length, dtype=float)
        return {
            'available': True,
            'fs': 1 / 30,
            'stage': np.asarray(stages, dtype=float),
            'probabilities': {
                'caisr_prob_n3': np.asarray(n3, dtype=float),
                'caisr_prob_n2': np.asarray(n2, dtype=float),
                'caisr_prob_n1': zeros if n1 is None else np.asarray(n1, dtype=float),
                'caisr_prob_r': zeros if rem is None else np.asarray(rem, dtype=float),
                'caisr_prob_w': zeros if wake is None else np.asarray(wake, dtype=float),
            },
        }

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

    def test_soft_weights_do_not_replace_the_hard_stage(self):
        annotation = self._annotation([5], n3=[.3], n2=[.4], wake=[.3])
        event = annotation_at_time(annotation, 10)
        self.assertEqual(event['stage_at_trough'], 'Wake')
        self.assertEqual(event['weight_N2'], .4)
        self.assertEqual(event['weight_N3'], .3)
        self.assertAlmostEqual(event['weight_NREM'], .7)

    def test_weighted_nrem_exposure_and_density(self):
        annotation = self._annotation([2], n3=[.3], n2=[.5], wake=[.2])
        exposure = weighted_stage_minutes_in_interval(annotation, 0, 30)
        self.assertAlmostEqual(exposure['NREM'], .4)
        metrics = weighted_event_metrics([{'weight_NREM': .7}], exposure)
        self.assertAlmostEqual(metrics['weighted_SW_count_NREM'], .7)
        self.assertAlmostEqual(metrics['weighted_SW_per_min_NREM'], 1.75)

    def test_partial_epoch_overlaps_are_probability_weighted(self):
        annotation = self._annotation([2, 2], n3=[.3, .1], n2=[.5, .1])
        exposure = weighted_stage_minutes_in_interval(annotation, 15, 45)
        self.assertAlmostEqual(exposure['NREM'], .25)

    def test_missing_caisr_probability_is_not_invented(self):
        annotation = self._annotation([2], n3=[.3], n2=[.4])
        del annotation['probabilities']['caisr_prob_n2']
        event = annotation_at_time(annotation, 5)
        self.assertTrue(np.isnan(event['weight_N2']))
        self.assertTrue(np.isnan(event['weight_NREM']))
        exposure = weighted_stage_minutes_in_interval(annotation, 0, 30)
        self.assertEqual(exposure['NREM'], 0.0)

    def test_caisr_uncertainty_metrics_are_deterministic(self):
        annotation = self._annotation(
            [2, 5], n3=[.1, .1], n2=[.6, .1], n1=[.1, .1],
            rem=[.1, .1], wake=[.1, .6])
        metrics = caisr_uncertainty_metrics(annotation)
        self.assertEqual(metrics['caisr_probability_epochs'], 2)
        self.assertAlmostEqual(metrics['median_max_stage_probability'], .6)
        self.assertEqual(metrics['fraction_epochs_max_probability_below_0_5'], 0.0)
        self.assertEqual(metrics['fraction_epochs_max_probability_below_0_7'], 1.0)
        self.assertGreater(metrics['mean_stage_entropy'], 0)

    def test_annotation_unavailable_is_non_fatal(self):
        annotation = {'available': False, 'fs': 1 / 30, 'stage': np.array([]),
                      'probabilities': {}}
        self.assertEqual(annotation_at_time(annotation, 10)['stage_at_trough'], 'unavailable')
        minutes = stage_minutes_in_interval(annotation, 0, 300)
        self.assertEqual(minutes['unavailable'], 5.0)

    def test_audit_and_production_share_the_same_sw_preparation(self):
        self.assertIs(
            slow_wave_audit.prepare_slow_wave_detector_input,
            eeg_processing.prepare_slow_wave_detector_input,
        )
        signal = np.linspace(-2, 2, 6000)
        signal[[0, 1, 2]] = [np.nan, np.inf, -np.inf]
        production_signal, production_fs = eeg_processing.prepare_slow_wave_detector_input(
            signal, 200)
        audit_signal, audit_fs = prepare_audit_slow_wave_detector_input(signal, 200)
        expected = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        np.testing.assert_array_equal(production_signal, expected)
        np.testing.assert_array_equal(audit_signal, expected)
        self.assertEqual((production_fs, audit_fs), (200, 200))

        spectral_filtered = eeg_features.butter_bandpass_filter(
            expected, lowcut=.3, highcut=35, fs=200, order=4)
        spectral_normalized = (
            spectral_filtered - np.mean(spectral_filtered)
        ) / np.std(spectral_filtered)
        self.assertFalse(np.allclose(audit_signal, spectral_normalized))

    def test_fixed_signal_reaches_both_detectors_identically(self):
        fs = 200
        time = np.arange(fs * 30) / fs
        signal = 10 * np.sin(2 * np.pi * time) + np.linspace(-1, 1, len(time))
        signal[0] = np.nan
        captured = {}
        fixed_features = {name: 0.0 for name in eeg_features.SLOW_WAVE_FEATURE_NAMES}

        def capture_production(detector_input, detector_fs):
            captured['production'] = np.asarray(detector_input).copy()
            return fixed_features

        with patch.object(eeg_features, 'get_SW_features', side_effect=capture_production):
            eeg_processing._extract_channel_metrics(signal, fs)

        def capture_audit(detector_input, detector_fs):
            captured['audit'] = np.asarray(detector_input).copy()
            return {'events': []}

        with patch.object(eeg_features, 'detect_slow_waves', side_effect=capture_audit):
            _, audit_input, audit_fs = detect_audited_slow_waves(signal, fs)

        np.testing.assert_array_equal(captured['production'], captured['audit'])
        np.testing.assert_array_equal(audit_input, captured['audit'])
        self.assertEqual(audit_fs, 200)

    def test_auditor_event_fields_are_not_mislabeled_as_normalized(self):
        for field in ('negative_peak_amplitude', 'peak_to_peak_amplitude',
                      'negative_slope', 'positive_slope',
                      'detector_amplitude_threshold', 'detector_data_deviation',
                      'detector_slope_threshold'):
            self.assertIn(field, EVENT_COLUMNS)
            self.assertNotIn(f'{field}_normalized', EVENT_COLUMNS)

    def test_physical_statistics_use_original_signal_scale(self):
        physical = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
        normalized = (physical - physical.mean()) / physical.std()
        stats = physical_signal_statistics(physical)
        normalized_stats = physical_signal_statistics(normalized)
        self.assertEqual(stats['signal_median'], 0.0)
        self.assertGreater(stats['signal_P99_minus_P1'], 100)
        self.assertLess(normalized_stats['signal_P99_minus_P1'], 4)

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

    def test_hard_and_soft_coverage_aggregate_from_sums(self):
        rows = []
        for patient, total_hard, analyzed_hard, total_soft, analyzed_soft in (
            ('a', 2.0, 1.0, 10.0, 1.0),
            ('b', 10.0, 9.0, 10.0, 9.0),
        ):
            row = {
                'patient_id': patient, 'session_id': '1', 'site_id': 'A',
                'channel': 'C3-M2', 'annotation_available': True,
                'number_in_N2': 0, 'number_in_N3': 0,
                'number_in_REM': 0, 'number_in_Wake': 0,
                'analyzed_minutes_N2': analyzed_hard,
                'analyzed_minutes_N3': 0.0,
                'analyzed_minutes_REM': 0.0, 'analyzed_minutes_Wake': 0.0,
                'total_N2_N3_minutes': total_hard,
                'total_weighted_NREM_minutes': total_soft,
                'analyzed_weighted_NREM_minutes': analyzed_soft,
            }
            for stage in ('N2', 'N3', 'NREM', 'REM', 'Wake'):
                row[f'weighted_SW_count_{stage}'] = 0.0
                row[f'weighted_minutes_{stage}'] = analyzed_soft if stage == 'NREM' else 0.0
            rows.append(row)
        subjects = pd.DataFrame(rows)
        events = pd.DataFrame(columns=['site_id', 'channel', 'peak_to_peak_amplitude',
                                      'negative_half_wave_duration_seconds',
                                      'detector_amplitude_threshold'])
        segments = pd.DataFrame(columns=['site_id', 'channel', 'TotalSW', 'SWdensity'])
        summary = _aggregate_summary(subjects, events, segments, ['site_id']).iloc[0]
        self.assertAlmostEqual(summary['analyzed_N2_N3_fraction'], 10 / 12)
        self.assertAlmostEqual(summary['analyzed_weighted_NREM_fraction'], .5)

    def test_soft_morphology_reports_weighted_mean_and_effective_weight(self):
        subjects = pd.DataFrame([{
            'site_id': 'A', 'channel': 'C3-M2',
            **{f'SW_per_min_{stage}': 0.0 for stage in
               ('N3', 'N2', 'N1', 'REM', 'Wake', 'unavailable')},
            **{f'weighted_SW_per_min_{stage}': 0.0 for stage in
               ('N2', 'N3', 'NREM', 'REM', 'Wake')},
        }])
        events = pd.DataFrame([
            {'site_id': 'A', 'channel': 'C3-M2', 'stage_at_trough': 'N2',
             'peak_to_peak_amplitude': 10.0, 'negative_peak_amplitude': -5.0,
             'negative_slope': -1.0, 'positive_slope': 1.0,
             'negative_half_wave_duration_seconds': .5,
             'detector_amplitude_threshold': 2.0, 'detector_slope_threshold': .2,
             'weight_N2': .25, 'weight_N3': 0.0, 'weight_NREM': .25,
             'weight_REM': 0.0, 'weight_Wake': .75},
            {'site_id': 'A', 'channel': 'C3-M2', 'stage_at_trough': 'Wake',
             'peak_to_peak_amplitude': 30.0, 'negative_peak_amplitude': -15.0,
             'negative_slope': -3.0, 'positive_slope': 3.0,
             'negative_half_wave_duration_seconds': 1.0,
             'detector_amplitude_threshold': 2.0, 'detector_slope_threshold': .2,
             'weight_N2': .75, 'weight_N3': 0.0, 'weight_NREM': .75,
             'weight_REM': 0.0, 'weight_Wake': .25},
        ])
        summary = _build_stage_summary(events, subjects)
        row = summary[(summary.staging_method == 'soft') &
                      (summary.stage == 'NREM') &
                      (summary.metric == 'peak_to_peak_amplitude')].iloc[0]
        self.assertEqual(row.total_effective_weight, 1.0)
        self.assertEqual(row.weighted_mean, 25.0)

    def test_event_triggered_average_is_aligned_by_trough(self):
        accumulator = TriggeredWaveforms(max_quantile_samples=10)
        accumulator.add(('A', 'C3-M2', 'N3'), np.array([0.0, -2.0, 0.0]))
        accumulator.add(('A', 'C3-M2', 'N3'), np.array([1.0, -4.0, 1.0]))
        accumulator.add_weighted_nrem(('A', 'C3-M2'), np.array([0.0, -2.0, 0.0]), .25)
        accumulator.add_weighted_nrem(('A', 'C3-M2'), np.array([2.0, -4.0, 2.0]), .75)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'eta.npz'
            accumulator.save(path, fs=.5)
            with np.load(path) as payload:
                np.testing.assert_array_equal(payload['A__C3_M2__N3__mean'], [.5, -3, .5])
                self.assertEqual(int(payload['A__C3_M2__N3__n_events']), 2)
                self.assertEqual(str(payload['waveform_domain']), 'sanitized_resampled_eeg')
                np.testing.assert_array_equal(
                    payload['A__C3_M2__weighted_NREM__mean_waveform'],
                    [1.5, -3.5, 1.5],
                )

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
