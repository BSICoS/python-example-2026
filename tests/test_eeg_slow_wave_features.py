from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

import numpy as np

from src import eeg_processing
from src.lib import eeg_features
from src.lib.swa.swa_FindSWRef import swa_FindSWRef
from src.lib.swa.swa_getInfoDefaults import swa_getInfoDefaults


class SlowWaveFeatureTests(unittest.TestCase):
    @staticmethod
    def _zc_info(fs=10, stages=None):
        info = swa_getInfoDefaults({}, 'SW', method='envelope')
        info['Recording'] = {'sRate': fs}
        info['Parameters'].update({
            'Ref_InspectionPoint': 'ZC', 'Ref_AmplitudeCriteria': 'absolute',
            'Ref_AmplitudeAbsolute': 1.0, 'Ref_AmplitudeMax': 1000.0,
            'Ref_SlopeMin': .9, 'Ref_WaveLength': [.1, 2.0],
        })
        if stages is not None:
            info['Parameters']['Ref_UseStages'] = [1, 2]
        return info

    @staticmethod
    def _fake_peaks(*_args, **_kwargs):
        # The implementation discards its first peak, leaving one NREM peak.
        return np.array([0, 1]), np.array([0, 1])

    def test_stage_gate_excludes_wake_from_slope_threshold_only_when_active(self):
        signal = np.array([1., 1., -1., -5., -1., 1., 2., 102., 202.])
        stages = np.array([2, 2, 2, 2, 2, 2, 2, 5, 5])
        expected_nrem = np.percentile(np.diff(signal, prepend=signal[0])[(np.diff(signal, prepend=signal[0]) > 0) & np.isin(stages, [1, 2])], 90)
        expected_all = np.percentile(np.diff(signal, prepend=signal[0])[np.diff(signal, prepend=signal[0]) > 0], 90)
        with patch('src.lib.swa.swa_get_peaks.swa_get_peaks', self._fake_peaks):
            _, gated_info, _ = swa_FindSWRef({'SWRef': signal[np.newaxis, :], 'sleep_stages': stages}, self._zc_info(stages=stages))
            _, current_info, _ = swa_FindSWRef({'SWRef': signal[np.newaxis, :]}, self._zc_info())
        self.assertAlmostEqual(gated_info['Recording']['Slope_Threshold'][0], expected_nrem)
        self.assertAlmostEqual(current_info['Recording']['Slope_Threshold'][0], expected_all)
        self.assertNotEqual(expected_nrem, expected_all)

    def test_zc_stage_gate_requires_trough_in_nrem(self):
        signal = np.array([1., 1., -1., -5., -1., 1., 1.])
        accepted = np.full(signal.size, 2)
        rejected = accepted.copy(); rejected[3] = 5  # DZC is sample 1, trough is sample 3.
        with patch('src.lib.swa.swa_get_peaks.swa_get_peaks', self._fake_peaks):
            _, _, waves_rejected = swa_FindSWRef(
                {'SWRef': signal[np.newaxis, :], 'sleep_stages': rejected}, self._zc_info(stages=rejected))
            _, _, waves_accepted = swa_FindSWRef(
                {'SWRef': signal[np.newaxis, :], 'sleep_stages': accepted}, self._zc_info(stages=accepted))
        self.assertEqual(waves_rejected, [])
        self.assertEqual(len(waves_accepted), 1)
        self.assertEqual(waves_accepted[0]['Ref_PeakInd'], 3)

    def test_summarize_slow_waves_returns_fixed_numeric_features(self):
        waves = [
            {
                'Ref_DownInd': 10,
                'Ref_UpInd': 110,
                'Ref_PeakAmp': -80,
                'Ref_P2PAmp': 130,
                'Ref_NegSlope': -4,
                'Ref_PosSlope': 5,
            },
            {
                'Ref_DownInd': 200,
                'Ref_UpInd': 350,
                'Ref_PeakAmp': -120,
                'Ref_P2PAmp': 190,
                'Ref_NegSlope': -8,
                'Ref_PosSlope': 9,
            },
        ]

        features = eeg_features.summarize_slow_waves(
            waves, fs=100, signal_duration_seconds=120
        )

        self.assertEqual(tuple(features), eeg_features.SLOW_WAVE_FEATURE_NAMES)
        self.assertEqual(features['TotalSW'], 2.0)
        self.assertEqual(features['SWdensity'], 1.0)
        self.assertEqual(features['SWp2p_mean'], 160.0)
        self.assertEqual(features['SWnegSlope_mean'], -6.0)
        self.assertAlmostEqual(features['SWduration_mean'], 1.25)

    def test_summarize_no_waves_distinguishes_absence_from_missing_morphology(self):
        features = eeg_features.summarize_slow_waves(
            [], fs=200, signal_duration_seconds=300
        )

        self.assertEqual(features['TotalSW'], 0.0)
        self.assertEqual(features['SWdensity'], 0.0)
        for name in eeg_features.SLOW_WAVE_FEATURE_NAMES[2:]:
            self.assertTrue(np.isnan(features[name]), name)

    def test_swa_reference_detector_uses_channels_by_samples_orientation(self):
        fs = 100
        seconds = 20
        time = np.arange(fs * seconds) / fs
        reference = 100.0 * np.sin(2 * np.pi * time)

        info = swa_getInfoDefaults({}, 'SW', method='envelope')
        info['Recording'] = {'sRate': fs}
        info['Parameters']['Ref_InspectionPoint'] = 'ZC'
        info['Parameters']['Ref_AmplitudeCriteria'] = 'absolute'
        info['Parameters']['Ref_AmplitudeAbsolute'] = 50.0
        data = {'SWRef': reference[np.newaxis, :]}

        with redirect_stdout(io.StringIO()):
            _, _, waves = swa_FindSWRef(data, info)

        self.assertGreater(len(waves), 0)
        self.assertTrue(all(0 <= wave['Ref_PeakInd'] < reference.size for wave in waves))

    def test_get_sw_features_detects_a_synthetic_slow_wave_signal(self):
        fs = 100
        time = np.arange(fs * 20) / fs
        signal = 100.0 * np.sin(2 * np.pi * time)

        features = eeg_features.get_SW_features(signal, fs)

        self.assertGreater(features['TotalSW'], 10)
        self.assertGreater(features['SWdensity'], 30)
        self.assertGreater(features['SWp2p_mean'], 160)
        self.assertTrue(0.4 < features['SWduration_mean'] < 0.6)

    def test_get_sw_features_aggregates_swa_events(self):
        fs = 100
        signal = np.zeros(fs * 60, dtype=float)
        waves = [{
            'Ref_DownInd': 100,
            'Ref_UpInd': 180,
            'Ref_PeakInd': 140,
            'Ref_PeakAmp': -75,
            'Ref_P2PAmp': 125,
            'Ref_NegSlope': -3,
            'Ref_PosSlope': 4,
        }]

        with (
            patch.object(
                eeg_features.swa_CalculateReference,
                'swa_CalculateReference',
                return_value=(signal[np.newaxis, :], {
                    'Recording': {'sRate': fs},
                    'Parameters': {},
                    'Electrodes': ['EEG'],
                }),
            ),
            patch.object(
                eeg_features.swa_FindSWRef,
                'swa_FindSWRef',
                side_effect=lambda data, info: (data, info, waves),
            ),
            patch.object(
                eeg_features.swa_FindSWChannels,
                'swa_FindSWChannels',
                side_effect=lambda data, info, detected, flag_progress: (
                    data, info, detected
                ),
            ),
        ):
            features = eeg_features.get_SW_features(signal, fs)

        self.assertEqual(tuple(features), eeg_features.SLOW_WAVE_FEATURE_NAMES)
        self.assertEqual(features['TotalSW'], 1.0)
        self.assertEqual(features['SWdensity'], 1.0)
        self.assertEqual(features['SWp2p_mean'], 125.0)

    def test_background_metrics_do_not_depend_on_slow_wave_extraction(self):
        fs = 200
        time = np.arange(fs * 30) / fs
        signal = (
            20.0 * np.sin(2 * np.pi * 10 * time)
            + 5.0 * np.sin(2 * np.pi * time)
        )

        with patch.object(eeg_features, 'get_SW_features', side_effect=RuntimeError('unused')):
            metrics = eeg_processing._extract_channel_metrics(signal, fs)

        self.assertIsNotNone(metrics)
        self.assertTrue(np.isfinite(metrics['Relative_Delta_Power']))

    def test_shared_sw_preparation_is_sanitized_resampled_not_spectral_normalized(self):
        fs = 200
        time = np.arange(fs * 30) / fs
        raw = 20.0 * np.sin(2 * np.pi * time) + 3.0 * np.sin(2 * np.pi * 10 * time)
        raw[[0, 1, 2]] = [np.nan, np.inf, -np.inf]
        detector_signal, detector_fs = eeg_processing.prepare_slow_wave_detector_input(raw, fs)

        expected = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        np.testing.assert_array_equal(detector_signal, expected)
        self.assertEqual(detector_fs, 200)

        spectral = eeg_features.butter_bandpass_filter(
            expected, lowcut=.3, highcut=35, fs=200, order=4)
        normalized = (spectral - np.mean(spectral)) / np.std(spectral)
        self.assertFalse(np.allclose(detector_signal, normalized))

    def test_shared_preparation_preserves_process_eeg_features(self):
        fs = 200
        time = np.arange(fs * 30) / fs
        physiological_data = {
            channel.lower(): (10 + index) * np.sin(2 * np.pi * time)
            + 2 * np.sin(2 * np.pi * 10 * time)
            for index, channel in enumerate(eeg_processing.EEG_CHANNEL_SPECS)
        }
        physiological_fs = {channel: fs for channel in physiological_data}
        fixed_sw = {
            'TotalSW': 3.0, 'SWdensity': 6.0, 'SWp2p_mean': 2.0,
            'SWnegSlope_mean': -1.0, 'SWduration_mean': .5,
        }

        def legacy_prepare(signal, input_fs):
            prepared = np.nan_to_num(
                np.asarray(signal, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            if prepared.size < max(int(input_fs * 30), 2):
                return None
            if input_fs != 200:
                prepared, input_fs = eeg_processing.resample_signal(prepared, input_fs, 200)
            return prepared, input_fs

        with patch.object(eeg_features, 'get_SW_features', return_value=fixed_sw):
            actual = eeg_processing.processEEG(
                physiological_data, physiological_fs, 'channel_table.csv')
            with patch.object(
                eeg_processing, 'prepare_slow_wave_detector_input',
                side_effect=legacy_prepare,
            ):
                legacy = eeg_processing.processEEG(
                    physiological_data, physiological_fs, 'channel_table.csv')

        np.testing.assert_allclose(actual, legacy, rtol=0, atol=0, equal_nan=True)

    def test_final_eeg_schema_contains_record_level_slow_wave_features(self):
        for channel_name in eeg_processing.EEG_CHANNEL_SPECS:
            for feature_name in eeg_processing.SLOW_WAVE_METRICS:
                self.assertIn(
                    f'{channel_name}_{feature_name}',
                    eeg_processing.EEG_FEATURE_NAMES,
                )
        self.assertEqual(
            len(eeg_processing.EEG_FEATURE_NAMES),
            eeg_processing.EEG_FEATURE_LENGTH,
        )


if __name__ == '__main__':
    unittest.main()
