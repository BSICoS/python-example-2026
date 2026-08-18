from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

import numpy as np

from src.pipeline.cross_validation import (
    CrossValidationConfig,
    CrossValidationResult,
    EnsembleCrossValidator,
)
from src.pipeline.training import (
    _build_modality_comparison_runner,
    _format_modality_comparison_by_hospital,
    _get_modality_comparison_feature_sets,
    _run_modality_comparisons,
    _run_modality_comparisons_across_seeds,
)


class ModalityComparisonTests(unittest.TestCase):
    def test_feature_sets_keep_demographics_with_requested_modalities(self):
        feature_indices = {
            'ecg': np.array([0, 1, 2], dtype=np.int32),
            'eeg': np.array([0, 3, 4], dtype=np.int32),
            'resp': np.array([0, 5], dtype=np.int32),
        }

        feature_sets = _get_modality_comparison_feature_sets(feature_indices)

        self.assertEqual(set(feature_sets['EEG']), {'eeg'})
        self.assertTrue(np.array_equal(feature_sets['EEG']['eeg'], [0, 3, 4]))
        self.assertEqual(set(feature_sets['ECG + EEG']), {'ecg', 'eeg'})
        self.assertTrue(np.array_equal(feature_sets['ECG + EEG']['ecg'], [0, 1, 2]))
        self.assertTrue(np.array_equal(feature_sets['ECG + EEG']['eeg'], [0, 3, 4]))
        self.assertEqual(set(feature_sets['ALL']), {'ecg', 'eeg', 'resp'})
        self.assertTrue(np.array_equal(feature_sets['ALL']['resp'], [0, 5]))

    def test_each_modality_configuration_runs_its_own_searches(self):
        features = np.array([
            [20.0, 0.1, 0.2, 0.3, 0.4], [21.0, 0.9, 0.8, 0.7, 0.6],
            [22.0, 0.2, 0.3, 0.4, 0.5], [23.0, 0.8, 0.7, 0.6, 0.4],
            [24.0, 0.3, 0.4, 0.5, 0.6], [25.0, 0.7, 0.6, 0.5, 0.4],
            [26.0, 0.4, 0.5, 0.6, 0.7], [27.0, 0.6, 0.5, 0.4, 0.3],
        ], dtype=np.float32)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
        feature_indices = {
            'ecg': np.array([0, 1], dtype=np.int32),
            'eeg': np.array([0, 2], dtype=np.int32),
            'resp': np.array([0, 3], dtype=np.int32),
        }
        modality_presence_indices = {
            'ecg': np.array([1], dtype=np.int32),
            'eeg': np.array([2], dtype=np.int32),
            'resp': np.array([3], dtype=np.int32),
        }
        runner = EnsembleCrossValidator(
            config=CrossValidationConfig(
                search_scoring='roc_auc',
                use_site_grouped_cv=False,
                optimize_hyperparameter_search=True,
                outer_random_splits=2,
                search_iterations=1,
            ),
            param_dist={'max_depth': [1]},
            default_threshold=0.5,
            build_preprocessor=lambda *args: None,
            build_search_model=lambda fold_labels: object(),
            fit_ensemble=lambda *args, **kwargs: {},
            predict_probabilities=lambda bundle, values: values[:, 1],
            search_age_feature_index=0,
            run_final_search=False,
        )

        with patch.object(runner, '_search_hyperparams', return_value=({'max_depth': 1}, 0.5)) as search:
            with redirect_stdout(io.StringIO()):
                _run_modality_comparisons(
                    features,
                    labels,
                    feature_indices,
                    modality_presence_indices,
                    [],
                    runner,
                )

        self.assertEqual(search.call_count, 6)
        self.assertTrue(all(len(call.args[0]) == 4 for call in search.call_args_list))
        self.assertEqual([call.args[0].shape[1] for call in search.call_args_list], [2, 2, 3, 3, 4, 4])

    def test_seed_loop_runs_all_modalities_with_seeded_nested_cv_only(self):
        features = np.array([
            [20.0, 0.1, 0.2, 0.3], [21.0, 0.9, 0.8, 0.7],
            [22.0, 0.2, 0.3, 0.4], [23.0, 0.8, 0.7, 0.6],
        ], dtype=np.float32)
        labels = np.array([0, 1, 0, 1], dtype=np.int32)
        feature_indices = {
            'ecg': np.array([0, 1], dtype=np.int32),
            'eeg': np.array([0, 2], dtype=np.int32),
            'resp': np.array([0, 3], dtype=np.int32),
        }
        modality_presence_indices = {
            'ecg': np.array([1], dtype=np.int32),
            'eeg': np.array([2], dtype=np.int32),
            'resp': np.array([3], dtype=np.int32),
        }
        configured_seeds = [1, 2, 3, 4, 5]
        created_seeds = []
        run_calls = []

        class RecordingRunner:
            def __init__(self, seed):
                self.seed = seed

            def run(self, *args, **kwargs):
                run_calls.append((self.seed, args[2]))
                return CrossValidationResult(
                    threshold=0.5,
                    final_params=None,
                    final_search_score=None,
                    metrics={
                        'oof_calibrated_metrics': {'age_conditioned_auroc': 0.5},
                    },
                )

        def runner_factory(feature_names, seed):
            created_seeds.append(seed)
            return RecordingRunner(seed)

        results = _run_modality_comparisons_across_seeds(
            features,
            labels,
            ['Age', 'ECG', 'EEG', 'RESP'],
            feature_indices,
            modality_presence_indices,
            [],
            configured_seeds,
            runner_factory=runner_factory,
        )

        self.assertEqual(created_seeds, configured_seeds)
        self.assertEqual(list(results), configured_seeds)
        self.assertEqual(len(run_calls), len(configured_seeds) * 3)
        self.assertEqual(
            [seed for seed, _ in run_calls],
            [seed for seed in configured_seeds for _ in range(3)],
        )
        self.assertTrue(all(result.final_params is None for results in results.values() for _, result in results.values()))

    def test_seeded_runner_skips_final_search(self):
        runner = _build_modality_comparison_runner(['Age', 'Sex'], random_state=3)

        self.assertEqual(runner.config.random_state, 3)
        self.assertFalse(runner.run_final_search)

    def test_hospital_groups_are_passed_to_all_modality_comparisons(self):
        features = np.array([
            [20.0, 0.1, 0.2, 0.3], [21.0, 0.9, 0.8, 0.7],
            [22.0, 0.2, 0.3, 0.4], [23.0, 0.8, 0.7, 0.6],
        ], dtype=np.float32)
        labels = np.array([0, 1, 0, 1], dtype=np.int32)
        feature_indices = {
            'ecg': np.array([0, 1], dtype=np.int32),
            'eeg': np.array([0, 2], dtype=np.int32),
            'resp': np.array([0, 3], dtype=np.int32),
        }
        modality_presence_indices = {
            'ecg': np.array([1], dtype=np.int32),
            'eeg': np.array([2], dtype=np.int32),
            'resp': np.array([3], dtype=np.int32),
        }
        site_groups = np.array(['HOSP1', 'HOSP1', 'HOSP2', 'HOSP2'])
        received_site_groups = []

        class RecordingRunner:
            def run(self, *args, **kwargs):
                received_site_groups.append(kwargs['site_groups'])
                return CrossValidationResult(
                    threshold=0.5,
                    final_params=None,
                    final_search_score=None,
                    metrics={},
                )

        _run_modality_comparisons(
            features,
            labels,
            feature_indices,
            modality_presence_indices,
            [],
            RecordingRunner(),
            site_groups=site_groups,
        )
        runner = _build_modality_comparison_runner(
            ['Age', 'Sex'],
            use_site_grouped_cv=True,
        )

        self.assertEqual(len(received_site_groups), 3)
        self.assertTrue(all(np.array_equal(groups, site_groups) for groups in received_site_groups))
        self.assertTrue(runner.config.use_site_grouped_cv)
        self.assertFalse(runner.run_final_search)

    def test_hospital_summary_lists_each_held_out_site(self):
        def result_for(scores):
            return CrossValidationResult(
                threshold=0.5,
                final_params=None,
                final_search_score=None,
                metrics={
                    'fold_metrics': [
                        {'held_out_site': hospital, 'age_conditioned_auroc': score}
                        for hospital, score in scores.items()
                    ],
                },
            )

        summary = _format_modality_comparison_by_hospital({
            'EEG': (4, result_for({'I0002': 0.4, 'I0006': 0.5, 'S0001': 0.6})),
            'ECG + EEG': (4, result_for({'I0002': 0.6, 'I0006': 0.4, 'S0001': 0.7})),
            'ALL': (4, result_for({'I0002': 0.5, 'I0006': 0.3, 'S0001': 0.8})),
        })

        self.assertIn('Held-out hospital Age-AUROC', summary)
        self.assertIn('I0002     0.400   0.600     0.500   0.200', summary)
        self.assertIn('I0006     0.500   0.400     0.300   -0.100', summary)
        self.assertIn('S0001     0.600   0.700     0.800   0.100', summary)


if __name__ == '__main__':
    unittest.main()