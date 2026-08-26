import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.pipeline.cross_validation import CrossValidationConfig
from src.pipeline import training


class DiagnosticLohoTests(unittest.TestCase):
    def test_training_uses_only_random_cv_outputs_for_deployable_model(self):
        records = ['record-0', 'record-1']
        extracted = {
            'record-0': ({'patient_id': 'P0', 'site_id': 'I0002'}, np.array([40.0, 0.0, 1.0]), 0, None),
            'record-1': ({'patient_id': 'P1', 'site_id': 'I0006'}, np.array([41.0, 1.0, 2.0]), 1, None),
        }
        feature_indices = {
            'all': np.array([0, 1, 2], dtype=np.int32),
            'demographics': np.array([0, 1], dtype=np.int32),
            'resp': np.array([2], dtype=np.int32),
            'eeg': np.array([], dtype=np.int32),
            'ecg': np.array([], dtype=np.int32),
        }

        class FakePreprocessor:
            _numerical_indices = np.array([0, 2], dtype=np.int32)
            categorical_indices_ = np.array([1], dtype=np.int32)
            selector = SimpleNamespace(selected_indices_=np.array([0, 1], dtype=np.int32))
            scaler = SimpleNamespace(scale_=np.ones(2), mean_=np.zeros(2))

            def fit_transform(self, values):
                return values

            def transform_feature_indices(self, indices):
                return indices

        random_result = SimpleNamespace(
            threshold=0.31,
            consensus_params={'max_depth': 7},
            metrics={'cv_strategy': 'random_stratified'},
        )
        loho_result = SimpleNamespace(
            threshold=0.89,
            consensus_params={'max_depth': 99},
            metrics={'cv_strategy': 'grouped_by_site'},
        )
        random_runner = SimpleNamespace(run=lambda *args, **kwargs: random_result)
        loho_runner = SimpleNamespace(run=lambda *args, **kwargs: loho_result)

        with patch.object(training, 'find_patients', return_value=records), \
                patch.object(training, 'build_training_metadata_cache', return_value=({}, {})), \
                patch.object(training, 'process_training_record', side_effect=lambda record, *args: extracted[record]), \
                patch.object(training, 'get_feature_names', return_value=['Age', 'Sex', 'Resp']), \
                patch.object(training, 'get_feature_group_indices', return_value=feature_indices), \
                patch.object(training, 'EnsembleCrossValidator', side_effect=[random_runner, loho_runner]) as validator_class, \
                patch.object(training, 'build_preprocessor', return_value=FakePreprocessor()), \
                patch.object(training, 'get_processed_feature_names', return_value=['Age', 'Sex', 'Resp']), \
                patch.object(training, '_get_combined_model_indices', return_value={}), \
                patch.object(training, '_fit_ensemble', return_value={}) as fit_ensemble, \
                patch.object(training, '_evaluate_and_display_models', return_value={}), \
                patch.object(training, 'export_feature_views', return_value={}), \
                patch.object(training, 'export_selected_features_csv'):
            bundle = training.train_multimodal_ensemble('data', False, 'records.csv', 'exports')

        self.assertEqual(bundle['threshold'], random_result.threshold)
        self.assertEqual(bundle['cv_metrics'], random_result.metrics)
        self.assertNotIn('diagnostic_loho_result', bundle)
        self.assertEqual(
            fit_ensemble.call_args.kwargs['consensus_params'],
            random_result.consensus_params,
        )
        self.assertEqual(validator_class.call_count, 2)
        self.assertFalse(validator_class.call_args_list[0].kwargs['config'].use_site_grouped_cv)
        self.assertTrue(validator_class.call_args_list[1].kwargs['config'].use_site_grouped_cv)

    def test_diagnostic_runner_only_enables_grouped_site_cv(self):
        production_config = CrossValidationConfig(
            search_scoring='age_conditioned_auroc',
            use_site_grouped_cv=False,
            optimize_hyperparameter_search=True,
            outer_random_splits=5,
            random_state=42,
            search_iterations=20,
            fixed_hyperparameters={'max_depth': 3},
        )
        features = np.zeros((6, 4), dtype=np.float32)
        labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        feature_indices = {'all': np.arange(4, dtype=np.int32)}
        site_groups = np.array(['I0002', 'I0002', 'I0006', 'I0006', 'S0001', 'S0001'])

        with patch.object(training, 'EnsembleCrossValidator') as validator_class:
            expected_result = object()
            validator_class.return_value.run.return_value = expected_result
            result = training._run_diagnostic_loho(
                production_config,
                features,
                labels,
                feature_indices,
                feature_indices,
                [1],
                site_groups,
                0,
            )

        diagnostic_config = validator_class.call_args.kwargs['config']
        self.assertFalse(production_config.use_site_grouped_cv)
        self.assertTrue(diagnostic_config.use_site_grouped_cv)
        self.assertEqual(
            {**diagnostic_config.__dict__, 'use_site_grouped_cv': False},
            production_config.__dict__,
        )
        validator_class.return_value.run.assert_called_once()
        run_call = validator_class.return_value.run.call_args
        np.testing.assert_array_equal(run_call.args[0], features)
        np.testing.assert_array_equal(run_call.args[1], labels)
        np.testing.assert_array_equal(run_call.kwargs['site_groups'], site_groups)
        self.assertIs(result, expected_result)

    def test_deployable_outputs_are_fixed_before_diagnostic_loho(self):
        source = inspect.getsource(training.train_multimodal_ensemble)

        random_run = source.index('cv_result = cv_runner.run(')
        threshold_source = source.index('threshold = cv_result.threshold')
        consensus_source = source.index('consensus = cv_result.consensus_params')
        final_fit = source.index('models = _fit_ensemble(')
        diagnostic_run = source.index('diagnostic_loho_result = _run_diagnostic_loho(')

        self.assertLess(random_run, threshold_source)
        self.assertLess(threshold_source, final_fit)
        self.assertLess(consensus_source, final_fit)
        self.assertLess(final_fit, diagnostic_run)
        self.assertEqual(source.count('diagnostic_loho_result'), 1)
        self.assertIn("'threshold': threshold", source)
        self.assertIn("'cv_metrics': cv_metrics", source)
        self.assertIn('consensus_params=consensus', source)
        self.assertFalse(training.USE_SITE_GROUPED_CV)


if __name__ == '__main__':
    unittest.main()
