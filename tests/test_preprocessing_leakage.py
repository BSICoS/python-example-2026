from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone

from src.pipeline.cross_validation import CrossValidationConfig, EnsembleCrossValidator, FoldSplit
from src.pipeline.preprocessing import CorrelationAwarePreprocessor
from src.pipeline.training import (
    _fit_ensemble,
    _get_ecg_eeg_search_data,
    _select_ensemble_model_name,
)


class _RecordingPreprocessor(CorrelationAwarePreprocessor):
    fitted_sample_counts = []

    def fit_transform(self, features, y=None, **fit_params):
        type(self).fitted_sample_counts.append(len(features))
        return super().fit_transform(features, y=y, **fit_params)


def _build_recording_preprocessor(num_samples, categorical_indices=None):
    return _RecordingPreprocessor(
        n_neighbors=min(5, max(1, num_samples - 1)),
        categorical_indices=categorical_indices,
        correlation_threshold=0.99,
    )


def _predict_from_fold_preprocessor(model_bundle, raw_features):
    preprocessor = model_bundle['preprocessor']
    processed = preprocessor.transform(raw_features)
    return 1.0 / (1.0 + np.exp(-processed[:, 0]))


class _FeatureProbabilityModel:
    def predict_proba(self, features):
        positive = np.asarray(features, dtype=np.float32)[:, 1]
        return np.column_stack([1.0 - positive, positive])


class PreprocessingLeakageTests(unittest.TestCase):
    def setUp(self):
        _RecordingPreprocessor.fitted_sample_counts = []

    def test_preprocessor_is_cloneable_for_sklearn_pipeline(self):
        original = _build_recording_preprocessor(8, [1])
        cloned = clone(original)

        self.assertIsInstance(cloned, CorrelationAwarePreprocessor)
        self.assertEqual(cloned.categorical_indices, [1])
        self.assertEqual(cloned.correlation_threshold, 0.99)

    def test_outer_cv_fits_preprocessing_only_on_each_training_fold(self):
        features = np.array([
            [40.0, 0.1],
            [41.0, 0.2],
            [50.0, 0.3],
            [51.0, 0.4],
            [60.0, 0.5],
            [61.0, 0.6],
            [70.0, 0.7],
            [71.0, 0.8],
        ], dtype=np.float32)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
        feature_indices = {
            'all': np.array([0, 1], dtype=np.int32),
            'demographics': np.array([0], dtype=np.int32),
            'resp': np.array([1], dtype=np.int32),
            'eeg': np.array([], dtype=np.int32),
            'ecg': np.array([], dtype=np.int32),
        }
        runner = EnsembleCrossValidator(
            config=CrossValidationConfig(
                search_scoring='roc_auc',
                optimize_hyperparameter_search=False,
                outer_random_splits=2,
            ),
            param_dist={},
            default_threshold=0.5,
            build_preprocessor=_build_recording_preprocessor,
            build_search_model=lambda fold_labels: object(),
            fit_ensemble=lambda *args, **kwargs: {},
            predict_probabilities=_predict_from_fold_preprocessor,
            search_age_feature_index=0,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            result = runner.evaluate_random_nested_cv(
                features,
                labels,
                feature_indices,
                modality_presence_indices=feature_indices,
            )

        self.assertFalse(result.metrics['skipped'])
        self.assertEqual(_RecordingPreprocessor.fitted_sample_counts, [4, 4])
        for fold_metrics in result.metrics['fold_metrics']:
            self.assertIn('age_conditioned_auroc', fold_metrics)
        self.assertIn('age_conditioned_auroc', result.metrics['fold_metric_summary'])
        self.assertIn(
            'age_conditioned_auroc',
            result.metrics['oof_calibrated_metrics'],
        )
        self.assertIn('Age-conditioned AUROC=', output.getvalue())

    def test_outer_fold_uses_only_its_own_inner_search_parameters(self):
        features = np.arange(16, dtype=np.float32).reshape(8, 2)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
        feature_indices = {'all': np.array([0, 1], dtype=np.int32)}
        split_plan = [
            FoldSplit(1, np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7]), 'first'),
            FoldSplit(2, np.array([4, 5, 6, 7]), np.array([0, 1, 2, 3]), 'second'),
        ]
        fitted_params = []

        def fit_ensemble(features, labels, indices, final_params=None):
            fitted_params.append(dict(final_params or {}))
            return {}

        runner = EnsembleCrossValidator(
            config=CrossValidationConfig(search_scoring='roc_auc'),
            param_dist={},
            default_threshold=0.5,
            build_preprocessor=lambda *args: None,
            build_search_model=lambda fold_labels: object(),
            fit_ensemble=fit_ensemble,
            predict_probabilities=lambda bundle, fold_features: np.full(len(fold_features), 0.5),
        )

        with redirect_stdout(io.StringIO()):
            runner._evaluate_with_fold_params(
                split_plan,
                features,
                labels,
                feature_indices,
                modality_presence_indices=feature_indices,
                selected_params_per_fold=[
                    {'fold': 1, 'params': {'max_depth': 1}},
                    {'fold': 2, 'params': {'max_depth': 2}},
                ],
            )

        self.assertEqual(fitted_params, [{'max_depth': 1}, {'max_depth': 2}])

    def test_cross_validation_saves_metrics_for_each_selected_model_route(self):
        features = np.array([
            [20.0, 0.1], [21.0, 0.9], [22.0, 0.2], [23.0, 0.8],
            [24.0, 0.3], [25.0, 0.7], [26.0, 0.4], [27.0, 0.6],
        ], dtype=np.float32)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
        feature_indices = {
            'all': np.array([0, 1], dtype=np.int32),
            'demographics': np.array([0], dtype=np.int32),
            'ecg': np.array([1], dtype=np.int32),
            'eeg': np.array([1], dtype=np.int32),
        }

        runner = EnsembleCrossValidator(
            config=CrossValidationConfig(
                search_scoring='roc_auc',
                optimize_hyperparameter_search=False,
                outer_random_splits=2,
            ),
            param_dist={},
            default_threshold=0.5,
            build_preprocessor=lambda *args: None,
            build_search_model=lambda labels: object(),
            fit_ensemble=lambda *args, **kwargs: {
                'ecg': _FeatureProbabilityModel(),
                'eeg': _FeatureProbabilityModel(),
            },
            predict_probabilities=lambda bundle, values: values[:, 1],
            select_model_names=lambda bundle, values: np.where(values[:, 0] < 24, 'ecg', 'eeg'),
            search_age_feature_index=0,
        )

        with redirect_stdout(io.StringIO()):
            result = runner.evaluate_random_nested_cv(
                features,
                labels,
                feature_indices,
                modality_presence_indices=feature_indices,
            )

        self.assertEqual(result.metrics['model_route_counts'], {'ecg': 4, 'eeg': 4})

        eligible_metrics = result.metrics['model_eligible_oof_metrics']
        self.assertEqual(eligible_metrics['ecg']['n_records'], 8)
        self.assertEqual(eligible_metrics['eeg']['n_records'], 8)

    def test_each_route_trains_only_with_its_required_modalities(self):
        features = np.array([
            [20.0, 1.0, np.nan, np.nan],
            [21.0, 1.0, 1.0, np.nan],
            [22.0, 1.0, 1.0, 1.0],
            [23.0, 1.0, 1.0, 1.0],
        ], dtype=np.float32)
        labels = np.array([0, 1, 0, 1], dtype=np.int32)
        feature_indices = {
            'demographics': np.array([0], dtype=np.int32),
            'ecg': np.array([1], dtype=np.int32),
            'eeg': np.array([2], dtype=np.int32),
            'resp': np.array([3], dtype=np.int32),
            'all': np.arange(4, dtype=np.int32),
        }
        fitted_labels = []
        fitted_params = []

        final_params = {'max_depth': 6, 'n_estimators': 100}

        def record_fit(features, route_labels, final_params=None):
            fitted_labels.append(np.asarray(route_labels, dtype=np.int32))
            fitted_params.append(dict(final_params or {}))
            return _FeatureProbabilityModel()

        with patch('src.pipeline.training._fit_model', side_effect=record_fit):
            models = _fit_ensemble(
                features,
                labels,
                feature_indices,
                modality_presence_indices={
                    name: feature_indices[name]
                    for name in ('ecg', 'eeg', 'resp')
                },
                final_params=final_params,
            )

        self.assertEqual(models['ecg']['n_train'], 4)
        self.assertEqual(models['eeg']['n_train'], 3)
        self.assertEqual(models['ecg_eeg']['n_train'], 3)
        self.assertEqual(tuple(models), ('ecg', 'eeg', 'ecg_eeg'))
        self.assertEqual(len(fitted_labels), len(models))
        self.assertEqual(fitted_params, [final_params] * len(models))

    def test_model_routing_prioritizes_ecg_eeg_over_respiration_routes(self):
        models = {
            name: object()
            for name in ('ecg', 'eeg', 'resp', 'ecg_eeg', 'ecg_resp', 'eeg_resp', 'all')
        }
        modality_presence_indices = {
            'ecg': np.array([0], dtype=np.int32),
            'eeg': np.array([1], dtype=np.int32),
            'resp': np.array([2], dtype=np.int32),
        }

        self.assertEqual(
            _select_ensemble_model_name(
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
                models,
                modality_presence_indices,
            ),
            'ecg_eeg',
        )
        self.assertEqual(
            _select_ensemble_model_name(
                np.array([np.nan, 1.0, 1.0], dtype=np.float32),
                models,
                modality_presence_indices,
            ),
            'eeg',
        )
        self.assertEqual(
            _select_ensemble_model_name(
                np.array([1.0, np.nan, 1.0], dtype=np.float32),
                models,
                modality_presence_indices,
            ),
            'ecg',
        )

    def test_ecg_eeg_search_data_excludes_ineligible_samples_and_resp_features(self):
        features = np.array([
            [20.0, 0.0, 10.0, 20.0, 30.0],
            [21.0, 1.0, np.nan, 21.0, 31.0],
            [22.0, 0.0, 12.0, np.nan, 32.0],
            [23.0, 1.0, 13.0, 23.0, 33.0],
        ], dtype=np.float32)
        labels = np.array([0, 1, 0, 1], dtype=np.int32)
        feature_indices = {
            'demographics': np.array([0, 1], dtype=np.int32),
            'resp': np.array([2], dtype=np.int32),
            'eeg': np.array([3], dtype=np.int32),
            'ecg': np.array([4], dtype=np.int32),
        }

        search_data = _get_ecg_eeg_search_data(
            features,
            labels,
            feature_indices,
            modality_presence_indices={
                'resp': feature_indices['resp'],
                'eeg': feature_indices['eeg'],
                'ecg': feature_indices['ecg'],
            },
            categorical_indices=[1],
            site_groups=np.array(['A', 'B', 'C', 'D']),
        )

        self.assertEqual(search_data['route_name'], 'ecg_eeg')
        self.assertTrue(np.array_equal(search_data['raw_indices'], [0, 1, 3, 4]))
        self.assertTrue(np.array_equal(search_data['features'], features[[0, 1, 3]][:, [0, 1, 3, 4]]))
        self.assertTrue(np.array_equal(search_data['labels'], labels[[0, 1, 3]]))
        self.assertEqual(search_data['categorical_indices'], [1])
        self.assertTrue(np.array_equal(search_data['site_groups'], ['A', 'B', 'D']))


if __name__ == '__main__':
    unittest.main()
