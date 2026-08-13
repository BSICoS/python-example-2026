from contextlib import redirect_stdout
import io
import unittest

import numpy as np
from sklearn.base import clone

from src.pipeline.cross_validation import CrossValidationConfig, EnsembleCrossValidator
from src.pipeline.preprocessing import CorrelationAwarePreprocessor


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
                use_site_grouped_cv=False,
                optimize_hyperparameter_search=False,
                outer_random_splits=2,
            ),
            param_dist={},
            default_threshold=0.5,
            build_preprocessor=_build_recording_preprocessor,
            build_search_model=lambda fold_labels: object(),
            fit_ensemble=lambda *args, **kwargs: {},
            predict_probabilities=_predict_from_fold_preprocessor,
        )

        with redirect_stdout(io.StringIO()):
            result = runner.run(
                features,
                labels,
                feature_indices,
                modality_presence_indices=feature_indices,
            )

        self.assertFalse(result.metrics['skipped'])
        self.assertEqual(_RecordingPreprocessor.fitted_sample_counts, [4, 4])


if __name__ == '__main__':
    unittest.main()
