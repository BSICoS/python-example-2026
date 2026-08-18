import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from src.pipeline.cross_validation import CrossValidationConfig, EnsembleCrossValidator

from src.pipeline.metrics import (
    AgeConditionedAUROCScorer,
    compute_age_conditioned_auroc,
    resolve_search_scoring,
)

from src.pipeline.preprocessing import build_preprocessor

def _read_config_literal(name):
    config_path = Path(__file__).parents[1] / 'src' / 'pipeline' / 'config.py'
    module = ast.parse(config_path.read_text(encoding='utf-8'))
    for statement in module.body:
        if isinstance(statement, ast.Assign):
            targets = [target.id for target in statement.targets if isinstance(target, ast.Name)]
            if name in targets:
                return ast.literal_eval(statement.value)
    raise AssertionError(f'Config value not found: {name}')

class _ProbabilityEstimator:
    def predict_proba(self, features):
        positive = np.asarray(features)[:, 1]
        return np.column_stack([1.0 - positive, positive])


class CrossValidationScoringTests(unittest.TestCase):
    def test_config_selects_age_conditioned_auroc(self):
        self.assertEqual(_read_config_literal('CV_SEARCH_SCORING'), 'age_conditioned_auroc')

    def test_age_conditioned_scorer_uses_age_in_years_after_scaling(self):
        labels = np.array([1, 1, 0, 0], dtype=np.int32)
        ages_years = np.array([20.0, 80.0, 21.0, 81.0])
        predictions = np.array([0.9, 0.7, 0.8, 0.1])
        scaled_ages = (ages_years - 50.0) / 10.0
        features = np.column_stack([scaled_ages, predictions])

        scorer = AgeConditionedAUROCScorer(
            age_feature_index=0,
            age_feature_scale=10.0,
            age_feature_offset=50.0,
        )

        self.assertAlmostEqual(scorer(_ProbabilityEstimator(), features, labels), 1.0)
        self.assertAlmostEqual(roc_auc_score(labels, predictions), 0.75)

    def test_age_conditioned_auroc_counts_ties_as_half(self):
        score = compute_age_conditioned_auroc(
            labels=[1, 0],
            predictions=[0.4, 0.4],
            ages=[50, 51],
        )

        self.assertEqual(score, 0.5)

    def test_cross_validator_passes_configured_scorer_to_search(self):
        config = CrossValidationConfig(
            outer_random_splits=2,
            search_iterations=1,
            search_scoring='age_conditioned_auroc',
            optimize_hyperparameter_search=True,
        )
        runner = EnsembleCrossValidator(
            config=config,
            param_dist={'max_depth': [3]},
            default_threshold=0.5,
            build_preprocessor=build_preprocessor,
            build_search_model=lambda labels: object(),
            fit_ensemble=lambda *args, **kwargs: None,
            predict_probabilities=lambda *args, **kwargs: None,
            search_age_feature_index=0,
            search_age_feature_scale=10.0,
            search_age_feature_offset=50.0,
        )
        features = np.array([[-3.0, 0.1], [3.0, 0.9], [-2.9, 0.2], [3.1, 0.8]])
        labels = np.array([1, 1, 0, 0], dtype=np.int32)

        with patch('src.pipeline.cross_validation.RandomizedSearchCV') as search_class:
            search_class.return_value.best_params_ = {'model__max_depth': 3}
            search_class.return_value.best_score_ = 0.75
            params, score = runner._select_final_params(features, labels)

        estimator = search_class.call_args.kwargs['estimator']
        self.assertIsInstance(estimator, Pipeline)
        self.assertEqual(list(estimator.named_steps), ['preprocessor', 'model'])
        self.assertEqual(search_class.call_args.kwargs['param_distributions'], {'model__max_depth': [3]})
        self.assertEqual(params, {'max_depth': 3})
        self.assertEqual(score, 0.75)
        scorer = search_class.call_args.kwargs['scoring']
        self.assertIsInstance(scorer, AgeConditionedAUROCScorer)
        self.assertEqual(scorer.age_feature_scale, 10.0)
        search_class.return_value.fit.assert_called_once()

    def test_final_search_is_independent_from_oof_evaluation(self):
        features = np.arange(16, dtype=np.float32).reshape(8, 2)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
        feature_indices = {'all': np.array([0, 1], dtype=np.int32)}

        def run_with_final_params(final_params):
            search_calls = []

            def record_search(search_features, search_labels, **kwargs):
                search_calls.append((np.asarray(search_features).copy(), np.asarray(search_labels).copy()))
                if len(search_calls) == 1:
                    return {'max_depth': 1}, 0.61
                if len(search_calls) == 2:
                    return {'max_depth': 2}, 0.62
                return final_params, 0.99

            runner = EnsembleCrossValidator(
                config=CrossValidationConfig(
                    search_scoring='roc_auc',
                    use_site_grouped_cv=False,
                    optimize_hyperparameter_search=True,
                    outer_random_splits=2,
                ),
                param_dist={'max_depth': [1, 2]},
                default_threshold=0.5,
                build_preprocessor=lambda *args: None,
                build_search_model=lambda fold_labels: object(),
                fit_ensemble=lambda values, fold_labels, indices, final_params=None: {
                    'params': dict(final_params),
                },
                predict_probabilities=lambda bundle, values: np.full(
                    len(values), bundle['models']['params']['max_depth'] / 10.0,
                ),
            )
            with patch.object(runner, '_search_hyperparams', side_effect=record_search):
                with redirect_stdout(io.StringIO()):
                    result = runner.run(
                        features,
                        labels,
                        feature_indices,
                        modality_presence_indices=feature_indices,
                    )
            return result, search_calls

        first_result, first_search_calls = run_with_final_params({'max_depth': 3})
        second_result, second_search_calls = run_with_final_params({'max_depth': 9})

        self.assertEqual(first_result.final_params, {'max_depth': 3})
        self.assertEqual(second_result.final_params, {'max_depth': 9})
        self.assertEqual(first_result.final_search_score, 0.99)
        self.assertEqual(first_result.metrics['selected_params_per_fold'][0]['params'], {'max_depth': 1})
        self.assertEqual(first_result.metrics['selected_params_per_fold'][1]['params'], {'max_depth': 2})
        self.assertEqual(first_result.metrics['oof_calibrated_metrics'], second_result.metrics['oof_calibrated_metrics'])
        self.assertEqual(first_result.metrics['fold_metrics'], second_result.metrics['fold_metrics'])
        self.assertTrue(np.array_equal(first_search_calls[-1][0], features))
        self.assertTrue(np.array_equal(first_search_calls[-1][1], labels))
        self.assertTrue(np.array_equal(second_search_calls[-1][0], features))
        self.assertFalse(hasattr(EnsembleCrossValidator, '_consensus_params'))

    def test_standard_auroc_selector_keeps_sklearn_string(self):
        self.assertEqual(resolve_search_scoring('roc_auc'), 'roc_auc')

    def test_unknown_selector_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported CV search scoring selector'):
            resolve_search_scoring('not-a-metric')


if __name__ == '__main__':
    unittest.main()
