import ast
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
            params, score = runner._search_hyperparams(features, labels)

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

    def test_standard_auroc_selector_keeps_sklearn_string(self):
        self.assertEqual(resolve_search_scoring('roc_auc'), 'roc_auc')

    def test_unknown_selector_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported CV search scoring selector'):
            resolve_search_scoring('not-a-metric')

    def test_consensus_selects_most_frequent_complete_configuration(self):
        runner = self._build_consensus_runner()
        configuration_a = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.03}
        selected = [
            {'fold': 1, 'params': configuration_a, 'score': 0.71},
            {'fold': 2, 'params': configuration_a, 'score': 0.73},
            {'fold': 3, 'params': {'n_estimators': 800, 'max_depth': 5, 'learning_rate': 0.30}, 'score': 0.90},
            {'fold': 4, 'params': {'n_estimators': 1000, 'max_depth': 10, 'learning_rate': 0.01}, 'score': 0.80},
            {'fold': 5, 'params': {'n_estimators': 1000, 'max_depth': 3, 'learning_rate': 0.01}, 'score': 0.85},
        ]

        consensus, frequency, folds, mean_score = runner._consensus_params(selected)

        self.assertEqual(consensus, configuration_a)
        self.assertEqual(frequency, 2)
        self.assertEqual(folds, [1, 2])
        self.assertAlmostEqual(mean_score, 0.72)
        self.assertIn(consensus, [item['params'] for item in selected])

    def test_consensus_uses_best_score_when_all_configurations_differ(self):
        runner = self._build_consensus_runner()
        selected = [
            {'fold': 1, 'params': {'max_depth': 3}, 'score': 0.70},
            {'fold': 2, 'params': {'max_depth': 5}, 'score': 0.92},
            {'fold': 3, 'params': {'max_depth': 7}, 'score': 0.80},
        ]

        consensus, _, _, _ = runner._consensus_params(selected)

        self.assertEqual(consensus, {'max_depth': 5})
        self.assertIn(consensus, [item['params'] for item in selected])

    def test_consensus_uses_mean_score_to_break_frequency_tie(self):
        runner = self._build_consensus_runner()
        selected = [
            {'fold': 1, 'params': {'max_depth': 3}, 'score': 0.70},
            {'fold': 2, 'params': {'max_depth': 5}, 'score': 0.85},
            {'fold': 3, 'params': {'max_depth': 3}, 'score': 0.80},
            {'fold': 4, 'params': {'max_depth': 5}, 'score': 0.95},
        ]

        consensus, frequency, folds, mean_score = runner._consensus_params(selected)

        self.assertEqual(consensus, {'max_depth': 5})
        self.assertEqual(frequency, 2)
        self.assertEqual(folds, [2, 4])
        self.assertAlmostEqual(mean_score, 0.90)

    def test_consensus_uses_first_fold_to_break_score_tie_deterministically(self):
        runner = self._build_consensus_runner()
        selected = [
            {'fold': 1, 'params': {'max_depth': 3}, 'score': 0.80},
            {'fold': 2, 'params': {'max_depth': 5}, 'score': 0.80},
        ]

        consensus, _, folds, _ = runner._consensus_params(selected)

        self.assertEqual(consensus, {'max_depth': 3})
        self.assertEqual(folds, [1])

    def _build_consensus_runner(self):
        return EnsembleCrossValidator(
            config=CrossValidationConfig(search_scoring='roc_auc'),
            param_dist={},
            default_threshold=0.5,
            build_preprocessor=lambda *args: None,
            build_search_model=lambda labels: object(),
            fit_ensemble=lambda *args, **kwargs: {},
            predict_probabilities=lambda *args, **kwargs: None,
        )


if __name__ == '__main__':
    unittest.main()
