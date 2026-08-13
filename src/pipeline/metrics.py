from dataclasses import dataclass

import numpy as np


AGE_CONDITIONED_AUROC_SCORING = 'age_conditioned_auroc'
STANDARD_AUROC_SCORING = 'roc_auc'
DEFAULT_AGE_GAP_YEARS = 2.0


def compute_age_conditioned_auroc(labels, predictions, ages, gap=DEFAULT_AGE_GAP_YEARS):
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    ages = np.asarray(ages, dtype=float).reshape(-1)

    if not (labels.size == predictions.size == ages.size):
        raise ValueError('Labels, predictions, and ages must have the same length.')

    positive_mask = labels == 1
    negative_mask = labels == 0
    if not np.any(positive_mask) or not np.any(negative_mask):
        return 0.0

    positive_predictions = predictions[positive_mask]
    negative_predictions = predictions[negative_mask]
    positive_ages = ages[positive_mask]
    negative_ages = ages[negative_mask]

    eligible_pairs = (
        np.abs(positive_ages[:, np.newaxis] - negative_ages[np.newaxis, :])
        <= float(gap)
    )
    if not np.any(eligible_pairs):
        return 0.0

    pair_scores = (
        (positive_predictions[:, np.newaxis] > negative_predictions[np.newaxis, :]).astype(float)
        + 0.5
        * (positive_predictions[:, np.newaxis] == negative_predictions[np.newaxis, :])
    )
    return float(np.mean(pair_scores[eligible_pairs]))


def _positive_class_probabilities(estimator, features):
    probabilities = np.asarray(estimator.predict_proba(features))
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError('The estimator must return probabilities for both classes.')
    return probabilities[:, 1]


def _feature_column(features, feature_index):
    column = features[:, int(feature_index)]
    if hasattr(column, 'toarray'):
        column = column.toarray()
    return np.asarray(column, dtype=float).reshape(-1)


@dataclass(frozen=True)
class AgeConditionedAUROCScorer:
    age_feature_index: int
    age_feature_scale: float = 1.0
    age_feature_offset: float = 0.0
    age_gap_years: float = DEFAULT_AGE_GAP_YEARS

    def __call__(self, estimator, features, labels):
        transformed_ages = _feature_column(features, self.age_feature_index)
        ages_years = (
            transformed_ages * float(self.age_feature_scale)
            + float(self.age_feature_offset)
        )
        predictions = _positive_class_probabilities(estimator, features)
        return compute_age_conditioned_auroc(
            labels,
            predictions,
            ages_years,
            gap=self.age_gap_years,
        )


def resolve_search_scoring(
    selector,
    *,
    age_feature_index=None,
    age_feature_scale=1.0,
    age_feature_offset=0.0,
):
    if selector == STANDARD_AUROC_SCORING:
        return STANDARD_AUROC_SCORING

    if selector == AGE_CONDITIONED_AUROC_SCORING:
        if age_feature_index is None:
            raise ValueError(
                'age_feature_index is required for age-conditioned AUROC scoring.'
            )
        return AgeConditionedAUROCScorer(
            age_feature_index=int(age_feature_index),
            age_feature_scale=float(age_feature_scale),
            age_feature_offset=float(age_feature_offset),
        )

    raise ValueError(
        f'Unsupported CV search scoring selector: {selector!r}. '
        f'Expected {STANDARD_AUROC_SCORING!r} or {AGE_CONDITIONED_AUROC_SCORING!r}.'
    )
