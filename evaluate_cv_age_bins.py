#!/usr/bin/env python

# Analyzes cross-validation out-of-fold predictions (exported by src/pipeline/training.py as
# training_cv_oof_predictions.csv) broken down by age band, to check whether the model
# generalizes worse for some age ranges than others under random-split and leave-one-hospital-out CV.
#
# Edit the CONFIG block below and just run the script (no CLI arguments needed):
#   python evaluate_cv_age_bins.py

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

# --------------------------------------------------------------------------------------
# CONFIG - edit these and run the script.
# --------------------------------------------------------------------------------------
INPUT_CSV = 'feature_exports/training_cv_oof_predictions.csv'
OUTPUT_CSV = None  # e.g. 'age_band_report.csv'; set to None to only print the report.
AGE_BIN_EDGES = [50, 60, 70, 80, 89]  # decade bands: 50-59, 60-69, 70-79, 80-89.
CV_STRATEGY = None  # 'random_stratified', 'grouped_by_site', or None for both.
BY_SITE = False  # also split each age band by SiteID (only meaningful with enough records per hospital).
# --------------------------------------------------------------------------------------


def build_age_bands(edges):
    labels = [f'{edges[i]}-{edges[i + 1] - 1}' for i in range(len(edges) - 1)]
    labels[-1] = f'{edges[-2]}-{edges[-1]}'
    return labels


def summarize_group(group, threshold_column='calibrated_threshold'):
    labels = group['label'].to_numpy(dtype=np.int32)
    probabilities = group['oof_probability'].to_numpy(dtype=np.float32)
    threshold = float(group[threshold_column].iloc[0])
    predictions = (probabilities >= threshold).astype(np.int32)

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    auroc = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) > 1 else float('nan')
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else float('nan')

    return {
        'n': int(len(labels)),
        'n_positive': int(np.sum(labels == 1)),
        'threshold': threshold,
        'auroc': auroc,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def build_report(predictions, bin_edges, by_site):
    band_labels = build_age_bands(bin_edges)
    predictions = predictions.copy()
    predictions['age_band'] = pd.cut(
        predictions['age'],
        bins=bin_edges,
        labels=band_labels,
        include_lowest=True,
        right=True,
    )

    group_columns = ['cv_strategy', 'age_band'] + (['site_id'] if by_site else [])
    rows = []
    for group_key, group in predictions.groupby(group_columns, observed=True):
        if len(group) == 0:
            continue
        row = dict(zip(group_columns, group_key if isinstance(group_key, tuple) else (group_key,)))
        row.update(summarize_group(group))
        rows.append(row)

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(group_columns).reset_index(drop=True)
    return report


def run():
    predictions = pd.read_csv(INPUT_CSV)
    if CV_STRATEGY:
        predictions = predictions[predictions['cv_strategy'] == CV_STRATEGY]

    report = build_report(predictions, AGE_BIN_EDGES, BY_SITE)

    with pd.option_context('display.max_rows', None, 'display.width', 160):
        print(report.to_string(index=False))

    if OUTPUT_CSV:
        report.to_csv(OUTPUT_CSV, index=False)
        print(f'\nSaved report to {OUTPUT_CSV}')


if __name__ == '__main__':
    run()
