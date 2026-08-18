from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .metrics import compute_age_conditioned_auroc, resolve_search_scoring


def normalize_site_group(site_id):
    site_text = str(site_id).strip().upper()
    return site_text[:5] if site_text else 'UNKNOWN'


@dataclass(frozen=True)
class CrossValidationConfig:
    search_scoring: str
    use_site_grouped_cv: bool = True
    optimize_hyperparameter_search: bool = False
    outer_random_splits: int = 5
    random_state: int = 42
    search_iterations: int = 50
    fixed_hyperparameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossValidationResult:
    threshold: float
    consensus_params: Optional[dict[str, Any]]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class FoldSplit:
    fold_index: int
    train_idx: np.ndarray
    validation_idx: np.ndarray
    label: str


class EnsembleCrossValidator:
    def __init__(
        self,
        config: CrossValidationConfig,
        param_dist,
        default_threshold,
        build_preprocessor: Callable[..., Any],
        build_search_model: Callable[..., Any],
        fit_ensemble: Callable[..., Any],
        predict_probabilities: Callable[..., Any],
        select_model_names: Optional[Callable[..., Any]] = None,
        predict_model_probabilities: Optional[Callable[..., Any]] = None,
        fit_ensemble_handles_preprocessing: bool = False,
        search_age_feature_index: Optional[int] = None,
        search_age_feature_scale: float = 1.0,
        search_age_feature_offset: float = 0.0,
    ):
        self.config = config
        self.param_dist = param_dist
        self.default_threshold = default_threshold
        self.build_preprocessor = build_preprocessor
        self.build_search_model = build_search_model
        self.fit_ensemble = fit_ensemble
        self.predict_probabilities = predict_probabilities
        self.select_model_names = select_model_names
        self.predict_model_probabilities = predict_model_probabilities
        self.fit_ensemble_handles_preprocessing = fit_ensemble_handles_preprocessing
        self.search_age_feature_index = search_age_feature_index
        self.search_age_feature_scale = float(search_age_feature_scale)
        self.search_age_feature_offset = float(search_age_feature_offset)

    def run(
        self,
        features,
        labels,
        feature_indices,
        modality_presence_indices,
        categorical_indices=None,
        site_groups=None,
    ):
        labels = np.asarray(labels, dtype=np.int32)
        if self.config.use_site_grouped_cv:
            return self._run_grouped_cv(
                features,
                labels,
                site_groups,
                feature_indices,
                modality_presence_indices,
                categorical_indices=categorical_indices,
            )

        return self._run_random_cv(
            features,
            labels,
            feature_indices,
            modality_presence_indices,
            categorical_indices=categorical_indices,
        )

    def _run_grouped_cv(
        self,
        features,
        labels,
        site_groups,
        feature_indices,
        modality_presence_indices,
        categorical_indices=None,
    ):
        site_groups = np.asarray(site_groups)
        unique_sites = np.unique(site_groups)
        if unique_sites.size < 2:
            return CrossValidationResult(
                threshold=self.default_threshold,
                consensus_params=None,
                metrics={
                    'skipped': True,
                    'cv_strategy': 'grouped_by_site',
                    'reason': 'Not enough hospitals to run grouped cross-validation.',
                    'site_groups': unique_sites.tolist(),
                },
            )

        classes = np.unique(labels)
        if len(classes) != 2:
            return CrossValidationResult(
                threshold=self.default_threshold,
                consensus_params=None,
                metrics={
                    'skipped': True,
                    'cv_strategy': 'grouped_by_site',
                    'reason': 'Need both classes to run grouped cross-validation.',
                    'site_groups': unique_sites.tolist(),
                },
            )

        group_cv = LeaveOneGroupOut()
        split_plan = []
        for fold_idx, (train_idx, val_idx) in enumerate(group_cv.split(features, labels, groups=site_groups), start=1):
            held_out_site = normalize_site_group(site_groups[val_idx][0])
            split_plan.append(FoldSplit(
                fold_index=fold_idx,
                train_idx=train_idx,
                validation_idx=val_idx,
                label=held_out_site,
            ))

        consensus, selected_params_per_fold = self._select_consensus_params(
            split_plan,
            features,
            labels,
            feature_indices,
            categorical_indices=categorical_indices,
            site_groups=site_groups,
            label_prefix='held-out hospital',
        )
        fold_metrics, oof_probabilities, routed_model_oof = self._evaluate_with_fold_params(
            split_plan,
            features,
            labels,
            feature_indices,
            modality_presence_indices,
            categorical_indices=categorical_indices,
            selected_params_per_fold=selected_params_per_fold,
            label_prefix='held-out hospital',
            extra_metric_fields=lambda split: {'held_out_site': split.label},
        )

        return self._finalize_result(
            consensus_params=consensus,
            labels=labels,
            oof_probabilities=oof_probabilities,
            routed_model_oof=routed_model_oof,
            ages=self._get_ages_in_years(features),
            best_params_per_fold=selected_params_per_fold,
            fold_metrics=fold_metrics,
            metadata={
                'cv_strategy': 'grouped_by_site',
                'n_splits': int(len(split_plan)),
                'site_groups': unique_sites.tolist(),
            },
        )

    def _run_random_cv(
        self,
        features,
        labels,
        feature_indices,
        modality_presence_indices,
        categorical_indices=None,
    ):
        outer_cv, n_splits = self._build_stratified_splitter(labels, self.config.outer_random_splits)
        if outer_cv is None:
            return CrossValidationResult(
                threshold=self.default_threshold,
                consensus_params=None,
                metrics={
                    'skipped': True,
                    'cv_strategy': 'random_stratified',
                    'reason': 'Not enough samples per class to run random stratified cross-validation.',
                    'requested_n_splits': int(self.config.outer_random_splits),
                },
            )

        split_plan = [
            FoldSplit(
                fold_index=fold_idx,
                train_idx=train_idx,
                validation_idx=val_idx,
                label='random stratified split',
            )
            for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(features, labels), start=1)
        ]
        consensus, selected_params_per_fold = self._select_consensus_params(
            split_plan,
            features,
            labels,
            feature_indices,
            categorical_indices=categorical_indices,
            site_groups=None,
            label_prefix='random split',
        )
        fold_metrics, oof_probabilities, routed_model_oof = self._evaluate_with_fold_params(
            split_plan,
            features,
            labels,
            feature_indices,
            modality_presence_indices,
            categorical_indices=categorical_indices,
            selected_params_per_fold=selected_params_per_fold,
            label_prefix='random split',
        )

        return self._finalize_result(
            consensus_params=consensus,
            labels=labels,
            oof_probabilities=oof_probabilities,
            routed_model_oof=routed_model_oof,
            ages=self._get_ages_in_years(features),
            best_params_per_fold=selected_params_per_fold,
            fold_metrics=fold_metrics,
            metadata={
                'cv_strategy': 'random_stratified',
                'n_splits': int(n_splits),
            },
        )

    def _select_consensus_params(
        self,
        split_plan,
        features,
        labels,
        feature_indices,
        categorical_indices=None,
        site_groups=None,
        label_prefix='fold',
    ):
        if not self.config.optimize_hyperparameter_search:
            fixed_params = dict(self.config.fixed_hyperparameters)
            print(f"  Hyperparameter search disabled. Using fixed parameters: {fixed_params}")
            return fixed_params, [
                {
                    'fold': int(split.fold_index),
                    'label': split.label,
                    'params': fixed_params,
                    'source': 'fixed_defaults',
                }
                for split in split_plan
            ]

        selected_params_per_fold = []

        for split in split_plan:
            print(f"  Search fold {split.fold_index}/{len(split_plan)} - {label_prefix} {split.label}")
            X_train = features[split.train_idx]
            y_train = labels[split.train_idx]
            search_site_groups = None if site_groups is None else site_groups[split.train_idx]


            fold_best_params, fold_best_score = self._search_hyperparams(
                X_train,
                y_train,
                site_groups=search_site_groups,
                categorical_indices=categorical_indices,
            )
            print(f"    Best params: {fold_best_params} (inner-CV score: {fold_best_score:.3f})")
            selected_params_per_fold.append({
                'fold': int(split.fold_index),
                'label': split.label,
                'params': fold_best_params,
                'score': fold_best_score,
                'source': 'search',
            })

        consensus, frequency, selected_folds, mean_score = self._consensus_params(selected_params_per_fold)
        print('  Final hyperparameter selection:')
        print(f"    Configuration frequency: {frequency}/{len(selected_params_per_fold)} folds")
        print(f"    Selected from folds: {selected_folds}")
        print(f"    Mean inner-CV score: {mean_score:.3f}")
        print(f"    Consensus hyperparameters: {consensus}")
        return consensus, selected_params_per_fold
    
    def _evaluate_with_fold_params(
        self,
        split_plan,
        features,
        labels,
        feature_indices,
        modality_presence_indices,
        categorical_indices=None,
        selected_params_per_fold=None,
        label_prefix='fold',
        extra_metric_fields=None,
    ):
        params_by_fold = {
            int(item['fold']): dict(item['params'])
            for item in (selected_params_per_fold or [])
        }
        missing_folds = [
            int(split.fold_index)
            for split in split_plan
            if int(split.fold_index) not in params_by_fold
        ]
        if missing_folds:
            raise ValueError(f'Missing selected hyperparameters for folds: {missing_folds}')

        oof_probabilities = np.zeros(len(labels), dtype=np.float32)
        fold_metrics = []
        routed_model_oof = {
            'model_names': np.full(len(labels), None, dtype=object),
            'probabilities': {},
        }

        for split in split_plan:
            print(f"  Eval fold {split.fold_index}/{len(split_plan)} - {label_prefix} {split.label}")
            X_train, X_val = features[split.train_idx], features[split.validation_idx]
            y_train, y_val = labels[split.train_idx], labels[split.validation_idx]

            fold_preprocessor = None
            if self.fit_ensemble_handles_preprocessing:
                X_train_proc = np.asarray(X_train, dtype=np.float32)
                remapped_feature_indices = feature_indices
            else:
                fold_preprocessor = self.build_preprocessor(len(y_train), categorical_indices)

            if fold_preprocessor is not None:
                # Flujo Original dinámico por fold
                X_train_proc = np.asarray(fold_preprocessor.fit_transform(X_train), dtype=np.float32)
                remapped_feature_indices = fold_preprocessor.transform_feature_indices(feature_indices)
                print(
                    f"    Correlation selector kept {X_train_proc.shape[1]}/{X_train.shape[1]} features"
                )
            elif not self.fit_ensemble_handles_preprocessing:
                # Flujo con Variables fijas pre-calculadas afuera
                X_train_proc = np.asarray(X_train, dtype=np.float32)
                remapped_feature_indices = feature_indices # Ya vienen mapeadas de afuera
                print(f"    Variables fijas detectadas. Usando {X_train_proc.shape[1]} características fijas.")

            fit_kwargs = {
                'consensus_params': params_by_fold[int(split.fold_index)],
            }
            if self.fit_ensemble_handles_preprocessing:
                fit_kwargs.update({
                    'modality_presence_indices': modality_presence_indices,
                    'categorical_indices': categorical_indices,
                })
            fold_models = self.fit_ensemble(
                X_train_proc,
                y_train,
                remapped_feature_indices,
                **fit_kwargs,
            )
            fold_bundle = {
                'models': fold_models,
                'feature_indices': remapped_feature_indices,
                'modality_presence_indices': modality_presence_indices,
                'preprocessor': fold_preprocessor, # Pasará como None, ideal para evitar re-transformar en predicciones internas del fold
                'threshold': self.default_threshold,
            }

            fold_probabilities = self.predict_probabilities(fold_bundle, X_val)
            oof_probabilities[split.validation_idx] = fold_probabilities

            if self.select_model_names is not None:
                model_names = np.asarray(
                    self.select_model_names(fold_bundle, X_val),
                    dtype=object,
                ).reshape(-1)
                if model_names.size != len(X_val):
                    raise ValueError(
                        'Model selector returned a different number of names than validation records.'
                    )

                routed_model_oof['model_names'][split.validation_idx] = model_names
                if self.predict_model_probabilities is not None:
                    probabilities_by_model = self.predict_model_probabilities(fold_bundle, X_val)
                else:
                    X_val_proc = (
                        np.asarray(fold_preprocessor.transform(X_val), dtype=np.float32)
                        if fold_preprocessor is not None
                        else np.asarray(X_val, dtype=np.float32)
                    )
                    probabilities_by_model = {}
                    for model_name, model in fold_models.items():
                        model_indices = self._get_model_feature_indices(
                            model_name,
                            remapped_feature_indices,
                        )
                        if model_indices.size == 0:
                            continue
                        probabilities_by_model[model_name] = model.predict_proba(
                            X_val_proc[:, model_indices]
                        )[:, 1]

                for model_name, model_probabilities in probabilities_by_model.items():
                    model_probabilities = np.asarray(model_probabilities, dtype=np.float32)
                    if model_name not in routed_model_oof['probabilities']:
                        routed_model_oof['probabilities'][model_name] = np.full(
                            len(labels),
                            np.nan,
                            dtype=np.float32,
                        )
                    routed_model_oof['probabilities'][model_name][split.validation_idx] = model_probabilities

            fold_metric_row = self._compute_evaluation_metrics(
                y_val,
                fold_probabilities,
                threshold=self.default_threshold,
                ages=self._get_ages_in_years(X_val),
            )
            extra_fields = {} if extra_metric_fields is None else extra_metric_fields(split)
            fold_metric_row = {
                **fold_metric_row,
                'fold': int(split.fold_index),
                'train_size': int(len(split.train_idx)),
                'validation_size': int(len(split.validation_idx)),
                **extra_fields,
            }
            fold_metrics.append(fold_metric_row)
            self._print_metrics(f"    Fold {split.fold_index} metrics:", fold_metric_row)

        return fold_metrics, oof_probabilities, routed_model_oof
    
    def _finalize_result(self, consensus_params, labels, ages, oof_probabilities, routed_model_oof, best_params_per_fold, fold_metrics, metadata):
        consensus = dict(consensus_params or {})

        fold_metric_summary = self._summarize_metrics(fold_metrics)
        self._print_metric_summary("  Mean fold metrics:", fold_metric_summary)

        oof_default_metrics = self._compute_evaluation_metrics(
            labels,
            oof_probabilities,
            threshold=self.default_threshold,
            ages=ages,
        )
        self._print_metrics("  OOF metrics before calibration:", oof_default_metrics)

        threshold = self._best_threshold(oof_probabilities, labels)
        oof_calibrated_metrics = self._compute_evaluation_metrics(
            labels,
            oof_probabilities,
            threshold=threshold,
            ages=ages,
        )
        self._print_metrics("  OOF metrics after calibration:", oof_calibrated_metrics)

        model_route_counts = self._count_selected_model_routes(routed_model_oof)
        if model_route_counts:
            print(f"  Selected model routes: {model_route_counts}")

        model_eligible_oof_metrics = self._compute_model_eligible_oof_metrics(
            labels,
            ages,
            routed_model_oof,
            threshold,
        )
        for model_name, model_metrics in model_eligible_oof_metrics.items():
            self._print_metrics(
                f"  Model OOF metrics ({model_name}, n={model_metrics['n_records']}):",
                model_metrics,
            )

        metrics = {
            'skipped': False,
            **metadata,
            'hyperparameter_optimization_enabled': bool(self.config.optimize_hyperparameter_search),
            'hyperparameter_search_scoring': self.config.search_scoring,
            'fixed_hyperparameters': dict(self.config.fixed_hyperparameters),
            'selected_params_per_fold': best_params_per_fold,
            'fold_metrics': fold_metrics,
            'fold_metric_summary': fold_metric_summary,
            'oof_default_threshold_metrics': oof_default_metrics,
            'oof_calibrated_metrics': oof_calibrated_metrics,
            'model_route_counts': model_route_counts,
            'model_eligible_oof_metrics': model_eligible_oof_metrics,
        }
        return CrossValidationResult(
            threshold=threshold,
            consensus_params=consensus,
            metrics=metrics,
        )

    def _get_model_feature_indices(self, model_name, feature_indices):
        if model_name == 'all':
            return np.asarray(feature_indices.get('all', []), dtype=np.int32)

        indices = set(np.asarray(feature_indices.get('demographics', []), dtype=np.int32).tolist())
        for modality_name in model_name.split('_'):
            indices.update(
                np.asarray(feature_indices.get(modality_name, []), dtype=np.int32).tolist()
            )
        return np.asarray(sorted(indices), dtype=np.int32)

    def _count_selected_model_routes(self, routed_model_oof):
        model_names = np.asarray(routed_model_oof['model_names'], dtype=object)
        selected_model_names = [
            str(model_name)
            for model_name in model_names
            if model_name is not None
        ]
        return {
            model_name: selected_model_names.count(model_name)
            for model_name in sorted(set(selected_model_names))
        }

    def _compute_model_eligible_oof_metrics(self, labels, ages, routed_model_oof, threshold):
        metrics_by_model = {}

        for model_name, probabilities in routed_model_oof['probabilities'].items():
            probabilities = np.asarray(probabilities, dtype=np.float32)
            eligible_mask = np.isfinite(probabilities)
            eligible_labels = labels[eligible_mask]
            if eligible_labels.size == 0:
                continue

            eligible_ages = None if ages is None else ages[eligible_mask]
            metrics = self._compute_evaluation_metrics(
                eligible_labels,
                probabilities[eligible_mask],
                threshold=threshold,
                ages=eligible_ages,
            )
            metrics.update({
                'n_records': int(eligible_labels.size),
                'n_positive': int(np.sum(eligible_labels == 1)),
                'n_negative': int(np.sum(eligible_labels == 0)),
            })
            metrics_by_model[model_name] = metrics

        return metrics_by_model

    def _search_hyperparams(self, X_train, y_train, site_groups=None, categorical_indices=None):
        inner_cv, fit_kwargs = self._build_inner_cv(y_train, site_groups)
        if inner_cv is None:
            return {}, float('nan')

        scoring = resolve_search_scoring(
            self.config.search_scoring,
            age_feature_index=self.search_age_feature_index,
            age_feature_scale=self.search_age_feature_scale,
            age_feature_offset=self.search_age_feature_offset,
        )
        search_preprocessor = self.build_preprocessor(len(y_train), categorical_indices)
        if search_preprocessor is None:
            estimator = self.build_search_model(y_train)
            param_distributions = self.param_dist
        else:
            estimator = Pipeline([
                ('preprocessor', search_preprocessor),
                ('model', self.build_search_model(y_train)),
            ])
            param_distributions = {
                f'model__{name}': values
                for name, values in self.param_dist.items()
            }
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=self.config.search_iterations,
            scoring=scoring,
            cv=inner_cv,
            random_state=self.config.random_state,
            n_jobs=-1,
            refit=False,
        )
        search.fit(X_train, y_train, **fit_kwargs)
        return (
            {
                name.removeprefix('model__'): value
                for name, value in search.best_params_.items()
            },
            float(search.best_score_),
        )

    def _build_inner_cv(self, y_train, site_groups=None):
        fit_kwargs = {}

        if self.config.use_site_grouped_cv and site_groups is not None:
            site_groups = np.asarray(site_groups)
            unique_groups = np.unique(site_groups)
            if unique_groups.size >= 2:
                fit_kwargs['groups'] = site_groups
                return LeaveOneGroupOut(), fit_kwargs

        inner_cv, _ = self._build_stratified_splitter(y_train, self.config.outer_random_splits)
        return inner_cv, fit_kwargs

    def _build_stratified_splitter(self, labels, requested_splits):
        classes, class_counts = np.unique(labels, return_counts=True)
        if len(classes) != 2:
            return None, 0

        n_splits = min(int(requested_splits), int(np.min(class_counts)))
        if n_splits < 2:
            return None, 0

        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        ), n_splits

    def _best_threshold(self, probabilities, labels):
        thresholds = np.linspace(0, 1, 101)
        best_score = -1.0
        best_value = self.default_threshold

        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(np.int32)
            score = f1_score(labels, predictions, zero_division=0)
            if score > best_score:
                best_score = score
                best_value = float(threshold)
            # if best_value < 0.5:
            #     best_value = 0.5

        return best_value

    def _safe_auroc(self, labels, probabilities):
        if len(np.unique(labels)) < 2:
            return np.nan
        return float(roc_auc_score(labels, probabilities))

    def _safe_auprc(self, labels, probabilities):
        if len(np.unique(labels)) < 2:
            return np.nan
        return float(average_precision_score(labels, probabilities, pos_label=1))

    def _compute_evaluation_metrics(self, labels, probabilities, threshold, ages=None):
        labels = np.asarray(labels, dtype=np.int32)
        probabilities = np.asarray(probabilities, dtype=np.float32)
        predictions = (probabilities >= threshold).astype(np.int32)

        age_conditioned_auroc = (
            np.nan
            if ages is None
            else compute_age_conditioned_auroc(labels, probabilities, ages)
        )

        return {
            'threshold': float(threshold),
            'auroc': self._safe_auroc(labels, probabilities),
            'auprc': self._safe_auprc(labels, probabilities),
            'age_conditioned_auroc': age_conditioned_auroc,
            'accuracy': float(accuracy_score(labels, predictions)),
            'f1': float(f1_score(labels, predictions, zero_division=0)),
            'precision': float(precision_score(labels, predictions, zero_division=0)),
            'recall': float(recall_score(labels, predictions, zero_division=0)),
        }

    def _summarize_metrics(self, metric_rows):
        summary = {}
        metric_names = ('age_conditioned_auroc', 'auroc', 'auprc', 'accuracy', 'f1', 'precision', 'recall')

        for metric_name in metric_names:
            metric_values = np.asarray([row[metric_name] for row in metric_rows], dtype=np.float32)
            finite_values = metric_values[np.isfinite(metric_values)]
            if finite_values.size == 0:
                summary[metric_name] = {'mean': None, 'std': None}
                continue

            summary[metric_name] = {
                'mean': float(np.mean(finite_values)),
                'std': float(np.std(finite_values)),
            }

        return summary

    def _consensus_params(self, selected_params_per_fold):
        configurations = {}
        for selected in selected_params_per_fold:
            params = dict(selected['params'])
            configuration_key = tuple(
                sorted((key, self._freeze_param_value(value)) for key, value in params.items())
            )
            configurations.setdefault(configuration_key, []).append({
                'fold': int(selected['fold']),
                'params': params,
                'score': float(selected.get('score', np.nan)),
            })

        def configuration_rank(occurrences):
            scores = np.asarray([entry['score'] for entry in occurrences], dtype=float)
            mean_score = float(np.mean(scores)) if np.all(np.isfinite(scores)) else -np.inf
            first_fold = min(entry['fold'] for entry in occurrences)
            return len(occurrences), mean_score, -first_fold

        winning_occurrences = max(configurations.values(), key=configuration_rank)
        winning_scores = np.asarray(
            [entry['score'] for entry in winning_occurrences],
            dtype=float,
        )
        mean_score = (
            float(np.mean(winning_scores))
            if np.all(np.isfinite(winning_scores))
            else float('nan')
        )
        return (
            dict(winning_occurrences[0]['params']),
            len(winning_occurrences),
            [entry['fold'] for entry in winning_occurrences],
            mean_score,
        )

    def _freeze_param_value(self, value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return tuple(self._freeze_param_value(item) for item in value.tolist())
        if isinstance(value, dict):
            return tuple(
                sorted((key, self._freeze_param_value(item)) for key, item in value.items())
            )
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_param_value(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(self._freeze_param_value(item) for item in value))
        return value

    def _format_metric_value(self, value):
        return 'nan' if value is None or not np.isfinite(value) else f'{value:.3f}'

    def _get_ages_in_years(self, features):
        if self.search_age_feature_index is None:
            return None

        transformed_ages = np.asarray(features)[:, int(self.search_age_feature_index)]
        return (
            np.asarray(transformed_ages, dtype=float).reshape(-1)
            * self.search_age_feature_scale
            + self.search_age_feature_offset
        )

    def _print_metrics(self, prefix, metrics):
        print(
            f"{prefix} AUROC={self._format_metric_value(metrics['auroc'])}, "
            f"Age-conditioned AUROC={self._format_metric_value(metrics['age_conditioned_auroc'])}, "
            f"AUPRC={self._format_metric_value(metrics['auprc'])}, "
            f"Accuracy={self._format_metric_value(metrics['accuracy'])}, "
            f"F1={self._format_metric_value(metrics['f1'])}, "
            f"Precision={self._format_metric_value(metrics['precision'])}, "
            f"Recall={self._format_metric_value(metrics['recall'])}, "
            f"Threshold={metrics['threshold']:.2f}"
        )

    def _print_metric_summary(self, prefix, summary):
        metric_names = ('age_conditioned_auroc', 'auroc', 'auprc', 'accuracy', 'f1', 'precision', 'recall')
        parts = []
        metric_labels = {
            'age_conditioned_auroc': 'Age-conditioned AUROC',
        }
        for metric_name in metric_names:
            metric_summary = summary.get(metric_name, {})
            mean_value = self._format_metric_value(metric_summary.get('mean'))
            std_value = self._format_metric_value(metric_summary.get('std'))
            parts.append(f"{metric_labels.get(metric_name, metric_name.upper())}={mean_value} +/- {std_value}")
        print(f"{prefix} {', '.join(parts)}")