import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier
from biosigpy.hrv.fdmetrics import FdMetricsWarning

from helper_code import DEMOGRAPHICS_FILE, HEADERS, find_patients, load_label

from .config import (
    CV_RANDOM_STATE,
    CV_SEARCH_ITERATIONS,
    CV_SEARCH_SCORING,
    DEFAULT_CV_HYPERPARAMETERS,
    FINAL_SEARCH_CV_STRATEGY,
    MAX_TRAIN_WORKERS,
    OPTIMIZE_HYPERPARAMETER_SEARCH,
    RANDOM_CV_N_SPLITS,
)
from .cross_validation import CrossValidationConfig, EnsembleCrossValidator, normalize_site_group
from .features import get_feature_group_indices, get_feature_names, get_or_create_record_feature_vector
from .metrics import compute_age_conditioned_auroc as compute_auroc_age
from .preprocessing import build_preprocessor, get_processed_feature_names, remap_feature_indices


DEFAULT_ENSEMBLE_THRESHOLD = 0.5
ENSEMBLE_MODALITIES = ('resp', 'eeg', 'ecg')

# Context needed to evaluate the age-conditioned metric on scaled model inputs.
AGE_FEATURE_INDEX = 0
AGE_FEATURE_SCALE = 1.0
AGE_FEATURE_OFFSET = 0.0


def _custom_auroc_age_metric(preds, dtrain):
  labels = dtrain.get_label()
  data = dtrain.get_data()

  # Si data es una matriz dispersa de scipy, convertimos la columna requerida a numpy array
  if hasattr(data, 'toarray'):
    ages = data[:, AGE_FEATURE_INDEX].toarray().ravel()
  else:
    ages = data[:, AGE_FEATURE_INDEX].ravel()

  ages = ages * AGE_FEATURE_SCALE + AGE_FEATURE_OFFSET
  score = compute_auroc_age(labels, preds, ages, gap=2)
  return 'auroc_age', score

# Hyperparameter search space
PARAM_DIST = {
    # El dúo dinámico (Capacidad de aprendizaje)
    'learning_rate':    [0.01, 0.03, 0.05, 0.1, 0.3],  # Tasa de aprendizaje (eta)
    'n_estimators':     [100, 300, 500, 800, 1000],
    
    # Complejidad del árbol (Control de Overfitting)
    'max_depth':        [3, 4, 5, 6, 7, 10], 
    'min_child_weight': [1, 3, 5, 7], 
    
    # Muestreo (Añade aleatoriedad para robustez)
    'subsample':        [0.6, 0.8, 1.0], 
    'colsample_bytree': [0.6, 0.8, 1.0], 
    
    # Regularización Matemática (¡Clave para cerrar la brecha Train vs Test!)
    'reg_lambda':       [1.0, 2.0, 5.0, 10.0],   # Penaliza pesos grandes (L2)
    'reg_alpha':        [0.0, 0.1, 0.5, 1.0, 2.0]   # Puede colapsar variables irrelevantes a cero (L1)
}

#Utiliza caches en memoria para evitar relecturas de disco y maneja las excepciones de forma defensiva para que un archivo corrupto no tire abajo el entrenamiento de horas.
def build_training_metadata_cache(patient_data_file):
    metadata = pd.read_csv(patient_data_file)
    demographics_cache = {}
    diagnosis_cache = {}

    for row in metadata.to_dict('records'):
        patient_id = row[HEADERS['bids_folder']]
        session_id = row[HEADERS['session_id']]
        demographics_cache[(patient_id, session_id)] = row
        diagnosis_cache[patient_id] = load_label(row)

    return demographics_cache, diagnosis_cache


def process_training_record(record, data_folder, demographics_cache, diagnosis_cache, csv_path):
    patient_id = record[HEADERS['bids_folder']]
    session_id = record[HEADERS['session_id']]
    site_id = record[HEADERS['site_id']]

    try:
        patient_data = demographics_cache.get((patient_id, session_id), {})
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter('always', FdMetricsWarning)
            feature_vector, cache_hit = get_or_create_record_feature_vector(
                record,
                data_folder,
                patient_data,
                csv_path=csv_path,
                require_physiological_data=True,
                return_cache_hit=True,
            )
        has_excessive_vlf_power = any(
            issubclass(warning.category, FdMetricsWarning)
            and getattr(warning.message, 'warning_id', None) == 'excessive_vlf_power'
            for warning in captured_warnings
        )

        label = diagnosis_cache.get(patient_id)
        metadata = {
            'patient_id': patient_id,
            'site_id': site_id,
            'session_id': session_id,
        }

        if label == 0 or label == 1:
            return metadata, feature_vector, label, None, has_excessive_vlf_power, cache_hit

        return metadata, None, None, f"Invalid label for {patient_id}. Skipping...", has_excessive_vlf_power, cache_hit

    except FileNotFoundError as exc:
        return {
            'patient_id': patient_id,
            'site_id': site_id,
            'session_id': session_id,
        }, None, None, f"{exc} Skipping...", False, False
    except Exception as exc:
        return {
            'patient_id': patient_id,
            'site_id': site_id,
            'session_id': session_id,
        }, None, None, f"Error processing {patient_id}: {exc}", False, False

def prepare_feature_matrix(feature_matrix, preprocessor=None):
    raw_feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
    if raw_feature_matrix.ndim == 1:
        raw_feature_matrix = raw_feature_matrix.reshape(1, -1)
    raw_feature_matrix = raw_feature_matrix.copy()
    raw_feature_matrix[~np.isfinite(raw_feature_matrix)] = np.nan

    if preprocessor is not None:
        processed_feature_matrix = np.asarray(preprocessor.transform(raw_feature_matrix), dtype=np.float32)
    else:
        processed_feature_matrix = raw_feature_matrix

    return raw_feature_matrix, processed_feature_matrix

def export_feature_matrix_csv(output_path, metadata_rows, feature_matrix, feature_names, labels=None):
    dataframe = pd.DataFrame(metadata_rows)
    if labels is not None:
        dataframe['label'] = labels
    feature_frame = pd.DataFrame(feature_matrix, columns=feature_names)
    dataframe = pd.concat([dataframe.reset_index(drop=True), feature_frame.reset_index(drop=True)], axis=1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataframe.to_csv(output_path, index=False)

def get_feature_export_paths(export_root, prefix):
    return {
        'raw': os.path.join(export_root, f'{prefix}_features_raw.csv'),
        'preprocessed': os.path.join(export_root, f'{prefix}_features_preprocessed.csv'),
    }


def _get_feature_group_name(feature_index, modality_presence_indices):
    for group_name in ('resp', 'eeg', 'ecg'):
        group_index_set = set(np.asarray(modality_presence_indices[group_name], dtype=np.int32).tolist())
        if feature_index in group_index_set:
            return group_name

    return 'demographics'


def export_selected_features_csv(output_path, feature_names, selected_raw_feature_indices, modality_presence_indices):
    selected_rows = []
    for processed_index, raw_index in enumerate(np.asarray(selected_raw_feature_indices, dtype=np.int32)):
        selected_rows.append({
            'processed_index': int(processed_index),
            'raw_index': int(raw_index),
            'feature_name': feature_names[int(raw_index)],
            'group': _get_feature_group_name(int(raw_index), modality_presence_indices),
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(selected_rows).to_csv(output_path, index=False)
    
def export_feature_views(export_root, prefix, metadata_rows, feature_matrix, feature_names, preprocessor=None, labels=None):
    raw_feature_matrix, processed_feature_matrix = prepare_feature_matrix(
        feature_matrix,
        preprocessor=preprocessor,
    )
    export_paths = get_feature_export_paths(export_root, prefix)
    export_feature_matrix_csv(
        export_paths['raw'],
        metadata_rows,
        raw_feature_matrix,
        feature_names,
        labels=labels,
    )
    export_feature_matrix_csv(
        export_paths['preprocessed'],
        metadata_rows,
        processed_feature_matrix,
        get_processed_feature_names(feature_names, preprocessor=preprocessor),
        labels=labels,
    )
    return export_paths    

#El dataset tieme pocas muestras positivas (enfermos, un 13.2% del dataset) frente a negativas (sanos). El cálculo automático de scale_pos_weight rebalancea la función de pérdida del algoritmo penalizando más los fallos en la clase minoritaria. Esto previene que el modelo se vuelva perezoso y prediga siempre la clase mayoritaria.
def _build_xgb_model(labels, extra_params=None):
   
    neg = int(np.sum(labels == 0))
    pos = int(np.sum(labels == 1))
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
 
    base_params = dict(
        scale_pos_weight=scale_pos_weight,
        n_estimators=500,
        learning_rate=0.05, 
        max_depth=4, 
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=2, 
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        eval_metric=_custom_auroc_age_metric,
        tree_method='hist',
    )
    if extra_params:
        base_params.update(extra_params)
                
    return XGBClassifier(**base_params)

def _fit_model(feature_matrix, labels, final_params=None):
    model = _build_xgb_model(labels, extra_params=final_params)
    model.fit(feature_matrix, labels)   
    
    return model


def _build_search_model(labels):
    return XGBClassifier(
        scale_pos_weight=(int(np.sum(labels == 0)) / max(int(np.sum(labels == 1)), 1)),
        n_estimators=500,
        learning_rate=0.05,
        random_state=CV_RANDOM_STATE,
        eval_metric=_custom_auroc_age_metric,
        tree_method='hist',
    )


def _get_combined_model_indices(feature_indices):
    """
    Combines each signal modality and group combinations with demographic features.
    Guarantees demographic indices are always included for all signal configurations.
    """
    demo_indices = np.asarray(feature_indices.get('demographics', []), dtype=np.int32)
    
    def _combine(modality_keys):
        combined = set(demo_indices.tolist())
        for key in modality_keys:
            if key in feature_indices:
                combined.update(np.asarray(feature_indices[key], dtype=np.int32).tolist())
        return np.array(sorted(list(combined)), dtype=np.int32)

    combined_map = {
        # Single signal + Demographics (3 models)
        'ecg': _combine(['ecg']),
        'eeg': _combine(['eeg']),
        'resp': _combine(['resp']),
        
        # Dual ensembles + Demographics (3 ensemble models)
        'ecg_eeg': _combine(['ecg', 'eeg']),
        'ecg_resp': _combine(['ecg', 'resp']),
        'eeg_resp': _combine(['eeg', 'resp']),
        
        # All signals + Demographics (1 complete ensemble model)
        'all': _combine(['ecg', 'eeg', 'resp'])
    }
    return combined_map


def _get_model_modalities(model_name):
    return ENSEMBLE_MODALITIES if model_name == 'all' else tuple(model_name.split('_'))


def _get_route_presence_mask(feature_matrix, model_name, modality_presence_indices):
    mask = np.ones(feature_matrix.shape[0], dtype=bool)
    for modality in _get_model_modalities(model_name):
        modality_indices = np.asarray(modality_presence_indices[modality], dtype=np.int32)
        mask &= np.any(np.isfinite(feature_matrix[:, modality_indices]), axis=1)
    return mask


def _get_route_categorical_indices(raw_indices, categorical_indices):
    categorical_set = set(np.asarray(categorical_indices or [], dtype=np.int32).tolist())
    return [
        position for position, raw_index in enumerate(raw_indices)
        if int(raw_index) in categorical_set
    ]


def _get_ecg_eeg_search_data(
    feature_matrix,
    labels,
    feature_indices,
    modality_presence_indices,
    categorical_indices=None,
    site_groups=None,
):
    raw_indices = _get_combined_model_indices(feature_indices)['ecg_eeg']
    eligible_mask = _get_route_presence_mask(
        feature_matrix,
        'ecg_eeg',
        modality_presence_indices,
    )
    return {
        'features': feature_matrix[eligible_mask][:, raw_indices],
        'labels': labels[eligible_mask],
        'categorical_indices': _get_route_categorical_indices(raw_indices, categorical_indices),
        'site_groups': None if site_groups is None else site_groups[eligible_mask],
        'route_name': 'ecg_eeg',
        'raw_indices': raw_indices,
        'age_feature_index': int(np.flatnonzero(raw_indices == 0)[0]),
    }


def _fit_route_model(route_features, route_labels, raw_indices, categorical_indices, final_params):
    route_preprocessor = build_preprocessor(
        len(route_labels),
        _get_route_categorical_indices(raw_indices, categorical_indices),
    )
    processed_features = np.asarray(
        route_preprocessor.fit_transform(route_features),
        dtype=np.float32,
    )
    return {
        'model': _fit_model(processed_features, route_labels, final_params),
        'preprocessor': route_preprocessor,
        'raw_indices': np.asarray(raw_indices, dtype=np.int32),
        'n_train': int(len(route_labels)),
        'n_positive_train': int(np.sum(route_labels == 1)),
        'n_negative_train': int(np.sum(route_labels == 0)),
    }


# Entrena cada ruta sólo con pacientes que aportan todas sus modalidades requeridas.
def _fit_ensemble(
    feature_matrix,
    labels,
    feature_indices,
    final_params=None,
    modality_presence_indices=None,
    categorical_indices=None,
):
    models = {}
    combined_indices = _get_combined_model_indices(feature_indices)
    if modality_presence_indices is None:
        modality_presence_indices = feature_indices

    for model_name, raw_indices in combined_indices.items():
        route_mask = _get_route_presence_mask(
            feature_matrix,
            model_name,
            modality_presence_indices,
        )
        route_labels = labels[route_mask]
        if np.unique(route_labels).size < 2:
            print(
                f"  Skipping {model_name}: its available-signal training subset "
                "does not contain both classes."
            )
            continue

        route_features = feature_matrix[route_mask][:, raw_indices]
        if raw_indices.size == 0:
            continue
        models[model_name] = _fit_route_model(
            route_features,
            route_labels,
            raw_indices,
            categorical_indices,
            final_params,
        )
        
    return models


def _has_modality_signal(feature_vector, modality_presence_indices):
    modality_values = feature_vector[modality_presence_indices]
    return bool(np.any(np.isfinite(modality_values)))


def _select_ensemble_model_name(raw_feature_vector, models, modality_presence_indices):
    active_modalities = {
        modality
        for modality in ENSEMBLE_MODALITIES
        if modality in modality_presence_indices
        and _has_modality_signal(raw_feature_vector, modality_presence_indices[modality])
    }

    # Prefer ECG+EEG whenever possible; RESP may still support ECG feature extraction,
    # but its standalone feature block is not part of the preferred XGBoost route.
    candidate_models = (
        (frozenset(('ecg', 'eeg')), 'ecg_eeg'),
        (frozenset(('eeg',)), 'eeg'),
        (frozenset(('ecg',)), 'ecg'),
        (frozenset(('ecg', 'eeg', 'resp')), 'all'),
        (frozenset(('ecg', 'resp')), 'ecg_resp'),
        (frozenset(('eeg', 'resp')), 'eeg_resp'),
        (frozenset(('resp',)), 'resp'),
    )
    for modalities, model_name in candidate_models:
        if modalities.issubset(active_modalities) and model_name in models:
            return model_name

    return 'all' if 'all' in models else next(iter(models))


def select_ensemble_model_names(model_bundle, feature_matrix):
    raw_feature_matrix, _ = prepare_feature_matrix(feature_matrix)
    modality_presence_indices = {
        name: np.asarray(indices, dtype=np.int32)
        for name, indices in model_bundle['modality_presence_indices'].items()
    }
    models = model_bundle['models']
    return np.asarray([
        _select_ensemble_model_name(raw_feature_vector, models, modality_presence_indices)
        for raw_feature_vector in raw_feature_matrix
    ], dtype=object)

def predict_ensemble_probabilities(model_bundle, feature_matrix):
    raw_feature_matrix, _ = prepare_feature_matrix(feature_matrix)

    models = model_bundle['models']
    
    # Reconstruct combined indices maps
    if 'combined_indices' in model_bundle:
        combined_indices = {
            k: np.asarray(v, dtype=np.int32) for k, v in model_bundle['combined_indices'].items()
        }
    else:
        raw_indices = {
            name: np.asarray(indices, dtype=np.int32)
            for name, indices in model_bundle['feature_indices'].items()
        }
        combined_indices = _get_combined_model_indices(raw_indices)

    modality_presence_indices = {
        name: np.asarray(indices, dtype=np.int32)
        for name, indices in model_bundle['modality_presence_indices'].items()
    }
    
    probabilities = np.zeros(raw_feature_matrix.shape[0], dtype=np.float32)
    for row_index, raw_feature_vector in enumerate(raw_feature_matrix):
        target_model = _select_ensemble_model_name(
            raw_feature_vector,
            models,
            modality_presence_indices,
        )

        route_model = models[target_model]
        if isinstance(route_model, dict):
            route_features = raw_feature_vector[route_model['raw_indices']].reshape(1, -1)
            processed_features = route_model['preprocessor'].transform(route_features)
            probabilities[row_index] = float(
                route_model['model'].predict_proba(processed_features)[0][1]
            )
        else:
            _, processed_feature_matrix = prepare_feature_matrix(
                raw_feature_matrix,
                preprocessor=model_bundle.get('preprocessor'),
            )
            target_indices = combined_indices[target_model]
            model_features = processed_feature_matrix[row_index, target_indices].reshape(1, -1)
            probabilities[row_index] = float(route_model.predict_proba(model_features)[0][1])

    return probabilities


def predict_route_model_probabilities(model_bundle, feature_matrix):
    raw_feature_matrix, _ = prepare_feature_matrix(feature_matrix)
    probabilities_by_model = {}
    modality_presence_indices = {
        name: np.asarray(indices, dtype=np.int32)
        for name, indices in model_bundle['modality_presence_indices'].items()
    }
    for model_name, route_model in model_bundle['models'].items():
        if not isinstance(route_model, dict):
            continue
        eligible_mask = _get_route_presence_mask(
            raw_feature_matrix,
            model_name,
            modality_presence_indices,
        )
        model_probabilities = np.full(len(raw_feature_matrix), np.nan, dtype=np.float32)
        if not np.any(eligible_mask):
            probabilities_by_model[model_name] = model_probabilities
            continue
        route_features = raw_feature_matrix[eligible_mask][:, route_model['raw_indices']]
        processed_features = route_model['preprocessor'].transform(route_features)
        model_probabilities[eligible_mask] = route_model['model'].predict_proba(
            processed_features
        )[:, 1]
        probabilities_by_model[model_name] = model_probabilities
    return probabilities_by_model

#define el umbral de decisión para convertir probabilidades en etiquetas binarias. Si la probabilidad es mayor o igual al umbral, se asigna la etiqueta 1 (positivo), de lo contrario, se asigna la etiqueta 0 (negativo).
def predict_ensemble_labels(model_bundle, feature_matrix):
    threshold = float(model_bundle.get('threshold', DEFAULT_ENSEMBLE_THRESHOLD))
    probabilities = predict_ensemble_probabilities(model_bundle, feature_matrix)
    labels = (probabilities >= threshold).astype(np.int32)
    return labels, probabilities


def _evaluate_and_display_models(
    models,
    feature_matrix,
    labels,
    modality_presence_indices,
    threshold=0.5,
):
    """
    Computes and displays evaluation metrics for all trained models on the dataset.
    """
    print("\n" + "="*85)
    print("                      EVALUATION METRICS FOR ALL MODELS                      ")
    print("="*85)
    print(f"{'Model Name':<25} | {'AUROC-Age':<10} | {'ROC-AUC':<10} | {'Accuracy':<10} | {'Sensitivity':<11} | {'Specificity':<11}")
    print("-" * 85)

    metrics_summary = {}

    for model_name, route_model in models.items():
        if not isinstance(route_model, dict):
            continue

        route_mask = _get_route_presence_mask(
            feature_matrix,
            model_name,
            modality_presence_indices,
        )
        route_labels = labels[route_mask]
        route_features = feature_matrix[route_mask][:, route_model['raw_indices']]
        processed_features = route_model['preprocessor'].transform(route_features)
        probs = route_model['model'].predict_proba(processed_features)[:, 1]
        preds = (probs >= threshold).astype(np.int32)
        age_position = np.flatnonzero(route_model['raw_indices'] == 0)
        ages = route_features[:, int(age_position[0])] if age_position.size else np.zeros(len(route_labels))
        
        # Calculate evaluation metrics
        auroc_age = compute_auroc_age(route_labels, probs, ages, gap=2)
        roc_auc = roc_auc_score(route_labels, probs) if len(np.unique(route_labels)) > 1 else 0.0
        acc = accuracy_score(route_labels, preds)
        sens = recall_score(route_labels, preds, zero_division=0)
        
        cm = confusion_matrix(route_labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics_summary[model_name] = {
            'auroc_age': float(auroc_age),
            'roc_auc': float(roc_auc),
            'accuracy': float(acc),
            'sensitivity': float(sens),
            'specificity': float(spec),
        }

        print(f"{model_name:<25} | {auroc_age:<10.4f} | {roc_auc:<10.4f} | {acc:<10.4f} | {sens:<11.4f} | {spec:<11.4f}")

    print("="*85 + "\n")
    return metrics_summary


def train_multimodal_ensemble(data_folder, verbose, csv_path, export_folder=None):
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    demographics_cache, diagnosis_cache = build_training_metadata_cache(patient_data_file)
    num_records = len(patient_metadata_list)
    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    features = []
    labels = []
    metadata_rows = []
    excessive_vlf_patient_count = 0
    extracted_feature_count = 0

    with ThreadPoolExecutor(max_workers=MAX_TRAIN_WORKERS) as executor:
        futures = {
            executor.submit(
                process_training_record,
                record,
                data_folder,
                demographics_cache,
                diagnosis_cache,
                csv_path,
            ): index
            for index, record in enumerate(patient_metadata_list)
        }
        ordered_results = [None] * num_records

        pbar = tqdm(
            total=num_records,
            desc='Extracting Features',
            unit='record',
            disable=not verbose,
        )
        for future in as_completed(futures):
            result = future.result()
            ordered_results[futures[future]] = result
            if verbose:
                pbar.set_postfix({'patient': result[0]['patient_id']})
            pbar.update(1)

        pbar.close()

    for result in ordered_results:
        if result is None:
            continue
        metadata, feature_vector, label, message, has_excessive_vlf_power, cache_hit = result
        excessive_vlf_patient_count += has_excessive_vlf_power
        extracted_feature_count += not cache_hit
        if message is not None:
            tqdm.write(f"  ! {message}")
            continue

        features.append(feature_vector)
        labels.append(label)
        metadata_rows.append(metadata)

    if extracted_feature_count:
        tqdm.write(
            f"  ! excessive_vlf_power affected {excessive_vlf_patient_count} patients."
        )

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    if features.size == 0 or features.ndim != 2 or features.shape[0] == 0:
        raise ValueError('No valid training samples were extracted. Review feature extraction logs for the skipped records.')

    feature_names = list(get_feature_names())
    feature_indices = get_feature_group_indices(include_demographics=True)
    modality_presence_indices = get_feature_group_indices(include_demographics=False)

    categorical_indices = [
        i for i, name in enumerate(feature_names)
        if name.lower() in ('sex', 'gender')
    ]
    site_groups = np.asarray([
        normalize_site_group(metadata_row['site_id'])
        for metadata_row in metadata_rows
    ])
    cv_config = CrossValidationConfig(
        final_search_cv_strategy=FINAL_SEARCH_CV_STRATEGY,
        optimize_hyperparameter_search=OPTIMIZE_HYPERPARAMETER_SEARCH,
        outer_random_splits=RANDOM_CV_N_SPLITS,
        random_state=CV_RANDOM_STATE,
        search_iterations=CV_SEARCH_ITERATIONS,
        search_scoring=CV_SEARCH_SCORING,
        fixed_hyperparameters=DEFAULT_CV_HYPERPARAMETERS,
    )

    print(f"  Categorical feature indices: {categorical_indices} "
          f"({[feature_names[i] for i in categorical_indices]})")
    print(f"  Hospital CV groups: {sorted(np.unique(site_groups).tolist())}")
    print(f"  Hyperparameter search: {'enabled' if cv_config.optimize_hyperparameter_search else 'disabled'}")
    print(f"  Hyperparameter search scoring: {cv_config.search_scoring}")
    
    raw_age_feature_index = feature_names.index('Age')

    # Cross-validation receives raw features and fits preprocessing inside each fold.
    cv_runner = EnsembleCrossValidator(
        config=cv_config,
        param_dist=PARAM_DIST,
        default_threshold=DEFAULT_ENSEMBLE_THRESHOLD,
        build_preprocessor=build_preprocessor,
        build_search_model=_build_search_model,
        fit_ensemble=_fit_ensemble,
        predict_probabilities=predict_ensemble_probabilities,
        select_model_names=select_ensemble_model_names,
        predict_model_probabilities=predict_route_model_probabilities,
        fit_ensemble_handles_preprocessing=True,
        select_search_data=_get_ecg_eeg_search_data,
        search_age_feature_index=raw_age_feature_index,
        search_age_feature_scale=1.0,
        search_age_feature_offset=0.0,
    )

    print("\n" + "=" * 60)
    print("RANDOM STRATIFIED NESTED CV")
    print("=" * 60)
    random_cv_result = cv_runner.evaluate_random_nested_cv(
        features,
        labels,
        feature_indices,
        modality_presence_indices=modality_presence_indices,
        categorical_indices=categorical_indices if categorical_indices else None,
    )
    print("\n" + "=" * 60)
    print("LEAVE-ONE-HOSPITAL-OUT NESTED CV")
    print("=" * 60)
    hospital_cv_result = cv_runner.evaluate_grouped_nested_cv(
        features,
        labels,
        feature_indices,
        modality_presence_indices=modality_presence_indices,
        categorical_indices=categorical_indices if categorical_indices else None,
        site_groups=site_groups,
    )
    print("\n" + "=" * 60)
    print("FINAL HYPERPARAMETER SEARCH")
    print("=" * 60)
    final_params, final_search_score = cv_runner.select_final_params(
        features,
        labels,
        feature_indices,
        modality_presence_indices,
        categorical_indices=categorical_indices if categorical_indices else None,
        site_groups=site_groups,
    )
    print(f"  Best final parameters: {final_params}")
    print(f"  Final CV search score: {final_search_score:.3f}")
    threshold = hospital_cv_result.threshold
    if hospital_cv_result.metrics.get('skipped'):
        threshold = random_cv_result.threshold

    # This preprocessor is retained for feature exports only. Each deployed route
    # fits and stores its own preprocessor below.
    print("\n[INFO] Fitting feature-export preprocessing on all training data...")
    preprocessor = build_preprocessor(
        len(labels),
        categorical_indices if categorical_indices else None,
    )
    processed_features = np.asarray(
        preprocessor.fit_transform(features),
        dtype=np.float32,
    )
    selected_feature_indices = preprocessor.transform_feature_indices(feature_indices)
    processed_feature_names = get_processed_feature_names(
        feature_names,
        preprocessor=preprocessor,
    )
    real_selected_num_indices = preprocessor._numerical_indices[
        preprocessor.selector.selected_indices_
    ]
    selected_raw_feature_indices = np.concatenate([
        real_selected_num_indices,
        preprocessor.categorical_indices_,
    ]).astype(np.int32)
    print(
        f"Correlation selector: kept {len(processed_feature_names)}/"
        f"{len(feature_names)} features for the final model."
    )

    global AGE_FEATURE_INDEX, AGE_FEATURE_SCALE, AGE_FEATURE_OFFSET
    if 'Age' not in processed_feature_names:
        raise ValueError(
            "The 'Age' feature is required for age-conditioned AUROC scoring."
        )
    AGE_FEATURE_INDEX = processed_feature_names.index('Age')
    numerical_age_positions = np.flatnonzero(
        preprocessor._numerical_indices == raw_age_feature_index
    )
    if numerical_age_positions.size != 1:
        raise ValueError(
            "Could not recover the scaling parameters for the 'Age' feature."
        )
    numerical_age_position = int(numerical_age_positions[0])
    AGE_FEATURE_SCALE = float(preprocessor.scaler.scale_[numerical_age_position])
    AGE_FEATURE_OFFSET = float(preprocessor.scaler.mean_[numerical_age_position])
    combined_indices = _get_combined_model_indices(feature_indices)


    print("\n" + "=" * 60)
    print("FINAL FIT")
    print("=" * 60)
    print("Fitting final ensemble on all training data...")
    models = _fit_ensemble(
        features,
        labels,
        feature_indices,
        final_params=final_params,
        modality_presence_indices=modality_presence_indices,
        categorical_indices=categorical_indices,
    )

    # --- Step 3: Compute & Display Metrics for All Trained Models ---
    training_metrics = _evaluate_and_display_models(
        models=models,
        feature_matrix=features,
        labels=labels,
        modality_presence_indices=modality_presence_indices,
        threshold=threshold,
    )

    # Exportar métricas e información de diagnóstico a los CSV
    export_root = export_folder or os.path.join(os.getcwd(), 'feature_exports')
    feature_exports = export_feature_views(
        export_root,
        'training',
        metadata_rows,
        features,
        feature_names,
        preprocessor=preprocessor,
        labels=labels,
    )
    selected_features_csv = os.path.join(export_root, 'training_features_selected.csv')
    export_selected_features_csv(
        selected_features_csv,
        feature_names,
        selected_raw_feature_indices,
        modality_presence_indices,
    )
    feature_exports['selected'] = selected_features_csv      
    return {
        'type': 'multimodal_xgb_ensemble',
        'threshold': threshold,
        'feature_names': feature_names,
        'processed_feature_names': processed_feature_names,
        'selected_raw_feature_indices': selected_raw_feature_indices.tolist(),
        'feature_indices': {
            name: indices.tolist()
            for name, indices in feature_indices.items()
            if name in {'all', 'resp', 'eeg', 'ecg', 'demographics'}
        },
        'combined_indices': {
            k: v.tolist() for k, v in combined_indices.items()
        },
        'modality_presence_indices': {
            modality: modality_presence_indices[modality].tolist()
            for modality in ENSEMBLE_MODALITIES
        },
        'models': models,
        'preprocessor': preprocessor,
        'feature_exports': feature_exports,
        'random_cv_metrics': random_cv_result.metrics,
        'hospital_cv_metrics': hospital_cv_result.metrics,
        'final_params': final_params,
        'final_search_score': final_search_score,
        'training_metrics': training_metrics,
    }