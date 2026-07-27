# import os
# from concurrent.futures import ThreadPoolExecutor
# from sklearn.model_selection import train_test_split
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from xgboost import XGBClassifier

# from helper_code import DEMOGRAPHICS_FILE, HEADERS, find_patients, load_label

# from .config import (
#     CV_RANDOM_STATE,
#     CV_SEARCH_ITERATIONS,
#     DEFAULT_CV_HYPERPARAMETERS,
#     MAX_TRAIN_WORKERS,
#     OPTIMIZE_HYPERPARAMETER_SEARCH,
#     RANDOM_CV_N_SPLITS,
#     USE_SITE_GROUPED_CV,
# )
# from .cross_validation import CrossValidationConfig, EnsembleCrossValidator, normalize_site_group
# from .features import get_feature_group_indices, get_feature_names, get_or_create_record_feature_vector
# from .preprocessing import build_preprocessor, get_processed_feature_names, remap_feature_indices


# DEFAULT_ENSEMBLE_THRESHOLD = 0.5
# ENSEMBLE_MODALITIES = ('resp', 'eeg', 'ecg')

# # Hyperparameter search space
# PARAM_DIST = {
#     # El dúo dinámico (Capacidad de aprendizaje)
#     'learning_rate':    [0.01, 0.03, 0.05, 0.1, 0.3],  # Tasa de aprendizaje (eta)
#     'n_estimators':     [100, 300, 500, 800, 1000],
    
#     # Complejidad del árbol (Control de Overfitting)
#     'max_depth':        [3, 4, 5, 6, 7, 10], 
#     'min_child_weight': [1, 3, 5, 7], 
    
#     # Muestreo (Añade aleatoriedad para robustez)
#     'subsample':        [0.6, 0.8, 1.0], 
#     'colsample_bytree': [0.6, 0.8, 1.0], 
    
#     # Regularización Matemática (¡Clave para cerrar la brecha Train vs Test!)
#     'reg_lambda':       [1.0, 2.0, 5.0, 10.0],   # Penaliza pesos grandes (L2)
#     'reg_alpha':        [0.0, 0.1, 0.5, 1.0, 2.0]   # Puede colapsar variables irrelevantes a cero (L1)
# }

# #Utiliza caches en memoria para evitar relecturas de disco y maneja las excepciones de forma defensiva para que un archivo corrupto no tire abajo el entrenamiento de horas.
# def build_training_metadata_cache(patient_data_file):
#     metadata = pd.read_csv(patient_data_file)
#     demographics_cache = {}
#     diagnosis_cache = {}

#     for row in metadata.to_dict('records'):
#         patient_id = row[HEADERS['bids_folder']]
#         session_id = row[HEADERS['session_id']]
#         demographics_cache[(patient_id, session_id)] = row
#         diagnosis_cache[patient_id] = load_label(row)

#     return demographics_cache, diagnosis_cache


# def process_training_record(record, data_folder, demographics_cache, diagnosis_cache, csv_path):
#     patient_id = record[HEADERS['bids_folder']]
#     session_id = record[HEADERS['session_id']]
#     site_id = record[HEADERS['site_id']]

#     try:
#         patient_data = demographics_cache.get((patient_id, session_id), {})
#         feature_vector = get_or_create_record_feature_vector(
#             record,
#             data_folder,
#             patient_data,
#             csv_path=csv_path,
#             require_physiological_data=True,
#         )

#         label = diagnosis_cache.get(patient_id)
#         metadata = {
#             'patient_id': patient_id,
#             'site_id': site_id,
#             'session_id': session_id,
#         }

#         if label == 0 or label == 1:
#             return metadata, feature_vector, label, None

#         return metadata, None, None, f"Invalid label for {patient_id}. Skipping..."

#     except FileNotFoundError as exc:
#         return {
#             'patient_id': patient_id,
#             'site_id': site_id,
#             'session_id': session_id,
#         }, None, None, f"{exc} Skipping..."
#     except Exception as exc:
#         return {
#             'patient_id': patient_id,
#             'site_id': site_id,
#             'session_id': session_id,
#         }, None, None, f"Error processing {patient_id}: {exc}"

# def prepare_feature_matrix(feature_matrix, preprocessor=None):
#     raw_feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
#     if raw_feature_matrix.ndim == 1:
#         raw_feature_matrix = raw_feature_matrix.reshape(1, -1)
#     raw_feature_matrix = raw_feature_matrix.copy()
#     raw_feature_matrix[~np.isfinite(raw_feature_matrix)] = np.nan

#     if preprocessor is not None:
#         processed_feature_matrix = np.asarray(preprocessor.transform(raw_feature_matrix), dtype=np.float32)
#     else:
#         processed_feature_matrix = raw_feature_matrix

#     return raw_feature_matrix, processed_feature_matrix

# def export_feature_matrix_csv(output_path, metadata_rows, feature_matrix, feature_names, labels=None):
#     dataframe = pd.DataFrame(metadata_rows)
#     if labels is not None:
#         dataframe['label'] = labels
#     feature_frame = pd.DataFrame(feature_matrix, columns=feature_names)
#     dataframe = pd.concat([dataframe.reset_index(drop=True), feature_frame.reset_index(drop=True)], axis=1)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     dataframe.to_csv(output_path, index=False)

# def get_feature_export_paths(export_root, prefix):
#     return {
#         'raw': os.path.join(export_root, f'{prefix}_features_raw.csv'),
#         'preprocessed': os.path.join(export_root, f'{prefix}_features_preprocessed.csv'),
#     }


# def _get_feature_group_name(feature_index, modality_presence_indices):
#     for group_name in ('resp', 'eeg', 'ecg'):
#         group_index_set = set(np.asarray(modality_presence_indices[group_name], dtype=np.int32).tolist())
#         if feature_index in group_index_set:
#             return group_name

#     return 'demographics'


# def export_selected_features_csv(output_path, feature_names, selected_raw_feature_indices, modality_presence_indices):
#     selected_rows = []
#     for processed_index, raw_index in enumerate(np.asarray(selected_raw_feature_indices, dtype=np.int32)):
#         selected_rows.append({
#             'processed_index': int(processed_index),
#             'raw_index': int(raw_index),
#             'feature_name': feature_names[int(raw_index)],
#             'group': _get_feature_group_name(int(raw_index), modality_presence_indices),
#         })

#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     pd.DataFrame(selected_rows).to_csv(output_path, index=False)
    
# def export_feature_views(export_root, prefix, metadata_rows, feature_matrix, feature_names, preprocessor=None, labels=None):
#     raw_feature_matrix, processed_feature_matrix = prepare_feature_matrix(
#         feature_matrix,
#         preprocessor=preprocessor,
#     )
#     export_paths = get_feature_export_paths(export_root, prefix)
#     export_feature_matrix_csv(
#         export_paths['raw'],
#         metadata_rows,
#         raw_feature_matrix,
#         feature_names,
#         labels=labels,
#     )
#     export_feature_matrix_csv(
#         export_paths['preprocessed'],
#         metadata_rows,
#         processed_feature_matrix,
#         get_processed_feature_names(feature_names, preprocessor=preprocessor),
#         labels=labels,
#     )
#     return export_paths    

# #El dataset tieme pocas muestras positivas (enfermos, un 13.2% del dataset) frente a negativas (sanos). El cálculo automático de scale_pos_weight rebalancea la función de pérdida del algoritmo penalizando más los fallos en la clase minoritaria. Esto previene que el modelo se vuelva perezoso y prediga siempre la clase mayoritaria.
# def _build_xgb_model(labels, extra_params=None):
   
#     neg = int(np.sum(labels == 0))
#     pos = int(np.sum(labels == 1))
#     scale_pos_weight = (neg / pos) if pos > 0 else 1.0
 
#     base_params = dict(
#         scale_pos_weight=scale_pos_weight,
#         n_estimators=500,
#         learning_rate=0.05, 
#         max_depth=4, 
#         subsample=0.8,
#         colsample_bytree=0.7,
#         min_child_weight=2, 
#         reg_alpha=0.0,
#         reg_lambda=1.0,
#         random_state=42,
#         eval_metric='auc',
#         tree_method='hist',
#     )
#     if extra_params:
#         base_params.update(extra_params)
                
#     return XGBClassifier(**base_params)

# def _fit_model(feature_matrix, labels, consensus_params=None):
#     model = _build_xgb_model(labels, extra_params=consensus_params) 
#     model.fit(feature_matrix, labels)   
    
#     return model


# def _build_search_model(labels):
#     return XGBClassifier(
#         scale_pos_weight=(int(np.sum(labels == 0)) / max(int(np.sum(labels == 1)), 1)),
#         n_estimators=500,
#         learning_rate=0.05,
#         random_state=CV_RANDOM_STATE,
#         eval_metric='auc',
#         tree_method='hist',
#     )
# #Entrena múltiples modelos XGBoost en paralelo: uno global (all) con todas las variables y uno independiente por cada señal médica disponible (resp, eeg, ecg)
# def _fit_ensemble(feature_matrix, labels, feature_indices, consensus_params=None):
#     models = {}
#     if feature_indices['all'].size == 0:
#         raise ValueError('Correlation selector removed all features for the global model.')

#     models['all'] = _fit_model(
#         feature_matrix[:, feature_indices['all']], labels, consensus_params
#     )
#     for modality in ENSEMBLE_MODALITIES:
#         if feature_indices[modality].size == 0:
#             continue
#         models[modality] = _fit_model(
#             feature_matrix[:, feature_indices[modality]], labels, consensus_params
#         )
#     return models


# def _has_modality_signal(feature_vector, modality_presence_indices):
#     modality_values = feature_vector[modality_presence_indices]
#     return bool(np.any(np.isfinite(modality_values)))

# #En la fase de predicción, recorre paciente por paciente. Si el paciente tiene señales válidas para una modalidad, calcula la probabilidad de esa modalidad. Al final, promedia las probabilidades de los modelos especializados disponibles. Si el paciente no tiene ninguna señal médica, recurre al modelo de respaldo all
# """ def predict_ensemble_probabilities(model_bundle, feature_matrix):
#     raw_feature_matrix, processed_feature_matrix = prepare_feature_matrix(
#         feature_matrix,
#         preprocessor=model_bundle.get('preprocessor'),
#     )

#     models = model_bundle['models']
#     feature_indices = {
#         name: np.asarray(indices, dtype=np.int32)
#         for name, indices in model_bundle['feature_indices'].items()
#     }
#     modality_presence_indices = {
#         name: np.asarray(indices, dtype=np.int32)
#         for name, indices in model_bundle['modality_presence_indices'].items()
#     }
    
#     probabilities = np.zeros(raw_feature_matrix.shape[0], dtype=np.float32)
#     for row_index, raw_feature_vector in enumerate(raw_feature_matrix):
#         processed_feature_vector = processed_feature_matrix[row_index]
#         modality_probabilities = []
#         for modality in ENSEMBLE_MODALITIES:
#             if modality not in models or feature_indices[modality].size == 0:
#                 continue
#             if _has_modality_signal(raw_feature_vector, modality_presence_indices[modality]):
#                 modality_vector = processed_feature_vector[feature_indices[modality]].reshape(1, -1)
#                 modality_probability = models[modality].predict_proba(modality_vector)[0][1]
#                 modality_probabilities.append(float(modality_probability))

#         if modality_probabilities:
#             probabilities[row_index] = float(np.mean(modality_probabilities))
#         else:
#             all_features = processed_feature_vector[feature_indices['all']].reshape(1, -1)
#             probabilities[row_index] = float(models['all'].predict_proba(all_features)[0][1])

#     return probabilities """

# def predict_ensemble_probabilities(model_bundle, feature_matrix):
#     raw_feature_matrix, processed_feature_matrix = prepare_feature_matrix(
#         feature_matrix,
#         preprocessor=model_bundle.get('preprocessor'),
#     )

#     models = model_bundle['models']
#     feature_indices = {
#         name: np.asarray(indices, dtype=np.int32)
#         for name, indices in model_bundle['feature_indices'].items()
#     }
#     modality_presence_indices = {
#         name: np.asarray(indices, dtype=np.int32)
#         for name, indices in model_bundle['modality_presence_indices'].items()
#     }
    
#     probabilities = np.zeros(raw_feature_matrix.shape[0], dtype=np.float32)
#     for row_index, raw_feature_vector in enumerate(raw_feature_matrix):
#         processed_feature_vector = processed_feature_matrix[row_index]
#         modality_probabilities = []
        
#         # 1. El código recorre las modalidades específicas: 'resp', 'eeg', 'ecg'
#         for modality in ENSEMBLE_MODALITIES:
#             if modality not in models or feature_indices[modality].size == 0:
#                 continue
#             if _has_modality_signal(raw_feature_vector, modality_presence_indices[modality]):
#                 modality_vector = processed_feature_vector[feature_indices[modality]].reshape(1, -1)
#                 modality_probability = models[modality].predict_proba(modality_vector)[0][1]
#                 modality_probabilities.append(float(modality_probability))

#         # 2. NUEVA LÓGICA DE DECISIÓN DE INFERENCIA:
#         num_modalities_present = len(modality_probabilities)

#         # Si el paciente tiene las 3 señales médicas completas, usamos el modelo global 'all'
#         if num_modalities_present == len(ENSEMBLE_MODALITIES):
#             all_features = processed_feature_vector[feature_indices['all']].reshape(1, -1)
#             probabilities[row_index] = float(models['all'].predict_proba(all_features)[0][1])

#         # Si le falta alguna señal (pero tiene al menos una), promediamos los modelos expertos disponibles
#         elif num_modalities_present > 0:
#             probabilities[row_index] = float(np.mean(modality_probabilities))

#         # Si no tiene ninguna señal médica (solo demografía), usamos 'all' como última defensa
#         else:
#             all_features = processed_feature_vector[feature_indices['all']].reshape(1, -1)
#             probabilities[row_index] = float(models['all'].predict_proba(all_features)[0][1])

#     return probabilities

# #define el umbral de decisión para convertir probabilidades en etiquetas binarias. Si la probabilidad es mayor o igual al umbral, se asigna la etiqueta 1 (positivo), de lo contrario, se asigna la etiqueta 0 (negativo).
# def predict_ensemble_labels(model_bundle, feature_matrix):
#     threshold = float(model_bundle.get('threshold', DEFAULT_ENSEMBLE_THRESHOLD))
#     probabilities = predict_ensemble_probabilities(model_bundle, feature_matrix)
#     labels = (probabilities >= threshold).astype(np.int32)
#     return labels, probabilities


# def train_multimodal_ensemble(data_folder, verbose, csv_path, export_folder=None):
#     patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
#     patient_metadata_list = find_patients(patient_data_file)
#     demographics_cache, diagnosis_cache = build_training_metadata_cache(patient_data_file)
#     num_records = len(patient_metadata_list)

#     if num_records == 0:
#         raise FileNotFoundError('No data were provided.')

#     features = []
#     labels = []
#     metadata_rows = []

#     with ThreadPoolExecutor(max_workers=MAX_TRAIN_WORKERS) as executor:
#         results = executor.map(
#             lambda record: process_training_record(
#                 record,
#                 data_folder,
#                 demographics_cache,
#                 diagnosis_cache,
#                 csv_path,
#             ),
#             patient_metadata_list,
#         )

#         pbar = tqdm(results, total=num_records, desc='Extracting Features', unit='record', disable=not verbose)
#         for metadata, feature_vector, label, message in pbar:
#             if verbose:
#                 pbar.set_postfix({'patient': metadata['patient_id']})

#             if message is not None:
#                 tqdm.write(f"  ! {message}")
#                 continue

#             features.append(feature_vector)
#             labels.append(label)
#             metadata_rows.append(metadata)

#         pbar.close()

#     features = np.asarray(features, dtype=np.float32)
#     labels = np.asarray(labels, dtype=np.int32)

#     if features.size == 0 or features.ndim != 2 or features.shape[0] == 0:
#         raise ValueError('No valid training samples were extracted. Review feature extraction logs for the skipped records.')

#     feature_names = list(get_feature_names())
#     feature_indices = get_feature_group_indices(include_demographics=True)
#     modality_presence_indices = get_feature_group_indices(include_demographics=False)

#     categorical_indices = [
#         i for i, name in enumerate(feature_names)
#         if name.lower() in ('sex', 'gender')
#     ]
#     site_groups = np.asarray([
#         normalize_site_group(metadata_row['site_id'])
#         for metadata_row in metadata_rows
#     ])
#     cv_config = CrossValidationConfig(
#         use_site_grouped_cv=USE_SITE_GROUPED_CV,
#         optimize_hyperparameter_search=OPTIMIZE_HYPERPARAMETER_SEARCH,
#         outer_random_splits=RANDOM_CV_N_SPLITS,
#         random_state=CV_RANDOM_STATE,
#         search_iterations=CV_SEARCH_ITERATIONS,
#         fixed_hyperparameters=DEFAULT_CV_HYPERPARAMETERS,
#     )
#     cv_runner = EnsembleCrossValidator(
#         config=cv_config,
#         param_dist=PARAM_DIST,
#         default_threshold=DEFAULT_ENSEMBLE_THRESHOLD,
#         build_preprocessor=build_preprocessor,
#         build_search_model=_build_search_model,
#         fit_ensemble=_fit_ensemble,
#         predict_probabilities=predict_ensemble_probabilities,
#     )
#     # --- [NUEVO FLUJO]: Ajuste global y único de variables para el CV y Modelo Final ---
#     print(f"  Categorical feature indices: {categorical_indices} "
#           f"({[feature_names[i] for i in categorical_indices]})")
#     print(f"  Hospital CV groups: {sorted(np.unique(site_groups).tolist())}")
#     print(f"  CV strategy: {'grouped by hospital' if cv_config.use_site_grouped_cv else 'random stratified folds'}")
#     print(f"  Hyperparameter search: {'enabled' if cv_config.optimize_hyperparameter_search else 'disabled'}")
    
#     # 1. AJUSTE GLOBAL ÚNICO: Ajustamos el preprocesador de correlación una sola vez AQUÍ afuera
#     print("\n[INFO] Fijando variables de forma global antes del CV...")
#     preprocessor = build_preprocessor(len(labels), categorical_indices if categorical_indices else None)
    
#     # Transformamos la matriz completa de entrenamiento
#     processed_features = np.asarray(preprocessor.fit_transform(features), dtype=np.float32)
    
#     # Remapeamos los índices de los grupos del ensamble una sola vez
#     selected_feature_indices = preprocessor.transform_feature_indices(feature_indices)
#     processed_feature_names = get_processed_feature_names(feature_names, preprocessor=preprocessor)
    
#     # Recuperamos los índices reales que sobrevivieron de las materias primas
#     real_selected_num_indices = preprocessor._numerical_indices[preprocessor.selector.selected_indices_]
#     selected_raw_feature_indices = np.concatenate([real_selected_num_indices, preprocessor.categorical_indices]).astype(np.int32)
    
#     print(f"Correlation selector: kept {len(processed_feature_names)}/{len(feature_names)} features fijas para todo el experimento.")

#     # 2. CONFIGURACIÓN DEL CV CON DATOS YA FILTRADOS:
#     # Pasamos una versión modificada del preprocesador al CV para que NO haga fit_transform interno, 
#     # sino que actúe meramente como un pasatopos (Identity o None) si tu clase lo requiere.
#     cv_runner = EnsembleCrossValidator(
#         config=cv_config,
#         param_dist=PARAM_DIST,
#         default_threshold=DEFAULT_ENSEMBLE_THRESHOLD,
#         build_preprocessor=lambda n, c: None, # <--- Truco: Le decimos al CV que no monte preprocesadores internos
#         build_search_model=_build_search_model,
#         fit_ensemble=_fit_ensemble,
#         predict_probabilities=predict_ensemble_probabilities,
#     )

#     # --- Step 1: Nested CV para calibración con variables 100% fijas ---
#     print("Running nested CV for hyperparameter consensus (Variables Fijas)...")
#     cv_result = cv_runner.run(
#         processed_features, # <--- ¡OJO! Le pasamos las características YA PREPROCESADAS fijas
#         labels,
#         selected_feature_indices, # <--- Índices ya remapeados fijos
#         modality_presence_indices=selected_feature_indices, # Alineado con las sobrevivientes
#         categorical_indices=None, # Ya fueron procesadas y pegadas, no hace falta que el CV las aísle de nuevo
#         site_groups=site_groups,
#     )
#     threshold = cv_result.threshold
#     consensus = cv_result.consensus_params
#     cv_metrics = cv_result.metrics
 
#     # --- Step 2: Fit final models on ALL data usando el consenso ---
#     print("Fitting final ensemble on all training data with consensus hyperparameters...")
#     # Como ya tenemos processed_features listas de arriba, entrenamos directo:
#     models = _fit_ensemble(processed_features, labels, selected_feature_indices, consensus_params=consensus)

#     # El resto del código de exportación de archivos CSV se mantiene igual...

#     # 5. Exportar métricas e información de diagnóstico a los CSV
#     export_root = export_folder or os.path.join(os.getcwd(), 'feature_exports')
#     feature_exports = export_feature_views(
#         export_root,
#         'training',
#         metadata_rows,
#         features,
#         feature_names,
#         preprocessor=preprocessor,
#         labels=labels,
#     )
#     selected_features_csv = os.path.join(export_root, 'training_features_selected.csv')
#     export_selected_features_csv(
#         selected_features_csv,
#         feature_names,
#         selected_raw_feature_indices,
#         modality_presence_indices,
#     )
#     feature_exports['selected'] = selected_features_csv      
#     return {
#         'type': 'multimodal_xgb_ensemble',
#         'threshold': threshold,
#         'feature_names': feature_names,
#         'processed_feature_names': processed_feature_names,
#         'selected_raw_feature_indices': selected_raw_feature_indices.tolist(),
#         'feature_indices': {
#             name: indices.tolist()
#             for name, indices in selected_feature_indices.items()
#             if name in {'all', 'resp', 'eeg', 'ecg'}
#         },
#         'modality_presence_indices': {
#             modality: modality_presence_indices[modality].tolist()
#             for modality in ENSEMBLE_MODALITIES
#         },
#         'models': models,
#         'preprocessor': preprocessor,
#         'feature_exports': feature_exports,
#         'cv_metrics': cv_metrics,
#     }

import os
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier

from helper_code import DEMOGRAPHICS_FILE, HEADERS, find_patients, load_label

from .config import (
    CV_RANDOM_STATE,
    CV_SEARCH_ITERATIONS,
    DEFAULT_CV_HYPERPARAMETERS,
    MAX_TRAIN_WORKERS,
    OPTIMIZE_HYPERPARAMETER_SEARCH,
    RANDOM_CV_N_SPLITS,
    USE_SITE_GROUPED_CV,
)
from .cross_validation import CrossValidationConfig, EnsembleCrossValidator, normalize_site_group
from .features import get_feature_group_indices, get_feature_names, get_or_create_record_feature_vector
from .preprocessing import build_preprocessor, get_processed_feature_names, remap_feature_indices


DEFAULT_ENSEMBLE_THRESHOLD = 0.5
ENSEMBLE_MODALITIES = ('resp', 'eeg', 'ecg')

# Métrica oficial de evaluación condicionada por edad
def compute_auroc_age(labels, predictions, ages, gap=2):
    m = len(labels)
    n = len(predictions)
    o = len(ages)
    assert(m == n == o)

    idx_pos = [i for i in range(m) if labels[i] == 1]
    idx_neg = [i for i in range(m) if labels[i] == 0]
    num_pos = len(idx_pos)
    num_neg = len(idx_neg)
    
    numer = 0
    denom = 0
    for i in range(num_pos):
        for j in range(num_neg):
            if abs(ages[idx_pos[i]] - ages[idx_neg[j]]) <= gap:
                if predictions[idx_pos[i]] > predictions[idx_neg[j]]:
                    numer += 1
                elif predictions[idx_pos[i]] == predictions[idx_neg[j]]:
                    numer += 0.5
                denom += 1
    return numer / denom if denom > 0 else 0.0

# Inicializamos la variable global
AGE_FEATURE_INDEX = 0


def _custom_auroc_age_metric(preds, dtrain):
  labels = dtrain.get_label()
  data = dtrain.get_data()

  # Si data es una matriz dispersa de scipy, convertimos la columna requerida a numpy array
  if hasattr(data, 'toarray'):
    ages = data[:, AGE_FEATURE_INDEX].toarray().ravel()
  else:
    ages = data[:, AGE_FEATURE_INDEX].ravel()

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
        feature_vector = get_or_create_record_feature_vector(
            record,
            data_folder,
            patient_data,
            csv_path=csv_path,
            require_physiological_data=True,
        )

        label = diagnosis_cache.get(patient_id)
        metadata = {
            'patient_id': patient_id,
            'site_id': site_id,
            'session_id': session_id,
        }

        if label == 0 or label == 1:
            return metadata, feature_vector, label, None

        return metadata, None, None, f"Invalid label for {patient_id}. Skipping..."

    except FileNotFoundError as exc:
        return {
            'patient_id': patient_id,
            'site_id': site_id,
            'session_id': session_id,
        }, None, None, f"{exc} Skipping..."
    except Exception as exc:
        return {
            'patient_id': patient_id,
            'site_id': site_id,
            'session_id': session_id,
        }, None, None, f"Error processing {patient_id}: {exc}"

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

def _fit_model(feature_matrix, labels, consensus_params=None):
    model = _build_xgb_model(labels, extra_params=consensus_params) 
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
#Entrena múltiples modelos XGBoost en paralelo: uno global (all) con todas las variables y uno independiente por cada señal médica disponible (resp, eeg, ecg)
def _fit_ensemble(feature_matrix, labels, feature_indices, consensus_params=None):
    models = {}
    if feature_indices['all'].size == 0:
        raise ValueError('Correlation selector removed all features for the global model.')

    models['all'] = _fit_model(
        feature_matrix[:, feature_indices['all']], labels, consensus_params
    )
    for modality in ENSEMBLE_MODALITIES:
        if feature_indices[modality].size == 0:
            continue
        models[modality] = _fit_model(
            feature_matrix[:, feature_indices[modality]], labels, consensus_params
        )
    return models


def _has_modality_signal(feature_vector, modality_presence_indices):
    modality_values = feature_vector[modality_presence_indices]
    return bool(np.any(np.isfinite(modality_values)))

def predict_ensemble_probabilities(model_bundle, feature_matrix):
    raw_feature_matrix, processed_feature_matrix = prepare_feature_matrix(
        feature_matrix,
        preprocessor=model_bundle.get('preprocessor'),
    )

    models = model_bundle['models']
    feature_indices = {
        name: np.asarray(indices, dtype=np.int32)
        for name, indices in model_bundle['feature_indices'].items()
    }
    modality_presence_indices = {
        name: np.asarray(indices, dtype=np.int32)
        for name, indices in model_bundle['modality_presence_indices'].items()
    }
    
    probabilities = np.zeros(raw_feature_matrix.shape[0], dtype=np.float32)
    for row_index, raw_feature_vector in enumerate(raw_feature_matrix):
        processed_feature_vector = processed_feature_matrix[row_index]
        modality_probabilities = []
        
        # 1. El código recorre las modalidades específicas: 'resp', 'eeg', 'ecg'
        for modality in ENSEMBLE_MODALITIES:
            if modality not in models or feature_indices[modality].size == 0:
                continue
            if _has_modality_signal(raw_feature_vector, modality_presence_indices[modality]):
                modality_vector = processed_feature_vector[feature_indices[modality]].reshape(1, -1)
                modality_probability = models[modality].predict_proba(modality_vector)[0][1]
                modality_probabilities.append(float(modality_probability))

        # 2. NUEVA LÓGICA DE DECISIÓN DE INFERENCIA:
        num_modalities_present = len(modality_probabilities)

        # Si el paciente tiene las 3 señales médicas completas, usamos el modelo global 'all'
        if num_modalities_present == len(ENSEMBLE_MODALITIES):
            all_features = processed_feature_vector[feature_indices['all']].reshape(1, -1)
            probabilities[row_index] = float(models['all'].predict_proba(all_features)[0][1])

        # Si le falta alguna señal (pero tiene al menos una), promediamos los modelos expertos disponibles
        elif num_modalities_present > 0:
            probabilities[row_index] = float(np.mean(modality_probabilities))

        # Si no tiene ninguna señal médica (solo demografía), usamos 'all' como última defensa
        else:
            all_features = processed_feature_vector[feature_indices['all']].reshape(1, -1)
            probabilities[row_index] = float(models['all'].predict_proba(all_features)[0][1])

    return probabilities

#define el umbral de decisión para convertir probabilidades en etiquetas binarias. Si la probabilidad es mayor o igual al umbral, se asigna la etiqueta 1 (positivo), de lo contrario, se asigna la etiqueta 0 (negativo).
def predict_ensemble_labels(model_bundle, feature_matrix):
    threshold = float(model_bundle.get('threshold', DEFAULT_ENSEMBLE_THRESHOLD))
    probabilities = predict_ensemble_probabilities(model_bundle, feature_matrix)
    labels = (probabilities >= threshold).astype(np.int32)
    return labels, probabilities


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

    with ThreadPoolExecutor(max_workers=MAX_TRAIN_WORKERS) as executor:
        results = executor.map(
            lambda record: process_training_record(
                record,
                data_folder,
                demographics_cache,
                diagnosis_cache,
                csv_path,
            ),
            patient_metadata_list,
        )

        pbar = tqdm(results, total=num_records, desc='Extracting Features', unit='record', disable=not verbose)
        for metadata, feature_vector, label, message in pbar:
            if verbose:
                pbar.set_postfix({'patient': metadata['patient_id']})

            if message is not None:
                tqdm.write(f"  ! {message}")
                continue

            features.append(feature_vector)
            labels.append(label)
            metadata_rows.append(metadata)

        pbar.close()

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
        use_site_grouped_cv=USE_SITE_GROUPED_CV,
        optimize_hyperparameter_search=OPTIMIZE_HYPERPARAMETER_SEARCH,
        outer_random_splits=RANDOM_CV_N_SPLITS,
        random_state=CV_RANDOM_STATE,
        search_iterations=CV_SEARCH_ITERATIONS,
        fixed_hyperparameters=DEFAULT_CV_HYPERPARAMETERS,
    )
    cv_runner = EnsembleCrossValidator(
        config=cv_config,
        param_dist=PARAM_DIST,
        default_threshold=DEFAULT_ENSEMBLE_THRESHOLD,
        build_preprocessor=build_preprocessor,
        build_search_model=_build_search_model,
        fit_ensemble=_fit_ensemble,
        predict_probabilities=predict_ensemble_probabilities,
    )
    # --- [NUEVO FLUJO]: Ajuste global y único de variables para el CV y Modelo Final ---
    print(f"  Categorical feature indices: {categorical_indices} "
          f"({[feature_names[i] for i in categorical_indices]})")
    print(f"  Hospital CV groups: {sorted(np.unique(site_groups).tolist())}")
    print(f"  CV strategy: {'grouped by hospital' if cv_config.use_site_grouped_cv else 'random stratified folds'}")
    print(f"  Hyperparameter search: {'enabled' if cv_config.optimize_hyperparameter_search else 'disabled'}")
    
    # 1. AJUSTE GLOBAL ÚNICO: Ajustamos el preprocesador de correlación una sola vez AQUÍ afuera
    print("\n[INFO] Fijando variables de forma global antes del CV...")
    preprocessor = build_preprocessor(len(labels), categorical_indices if categorical_indices else None)
    
    # Transformamos la matriz completa de entrenamiento
    processed_features = np.asarray(preprocessor.fit_transform(features), dtype=np.float32)
    
    # Remapeamos los índices de los grupos del ensamble una sola vez
    selected_feature_indices = preprocessor.transform_feature_indices(feature_indices)
    processed_feature_names = get_processed_feature_names(feature_names, preprocessor=preprocessor)
    
    # Recuperamos los índices reales que sobrevivieron de las materias primas
    real_selected_num_indices = preprocessor._numerical_indices[preprocessor.selector.selected_indices_]
    selected_raw_feature_indices = np.concatenate([real_selected_num_indices, preprocessor.categorical_indices]).astype(np.int32)
    
    print(f"Correlation selector: kept {len(processed_feature_names)}/{len(feature_names)} features fijas para todo el experimento.")
    global AGE_FEATURE_INDEX
    AGE_FEATURE_INDEX = 0  # Valor por defecto
    for idx, name in enumerate(processed_feature_names):
      if 'age' in name.lower():
        AGE_FEATURE_INDEX = idx
        print(f"--> [INFO] Variable 'Age' encontrada en el índice: {idx}")
        break
    # 2. CONFIGURACIÓN DEL CV CON DATOS YA FILTRADOS:
    # Pasamos una versión modificada del preprocesador al CV para que NO haga fit_transform interno, 
    # sino que actúe meramente como un pasatopos (Identity o None) si tu clase lo requiere.
    cv_runner = EnsembleCrossValidator(
        config=cv_config,
        param_dist=PARAM_DIST,
        default_threshold=DEFAULT_ENSEMBLE_THRESHOLD,
        build_preprocessor=lambda n, c: None, # <--- Truco: Le decimos al CV que no monte preprocesadores internos
        build_search_model=_build_search_model,
        fit_ensemble=_fit_ensemble,
        predict_probabilities=predict_ensemble_probabilities,
    )

    # --- Step 1: Nested CV para calibración con variables 100% fijas ---
    print("Running nested CV for hyperparameter consensus (Variables Fijas)...")
    cv_result = cv_runner.run(
        processed_features, # <--- ¡OJO! Le pasamos las características YA PREPROCESADAS fijas
        labels,
        selected_feature_indices, # <--- Índices ya remapeados fijos
        modality_presence_indices=selected_feature_indices, # Alineado con las sobrevivientes
        categorical_indices=None, # Ya fueron procesadas y pegadas, no hace falta que el CV las aísle de nuevo
        site_groups=site_groups,
    )
    threshold = cv_result.threshold
    consensus = cv_result.consensus_params
    cv_metrics = cv_result.metrics
 
    # --- Step 2: Fit final models on ALL data usando el consenso ---
    print("Fitting final ensemble on all training data with consensus hyperparameters...")
    # Como ya tenemos processed_features listas de arriba, entrenamos directo:
    models = _fit_ensemble(processed_features, labels, selected_feature_indices, consensus_params=consensus)

    # 5. Exportar métricas e información de diagnóstico a los CSV
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
            for name, indices in selected_feature_indices.items()
            if name in {'all', 'resp', 'eeg', 'ecg'}
        },
        'modality_presence_indices': {
            modality: modality_presence_indices[modality].tolist()
            for modality in ENSEMBLE_MODALITIES
        },
        'models': models,
        'preprocessor': preprocessor,
        'feature_exports': feature_exports,
        'cv_metrics': cv_metrics,
    }