import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_CORRELATION_THRESHOLD

DEFAULT_KNN_NEIGHBORS = 5

#Crea una instancia del preprocesador principal de forma segura. Tiene un control para evitar que el número de vecinos (n_neighbors) para el algoritmo KNN sea mayor o igual al número total de muestras disponibles en el dataset (lo cual rompería el algoritmo).
def build_preprocessor(num_samples, categorical_indices=None):
    neighbors = min(DEFAULT_KNN_NEIGHBORS, max(1, num_samples - 1)) if num_samples > 1 else 1
    
    return CorrelationAwarePreprocessor(
        n_neighbors=neighbors,
        categorical_indices=categorical_indices,
        correlation_threshold=FEATURE_CORRELATION_THRESHOLD
    )

#Obtain feature names after preprocessing. If a preprocessor is provided, it will use its get_feature_names_out method to get the processed feature names. Otherwise, it will return the original feature names as a list.
def get_processed_feature_names(feature_names, preprocessor=None):
    if preprocessor is None:
        return list(feature_names)

    if hasattr(preprocessor, 'get_feature_names_out'):
        return list(preprocessor.get_feature_names_out(feature_names))

    return list(feature_names)

#Obtain feature indices after preprocessing. If a preprocessor is provided, it will use its transform_feature_indices method to get the processed feature indices. Otherwise, it will return the original feature indices as a dictionary.
def remap_feature_indices(preprocessor, feature_indices):
    if preprocessor is None or not hasattr(preprocessor, 'transform_feature_indices'):
        return {
            name: np.asarray(indices, dtype=np.int32)
            for name, indices in feature_indices.items()
        }

    return preprocessor.transform_feature_indices(feature_indices)

#El método fit calcula la matriz de correlación de Pearson entre todas las variables de X. Si dos variables superan el umbral estipulado (threshold), añade la segunda a una "lista negra" para descartarla, quedándose con los índices limpios en self.selected_indices_.
class CorrelationThresholdSelector:
    def __init__(self, threshold):
        self.threshold = float(threshold)
        self.selected_indices_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X.shape

        if n_features == 0:
            self.selected_indices_ = np.array([], dtype=np.int32)
            return self

        if n_samples < 2 or n_features == 1:
            self.selected_indices_ = np.arange(n_features, dtype=np.int32)
            return self

        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.corrcoef(X, rowvar=False)
        corr = np.asarray(corr, dtype=np.float32)
        corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)

        keep_mask = np.ones(n_features, dtype=bool)
        for index in range(n_features):
            if not keep_mask[index]:
                continue

            correlated_indices = np.where(corr[index, index + 1:] > self.threshold)[0]
            if correlated_indices.size:
                keep_mask[correlated_indices + index + 1] = False

        if not np.any(keep_mask):
            keep_mask[0] = True

        self.selected_indices_ = np.flatnonzero(keep_mask).astype(np.int32)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.selected_indices_ is None:
            raise ValueError('Correlation selector has not been fitted.')
        return X[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        if self.selected_indices_ is None:
            raise ValueError('Correlation selector has not been fitted.')

        if input_features is None:
            input_features = [f'feature_{index}' for index in self.selected_indices_]

        return np.asarray([input_features[index] for index in self.selected_indices_], dtype=object)
    
#Separa mediante índices qué columnas son numéricas y cuáles categóricas para asegurarse de aplicar el StandardScaler únicamente a las numéricas a través del método _scale_numerical_columns
class CorrelationAwarePreprocessor:
    def __init__(self, n_neighbors, categorical_indices, correlation_threshold):
        self.n_neighbors = n_neighbors
        if categorical_indices is None:
            self.categorical_indices = np.array([], dtype=np.int32)
        else:
            self.categorical_indices = np.asarray(categorical_indices, dtype=np.int32)
        self.imputer = KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
        self.scaler = StandardScaler()
        self.selector = CorrelationThresholdSelector(correlation_threshold)
        self._numerical_indices = np.array([], dtype=np.int32)
        # Guardaremos las modas calculadas en el fit para usarlas en el transform
        self._categorical_modes = {} 

    def _get_numerical_indices(self, n_features):
        all_idx = np.arange(n_features, dtype=np.int32)
        return np.setdiff1d(all_idx, self.categorical_indices)

    def _scale_numerical_columns(self, X_imputed, fit=False):
        X_out = X_imputed.copy()
        if self._numerical_indices.size == 0:
            return X_out

        X_num = X_imputed[:, self._numerical_indices]
        if fit:
            X_num_scaled = np.asarray(self.scaler.fit_transform(X_num), dtype=np.float32)
        else:
            X_num_scaled = np.asarray(self.scaler.transform(X_num), dtype=np.float32)

        X_out[:, self._numerical_indices] = X_num_scaled
        return X_out

    def fit_transform(self, X):
        X = np.asarray(X, dtype=np.float32).copy()
        X[~np.isfinite(X)] = np.nan
        
        self._numerical_indices = self._get_numerical_indices(X.shape[1])
        X_out = X.copy()

        # 1. Imputación de Categóricas usando la Moda (Most Frequent) sin alterar nombres
        for idx in self.categorical_indices:
            col_data = X[:, idx]
            valid_vals = col_data[np.isnan(col_data) == False]
            if valid_vals.size > 0:
                # Calcular la moda con numpy
                vals, counts = np.unique(valid_vals, return_counts=True)
                mode_val = vals[np.argmax(counts)]
            else:
                mode_val = 0.0 # Valor por defecto si toda la columna es NaN
            self._categorical_modes[idx] = mode_val
            X_out[np.isnan(col_data), idx] = mode_val

        # 2. Imputación KNN aplicada EXCLUSIVAMENTE a las continuas
        if self._numerical_indices.size > 0:
            X_num = X[:, self._numerical_indices]
            X_num_imputed = np.asarray(self.imputer.fit_transform(X_num), dtype=np.float32)
            X_out[:, self._numerical_indices] = X_num_imputed

        # 3. Escalado exclusivo de continuas
        X_out = self._scale_numerical_columns(X_out, fit=True)

        # 4. Ajustar el selector de correlación SOLAMENTE con las continuas
        if self._numerical_indices.size > 0:
            X_num_scaled = X_out[:, self._numerical_indices]
            self.selector.fit(X_num_scaled)
        else:
            # Si no hay numéricas, el selector selecciona todo (vacío)
            self.selector.selected_indices_ = np.array([], dtype=np.int32)

        # 5. Retornar las Continuas Seleccionadas + TODAS las Categóricas (sin filtrar)
        return self._combine_outputs(X_out)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32).copy()
        X[~np.isfinite(X)] = np.nan
        X_out = X.copy()

        # 1. Aplicar la moda guardada a las categóricas
        for idx in self.categorical_indices:
            col_data = X[:, idx]
            mode_val = self._categorical_modes.get(idx, 0.0)
            X_out[np.isnan(col_data), idx] = mode_val

        # 2. Aplicar el KNN transform a las continuas
        if self._numerical_indices.size > 0:
            X_num = X[:, self._numerical_indices]
            X_num_imputed = np.asarray(self.imputer.transform(X_num), dtype=np.float32)
            X_out[:, self._numerical_indices] = X_num_imputed

        # 3. Aplicar el escalado a las continuas
        X_out = self._scale_numerical_columns(X_out, fit=False)

        # 4. Combinar las continuas filtradas y las categóricas
        return self._combine_outputs(X_out)

    def _combine_outputs(self, X_out):
        # Mapea los índices internos elegidos por el selector a los índices reales de la matriz original X
        real_selected_num_indices = self._numerical_indices[self.selector.selected_indices_]
        
        # Combinamos los índices de las numéricas que sobrevivieron junto a todas las categóricas
        final_indices = np.concatenate([real_selected_num_indices, self.categorical_indices]).astype(np.int32)
        return X_out[:, final_indices]

    def transform_feature_indices(self, feature_indices):
        if self.selector.selected_indices_ is None:
            raise ValueError('Preprocessor has not been fitted.')

        real_selected_num_indices = self._numerical_indices[self.selector.selected_indices_]
        final_indices = np.concatenate([real_selected_num_indices, self.categorical_indices]).astype(np.int32)

        index_lookup = {
            int(raw_index): int(processed_index)
            for processed_index, raw_index in enumerate(final_indices)
        }
        remapped = {}
        for name, indices in feature_indices.items():
            kept_indices = [
                index_lookup[int(raw_index)]
                for raw_index in np.asarray(indices, dtype=np.int32)
                if int(raw_index) in index_lookup
            ]
            remapped[name] = np.asarray(kept_indices, dtype=np.int32)
        return remapped

    def get_feature_names_out(self, input_features=None):
        if self.selector.selected_indices_ is None:
            raise ValueError('Preprocessor has not been fitted.')
        
        if input_features is None:
            return None

        real_selected_num_indices = self._numerical_indices[self.selector.selected_indices_]
        final_indices = np.concatenate([real_selected_num_indices, self.categorical_indices]).astype(np.int32)
        
        return np.asarray([input_features[index] for index in final_indices], dtype=object)
    #Cambios:
    #Aislamiento de la Correlación: El CorrelationThresholdSelector ahora trabaja solo dentro del sub-pipeline numérico (numeric_sub_pipeline). La matriz de correlación de Pearson se calcula única y exclusivamente con los datos continuos escalados. Las variables categóricas quedan completamente a salvo y no se eliminan por este criterio.
    # Imputación Segmentada por Tipo de Dato: * Las numéricas van al KNNImputer (donde la distancia euclídea ahora sí tiene sentido matemático).Las categóricas se desvían al SimpleImputer(strategy='most_frequent'), calculando la moda de forma independiente.
