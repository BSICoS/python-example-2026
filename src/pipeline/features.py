import os

import edfio
import joblib
import numpy as np
import pandas as pd

from src.common.channel_utils import normalize_channel_label
from helper_code import HEADERS, PHYSIOLOGICAL_DATA_SUBFOLDER, load_age, load_sex, load_signal_data
from src.ecg_processing import ECG_FEATURE_LENGTH, ECG_FEATURE_NAMES, ECG_KEYWORDS, processECG
from src.eeg_processing import (
    EEG_CHANNEL_SPECS, EEG_FEATURE_LENGTH, EEG_FEATURE_NAMES,
    EEG_SEGMENT_FEATURE_NAMES, processEEG, _get_eeg_aliases,
)
from src.resp_processing import (
    _get_resp_alias_groups,
    select_best_respiration_signal,
)

from .config import (
    DEFAULT_CSV_PATH,
    FEATURE_CACHE_FOLDER_NAME,
    SEGMENT_AGGREGATION_NAMES,
    SCRIPT_DIR,
    SEGMENT_DURATION_SECONDS,
    SEGMENT_STRIDE_SECONDS,
)

REQUIRED_SIGNAL_ALIASES_CACHE = {}
DEMOGRAPHIC_FEATURE_NAMES = (
    'Age',
    'Sex',
)
def _build_aggregated_feature_names(segment_feature_names):
    names = []
    for feature_name in segment_feature_names:
        aggregations = ('Median',) if feature_name == 'ECGage' else SEGMENT_AGGREGATION_NAMES
        for aggregation_name in aggregations:
            names.append(f'{feature_name}_{aggregation_name}')
    
    # Add the difference feature if ECGage is present in this group
    if 'ECGage' in segment_feature_names:
        names.append('ECGage_Age_Diff')
        
    return tuple(names)

FEATURE_NAME_GROUPS = {
    'demographics': DEMOGRAPHIC_FEATURE_NAMES,
    'eeg': EEG_FEATURE_NAMES,
    'ecg': _build_aggregated_feature_names(ECG_FEATURE_NAMES),
}
FEATURE_NAMES = (
    *FEATURE_NAME_GROUPS['demographics'],
    *FEATURE_NAME_GROUPS['eeg'],
    *FEATURE_NAME_GROUPS['ecg'],
)

SEGMENT_AGGREGATION_FUNCTIONS = {
    'Max': lambda values: float(np.percentile(values, 90)),
    'Min': lambda values: float(np.percentile(values, 10)),
    'Median': lambda values: float(np.median(values)),
    'IQR': lambda values: float(np.percentile(values, 75) - np.percentile(values, 25)),
}

def _get_feature_cache_root(data_folder):
    return os.path.join(SCRIPT_DIR, FEATURE_CACHE_FOLDER_NAME)

def _coerce_feature_vector(features):
    vector = np.asarray(features, dtype=np.float32).reshape(-1)
    vector[~np.isfinite(vector)] = np.nan
    return vector

def _extract_optional_features(extractor, expected_length, *args, **kwargs):
    vector = _coerce_feature_vector(extractor(*args, **kwargs))
    if vector.size != expected_length:
        raise ValueError(
            f"{extractor.__name__} returned {vector.size} features; expected {expected_length}."
        )
    return vector

def _get_physiological_data_file(data_folder, site_id, patient_id, session_id):
    return os.path.join(
        data_folder,
        PHYSIOLOGICAL_DATA_SUBFOLDER,
        site_id,
        f"{patient_id}_ses-{session_id}.edf",
    )
def _get_required_signal_aliases(csv_path):
    normalized_csv_path = os.path.abspath(csv_path)
    required_aliases = REQUIRED_SIGNAL_ALIASES_CACHE.get(normalized_csv_path)
    if required_aliases is not None:
        return required_aliases

    resp_alias_groups = _get_resp_alias_groups(normalized_csv_path)
    eeg_aliases = _get_eeg_aliases(normalized_csv_path)

    required_aliases = set()
    for aliases in resp_alias_groups.values():
        required_aliases.update(aliases)

    for channel_spec in EEG_CHANNEL_SPECS.values():
        required_aliases.update(eeg_aliases.get(normalize_channel_label(channel_spec['direct']), set()))
        required_aliases.update(eeg_aliases.get(normalize_channel_label(channel_spec['positive']), set()))
        required_aliases.update(eeg_aliases.get(normalize_channel_label(channel_spec['reference']), set()))

    REQUIRED_SIGNAL_ALIASES_CACHE[normalized_csv_path] = required_aliases
    return required_aliases

def _load_required_signal_data(edf_path, csv_path):
    required_aliases = _get_required_signal_aliases(csv_path)
    channel_dict = {}
    fs_dict = {}

    try:
        edf = edfio.read_edf(edf_path, lazy_load_data=True)
    except Exception:
        return load_signal_data(edf_path)

    for signal in edf.signals:
        label = signal.label.lower().strip()
        normalized_label = normalize_channel_label(label)
        is_required_signal = normalized_label in required_aliases
        is_ecg_signal = any(keyword in normalized_label for keyword in ECG_KEYWORDS)

        if not is_required_signal and not is_ecg_signal:
            continue

        fs_dict[label] = float(signal.sampling_frequency)
        channel_dict[label] = signal.data

    if channel_dict:
        return channel_dict, fs_dict

    return load_signal_data(edf_path)

def _get_feature_cache_file(data_folder, site_id, patient_id, session_id):
    cache_dir = os.path.join(_get_feature_cache_root(data_folder), site_id)
    return os.path.join(cache_dir, f"{patient_id}_ses-{session_id}.sav")

def get_feature_export_dir(data_folder):
    return os.path.join(_get_feature_cache_root(data_folder), 'exports')

def _get_feature_cache_csv_file(cache_file):
    return f"{os.path.splitext(cache_file)[0]}.csv"

def _load_cached_feature_vector(cache_file):
    if not os.path.exists(cache_file):
        return None

    try:
        payload = joblib.load(cache_file)
    except Exception:
        return None

    if isinstance(payload, dict):
        payload = payload.get('features')

    if payload is None:
        return None

    vector = _coerce_feature_vector(payload)
    if vector.size != len(FEATURE_NAMES):
        return None
    return vector

def _save_cached_feature_vector(cache_file, feature_vector):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    temp_cache_file = f"{cache_file}.tmp"
    payload = _coerce_feature_vector(feature_vector)

    try:
        joblib.dump(payload, temp_cache_file, protocol=0)
        os.replace(temp_cache_file, cache_file)
    finally:
        if os.path.exists(temp_cache_file):
            os.remove(temp_cache_file)

    csv_file = _get_feature_cache_csv_file(cache_file)
    temp_csv_file = f"{csv_file}.tmp"
    
    try:
        pd.DataFrame([payload], columns=get_feature_names()).to_csv(temp_csv_file, index=False)
        os.replace(temp_csv_file, csv_file)
    finally:
        if os.path.exists(temp_csv_file):
            os.remove(temp_csv_file)

def extract_demographic_features(data):
    age_value = data.get(HEADERS['age'])
    try:
        age = float(age_value)
    except (TypeError, ValueError):
        age = np.nan
    if not np.isfinite(age):
        age = np.nan
    age = np.array([age], dtype=np.float32)

    sex = load_sex(data)
    if sex == 'Male':
        sex_value = 1.0
    elif sex == 'Female':
        sex_value = 0.0
    else:
        sex_value = np.nan
    sex_vec = np.array([sex_value], dtype=np.float32)

    return np.concatenate([age, sex_vec]).astype(np.float32)


def get_feature_names():
    return FEATURE_NAMES

def get_feature_group_indices(include_demographics=False):
    groups = {}
    start = len(FEATURE_NAME_GROUPS['demographics'])

    if include_demographics:
        demo_indices = np.arange(start, dtype=np.int32)
    else:
        demo_indices = np.array([], dtype=np.int32)

    for group_name in ('eeg', 'ecg'):
        group_length = len(FEATURE_NAME_GROUPS[group_name])
        group_indices = np.arange(start, start + group_length, dtype=np.int32)
        if include_demographics:
            groups[group_name] = np.concatenate([demo_indices, group_indices])
        else:
            groups[group_name] = group_indices
        start += group_length

    groups['all'] = np.arange(len(FEATURE_NAMES), dtype=np.int32)
    return groups

def _iter_signal_segments(physiological_data, physiological_fs):
    if not physiological_data:
        return []

    durations = []
    for label, signal in physiological_data.items():
        fs = physiological_fs.get(label)
        if fs is None or fs <= 0:
            continue
        durations.append(len(signal) / float(fs))

    if not durations:
        return []

    max_duration_seconds = max(durations)
    last_full_segment_start = max_duration_seconds - SEGMENT_DURATION_SECONDS
    if last_full_segment_start < 0:
        return []

    segment_starts = np.arange(0.0, last_full_segment_start + 1e-9, SEGMENT_STRIDE_SECONDS, dtype=float)
    segments = []

    for start_seconds in segment_starts:
        end_seconds = start_seconds + SEGMENT_DURATION_SECONDS
        segment_data = {}
        segment_fs = {}

        for label, signal in physiological_data.items():
            fs = physiological_fs.get(label)
            if fs is None or fs <= 0:
                continue

            start_index = int(round(start_seconds * fs))
            end_index = int(round(end_seconds * fs))
            if start_index >= len(signal) or end_index > len(signal):
                continue

            sliced_signal = np.asarray(signal[start_index:end_index], dtype=float)
            if sliced_signal.size == 0:
                continue

            segment_data[label] = sliced_signal
            segment_fs[label] = fs

        if segment_data:
            segments.append((segment_data, segment_fs))

    return segments

def _aggregate_segment_feature_vectors(feature_vectors, segment_feature_names):
    if not feature_vectors:
        # Calculate total length dynamically based on the features being processed
        total_len = sum(
            1 if name == 'ECGage' else len(SEGMENT_AGGREGATION_NAMES)
            for name in segment_feature_names
        )
        return np.full(total_len, np.nan, dtype=np.float32)

    matrix = np.asarray(feature_vectors, dtype=np.float32)
    aggregated_values = []

    for column_index, feature_name in enumerate(segment_feature_names):
        column_values = matrix[:, column_index]
        finite_values = column_values[np.isfinite(column_values)]

        # Determine which aggregations to apply for this feature
        aggregations_to_run = (
            ('Median',) if feature_name == 'ECGage' else SEGMENT_AGGREGATION_NAMES
        )

        if finite_values.size == 0:
            aggregated_values.extend([np.nan] * len(aggregations_to_run))
            continue

        aggregated_values.extend(
            SEGMENT_AGGREGATION_FUNCTIONS[aggregation_name](finite_values)
            for aggregation_name in aggregations_to_run
        )

    return np.asarray(aggregated_values, dtype=np.float32)

def _extract_segmented_features(extractor, segment_feature_names, physiological_data, physiological_fs, csv_path):
    expected_length = len(segment_feature_names)
    segments = _iter_signal_segments(physiological_data, physiological_fs)
    if not segments:
        aggregated_length = expected_length * len(SEGMENT_AGGREGATION_NAMES)
        return np.full(aggregated_length, np.nan, dtype=np.float32)

    segment_feature_vectors = []
    for segment_data, segment_fs in segments:
        try:
            vector = _extract_optional_features(
                extractor,
                expected_length,
                segment_data,
                segment_fs,
                csv_path=csv_path,
            )
        except Exception:
            continue

        if np.all(np.isnan(vector)):
            continue

        segment_feature_vectors.append(vector)

    return _aggregate_segment_feature_vectors(segment_feature_vectors, segment_feature_names)

def _extract_ecg_features(
    physiological_data,
    physiological_fs,
    csv_path,
):
    segments = _iter_signal_segments(physiological_data, physiological_fs)
    ecg_vectors = []

    for segment_data, segment_fs in segments:
        try:
            selected_respiration = select_best_respiration_signal(
                segment_data,
                segment_fs,
                csv_path,
            )
        except Exception:
            selected_respiration = None
        try:
            ecg_vector = _extract_optional_features(
                processECG,
                len(ECG_FEATURE_NAMES),
                segment_data,
                segment_fs,
                csv_path=csv_path,
                selected_respiration=selected_respiration,
            )
        except Exception:
            continue
        if not np.all(np.isnan(ecg_vector)):
            ecg_vectors.append(ecg_vector)

    return _aggregate_segment_feature_vectors(ecg_vectors, ECG_FEATURE_NAMES)

def extract_extended_physiological_features(physiological_data, physiological_fs, csv_path=DEFAULT_CSV_PATH):
    ecg_features = _extract_ecg_features(
        physiological_data,
        physiological_fs,
        csv_path,
    )
    eeg_background_features = _extract_segmented_features(
        processEEG,
        EEG_SEGMENT_FEATURE_NAMES,
        physiological_data,
        physiological_fs,
        csv_path,
    )
    # ECGage_Age_Diff is populated after demographics are joined to this vector.
    return np.hstack([eeg_background_features, ecg_features, np.nan]).astype(np.float32)

def _compute_record_feature_vector(patient_data, data_folder, site_id, patient_id, session_id, csv_path, require_physiological_data):
    demographic_features = extract_demographic_features(patient_data)
    chronological_age = demographic_features[0]

    physiological_data_file = _get_physiological_data_file(
        data_folder,
        site_id,
        patient_id,
        session_id,
    )
    total_physio_names_len = sum(len(FEATURE_NAME_GROUPS[g]) for g in ('eeg', 'ecg'))

    if os.path.exists(physiological_data_file):
        physiological_data, physiological_fs = _load_required_signal_data(physiological_data_file, csv_path)
        physiological_features = extract_extended_physiological_features(
            physiological_data,
            physiological_fs,
            csv_path=csv_path,
        )
    elif require_physiological_data:
        raise FileNotFoundError(f"Missing physiological data for {patient_id}.")
    else:
        physiological_features = np.full(total_physio_names_len, np.nan, dtype=np.float32)

    # Combine demographic and physiological features first
    combined = np.hstack([demographic_features, physiological_features]).astype(np.float32)

    # Fill in ECGage_Age_Diff within the final demographic/physiology schema.
    feature_names = get_feature_names()
    if 'ECGage_Age_Diff' in feature_names:
        diff_idx = feature_names.index('ECGage_Age_Diff')
        
        # Calculate diff if ECGage_Median is present
        if 'ECGage_Median' in feature_names:
            ecg_median_idx = feature_names.index('ECGage_Median')
            ecg_age_median = combined[ecg_median_idx]
            
            if np.isfinite(ecg_age_median) and np.isfinite(chronological_age):
                combined[diff_idx] = np.float32(ecg_age_median - chronological_age)
            else:
                combined[diff_idx] = np.nan

    if len(combined) != len(FEATURE_NAMES):
        if len(combined) > len(FEATURE_NAMES):
            combined = combined[:len(FEATURE_NAMES)]
        else:
            combined = np.pad(
                combined,
                (0, len(FEATURE_NAMES) - len(combined)),
                constant_values=np.nan,
            )
    return combined.astype(np.float32)

def get_or_create_record_feature_vector(
    record,
    data_folder,
    patient_data,
    csv_path=DEFAULT_CSV_PATH,
    require_physiological_data=True,
    return_cache_hit=False,
):
    patient_id = record[HEADERS['bids_folder']]
    site_id = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]
    cache_file = _get_feature_cache_file(data_folder, site_id, patient_id, session_id)
    cached_features = _load_cached_feature_vector(cache_file)
    if cached_features is not None:
        return (cached_features, True) if return_cache_hit else cached_features

    feature_vector = _compute_record_feature_vector(
        patient_data,
        data_folder,
        site_id,
        patient_id,
        session_id,
        csv_path,
        require_physiological_data,
    )
    _save_cached_feature_vector(cache_file, feature_vector)
    return (feature_vector, False) if return_cache_hit else feature_vector
