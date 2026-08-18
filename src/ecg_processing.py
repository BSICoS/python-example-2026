from .lib.ecg_features import compute_ecg_features
import numpy as np
from src.common.channel_utils import normalize_channel_label
from src.resp_processing import select_best_respiration_signal

ECG_KEYWORDS = ['ecg', 'ekg']

ECG_SEGMENT_FEATURE_NAMES = [
    "MHR",
    "SDNN",
    "RMSSD",
    "PNN50",
    "LF",
    "HF_RESP",
    "LFN_RESP",
    "URLF",
    "RE",
    "R",
    "ECGage",
]
ECG_SEGMENT_FEATURE_LENGTH = len(ECG_SEGMENT_FEATURE_NAMES)
ECG_FEATURE_NAMES = ECG_SEGMENT_FEATURE_NAMES
ECG_FEATURE_LENGTH = ECG_SEGMENT_FEATURE_LENGTH
_RESPIRATION_NOT_PROVIDED = object()

def _find_ecg_channel(physiological_data):
    for label in physiological_data.keys():
        label_clean = normalize_channel_label(label)
        if any(keyword in label_clean for keyword in ECG_KEYWORDS):
            return label
    return None

def processECG(
    physiological_data,
    physiological_fs,
    csv_path,
    *,
    selected_respiration=_RESPIRATION_NOT_PROVIDED,
):
    results = np.full(ECG_SEGMENT_FEATURE_LENGTH, np.nan, dtype=np.float32)

    ecg_label = _find_ecg_channel(physiological_data)

    if ecg_label is None:
        return results  # no ECG found

    if ecg_label not in physiological_fs:
        return results

    ecg_signal = np.asarray(physiological_data[ecg_label], dtype=float)
    fs = physiological_fs[ecg_label]

    if ecg_signal.size == 0:
        return results

    if selected_respiration is _RESPIRATION_NOT_PROVIDED:
        selected_respiration = select_best_respiration_signal(
            physiological_data,
            physiological_fs,
            csv_path,
        )

    try:
        values = compute_ecg_features(
            ecg_signal,
            fs,
            ECG_SEGMENT_FEATURE_LENGTH,
            respiration_signal=(
                selected_respiration.resampled_signal
                if selected_respiration is not None
                else None
            ),
            respiration_sampling_frequency=(
                selected_respiration.resampled_frequency
                if selected_respiration is not None
                else None
            ),
        )

        if values is None or len(values) == 0:
            return results

        values = np.asarray(values, dtype=np.float32)
        values[~np.isfinite(values)] = np.nan

        if len(values) >= ECG_SEGMENT_FEATURE_LENGTH:
            results[:] = values[:ECG_SEGMENT_FEATURE_LENGTH]
        else:
            results[:len(values)] = values

    except Exception:
        pass

    return results
