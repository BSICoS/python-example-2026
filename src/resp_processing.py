from dataclasses import dataclass

from .lib import resp_features
import numpy as np
from src.common.channel_utils import get_cached_channel_table, normalize_channel_label, split_channel_aliases
from src.common.signal_utils import resample_signal

RESP_CHANNEL_GROUPS = ("Abdomen", "Chest", "Nasal", "Flow")
RESP_SEGMENT_FEATURE_NAMES = [
    f"{group}_Peakedness"
    for group in RESP_CHANNEL_GROUPS
] + [
    "SpO2",
    "CET90",
    "ODI",
    "ODI_deepness",
]
RESP_SEGMENT_FEATURE_LENGTH = len(RESP_SEGMENT_FEATURE_NAMES)
RESP_FEATURE_NAMES = RESP_SEGMENT_FEATURE_NAMES
RESP_FEATURE_LENGTH = RESP_SEGMENT_FEATURE_LENGTH
RESP_ALIAS_GROUPS_CACHE = {}

@dataclass(frozen=True)
class SelectedRespiration:
    """Best direct respiration channel according to the feature pipeline."""

    label: str
    group: str
    signal: np.ndarray
    sampling_frequency: float
    resampled_signal: np.ndarray
    resampled_frequency: float
    quality: float
    peakedness: float

@dataclass(frozen=True)
class RespirationSegmentResult:
    features: np.ndarray
    selected: SelectedRespiration | None

def _build_resp_alias_groups(channels):
    resp_rows = channels[channels['Category'].eq('resp')].reset_index(drop=True)
    if len(resp_rows) < 7:
        return {}
    return {
        'Abdomen': split_channel_aliases(resp_rows.iloc[0]['Channel_Names']),
        'Chest': split_channel_aliases(resp_rows.iloc[1]['Channel_Names']),
        'Nasal': split_channel_aliases(resp_rows.iloc[2]['Channel_Names']),
        'Flow': split_channel_aliases(resp_rows.iloc[3]['Channel_Names']),
        'SpO2': split_channel_aliases(resp_rows.iloc[6]['Channel_Names']),
    }

def _get_resp_alias_groups(csv_path):
    channels, normalized_csv_path = get_cached_channel_table(csv_path)
    alias_groups = RESP_ALIAS_GROUPS_CACHE.get(normalized_csv_path)
    if alias_groups is None:
        alias_groups = _build_resp_alias_groups(channels)
        RESP_ALIAS_GROUPS_CACHE[normalized_csv_path] = alias_groups
    return alias_groups

def _find_resp_group(label, alias_groups):
    normalized = normalize_channel_label(label)
    for group_name, aliases in alias_groups.items():
        if normalized in aliases:
            return group_name
    return None

def get_respiration_feature_group(label, csv_path):
    """Return the production respiratory group, excluding SpO2 and extras."""

    group_name = _find_resp_group(label, _get_resp_alias_groups(csv_path))
    return group_name if group_name in RESP_CHANNEL_GROUPS else None

def _compute_resp_quality(used, hat_br):
    used_array = np.asarray(used, dtype=float)
    if used_array.size:
        quality = float(np.nanmean(used_array))
        if np.isfinite(quality):
            return quality
    hat_br = np.asarray(hat_br, dtype=float)
    if hat_br.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(hat_br)))

def _compute_peakedness_metric(hat_br):
    finite_values = np.asarray(hat_br, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return np.nan
    return float(np.mean(finite_values))

def _evaluate_direct_respiration(label, signal, sampling_frequency, group_name):
    original = np.asarray(signal, dtype=float)
    resampled, resampled_fs = resample_signal(
        original,
        sampling_frequency,
        25,
    )
    resampled = np.nan_to_num(
        resampled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    try:
        hat_br, _, _, used = resp_features.peakedness_application(
            resampled,
            stage=label,
            subject_id=label,
        )
    except Exception:
        return None

    peakedness = _compute_peakedness_metric(hat_br)
    if not np.isfinite(peakedness):
        return None

    return SelectedRespiration(
        label=label,
        group=group_name,
        signal=original,
        sampling_frequency=float(sampling_frequency),
        resampled_signal=np.asarray(resampled, dtype=float),
        resampled_frequency=float(resampled_fs),
        quality=_compute_resp_quality(used, hat_br),
        peakedness=peakedness,
    )

def select_best_respiration_signal(
    physiological_data,
    physiological_fs,
    csv_path,
):
    """Select one direct respiratory signal using production criteria.

    Only the four respiratory groups used by processResp are eligible.
    Returning None is intentional: downstream ECG/HRV processing must use
    sloperange in that case instead of silently selecting another EDF channel.
    """

    alias_groups = _get_resp_alias_groups(csv_path)
    best = None

    for label, signal in physiological_data.items():
        if label not in physiological_fs:
            continue

        group_name = _find_resp_group(label, alias_groups)
        if group_name not in RESP_CHANNEL_GROUPS:
            continue

        candidate = _evaluate_direct_respiration(
            label,
            signal,
            physiological_fs[label],
            group_name,
        )
        if candidate is None:
            continue
        if best is not None and candidate.quality <= best.quality:
            continue

        best = candidate

    return best

def _compute_spo2_segment_metrics(data, fs):
    if data.size == 0:
        return {}
    working = np.asarray(data, dtype=float).copy()
    if np.nanmax(working) < 2:
        working = np.round((working / 1.055) * 100)

    desaturation_mask = working.copy()
    threshold = 0.7
    for index, value in enumerate(working):
        if value < threshold:
            start = int(max(0, index - fs * 2))
            end = int(min(working.size, index + fs * 2))
            desaturation_mask[start:end] = np.nan

    cet90 = float(np.count_nonzero(desaturation_mask < 90) / max(working.size, 1))
    valid = desaturation_mask[np.isfinite(desaturation_mask)]
    if valid.size == 0:
        return {'CET90': cet90}

    odi_mean, odi_deepness = resp_features.odi_application(desaturation_mask, fs)
    return {
        'SpO2': float(np.mean(valid)),
        'CET90': cet90,
        'ODI': float(odi_mean),
        'ODI_deepness': float(odi_deepness),
    }

def process_respiration_segment(
    physiological_data,
    physiological_fs,
    csv_path,
):
    alias_groups = _get_resp_alias_groups(csv_path)
    results = {feature_name: np.nan for feature_name in RESP_SEGMENT_FEATURE_NAMES}
    best_quality = {group_name: -np.inf for group_name in RESP_CHANNEL_GROUPS}
    selected = None

    for label, signal in physiological_data.items():
        if label not in physiological_fs:
            continue

        group_name = _find_resp_group(label, alias_groups)
        if group_name is None:
            continue

        if group_name == 'SpO2':
            resampled, fs = resample_signal(
                signal,
                physiological_fs[label],
                25,
            )
            resampled = np.nan_to_num(
                resampled,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            results.update(_compute_spo2_segment_metrics(resampled, fs))
            continue

        candidate = _evaluate_direct_respiration(
            label,
            signal,
            physiological_fs[label],
            group_name,
        )
        if candidate is None:
            continue
        if selected is None or candidate.quality > selected.quality:
            selected = candidate
        if candidate.quality <= best_quality[group_name]:
            continue

        best_quality[group_name] = candidate.quality
        results[f'{group_name}_Peakedness'] = candidate.peakedness

    return RespirationSegmentResult(
        features=np.array(
            [results[name] for name in RESP_SEGMENT_FEATURE_NAMES],
            dtype=np.float32,
        ),
        selected=selected,
    )

def processResp(physiological_data, physiological_fs, csv_path):
    return process_respiration_segment(
        physiological_data,
        physiological_fs,
        csv_path,
    ).features
