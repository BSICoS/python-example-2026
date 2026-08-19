import pandas as pd
import numpy as np
from src.common.channel_utils import find_matching_label, get_cached_channel_table, normalize_channel_label, split_channel_aliases
from src.common.caisr import expand_stages_to_samples, p_nrem_at_time, weighted_nrem_minutes
from src.common.signal_utils import resample_signal
from .lib import eeg_features

EEG_CHANNEL_SPECS = {
    'C3-M2': {'direct': 'c3-m2', 'positive': 'c3', 'reference': 'm2'},
    'C4-M1': {'direct': 'c4-m1', 'positive': 'c4', 'reference': 'm1'},
    'F3-M2': {'direct': 'f3-m2', 'positive': 'f3', 'reference': 'm2'},
    'F4-M1': {'direct': 'f4-m1', 'positive': 'f4', 'reference': 'm1'},
}

# Segment-level background EEG metrics: aggregated later by the pipeline.
BACKGROUND_METRICS = [
    'Relative_Delta_Power',
    'Theta_Alpha_Ratio',
    'Theta_Beta_Ratio',
    'Rel_Beta',
    # Complexity & Dynamics
    'Hjorth_Complexity',
    'variability_Delta',
]
SLOW_WAVE_METRICS = [
    'NREM_SW_density', 'NREM_SW_p2p_median', 'NREM_SW_p2p_IQR',
    'NREM_SW_neg_slope_median', 'NREM_SW_neg_slope_IQR',
    'NREM_SW_neg_half_duration_median', 'NREM_SW_neg_half_duration_IQR',
]
EEG_BACKGROUND_FEATURE_SPECS = [
    (channel_name, feature_name)
    for channel_name in EEG_CHANNEL_SPECS
    for feature_name in BACKGROUND_METRICS
]
EEG_SEGMENT_FEATURE_NAMES = [f'{channel}_{metric}' for channel, metric in EEG_BACKGROUND_FEATURE_SPECS]
EEG_SEGMENT_FEATURE_LENGTH = len(EEG_SEGMENT_FEATURE_NAMES)
EEG_BACKGROUND_AGGREGATED_FEATURE_NAMES = [
    f'{name}_{aggregation}' for name in EEG_SEGMENT_FEATURE_NAMES
    for aggregation in ('Max', 'Min', 'Median', 'IQR')
]
EEG_SLOW_WAVE_FEATURE_NAMES = [
    f'{channel}_{metric}' for channel in EEG_CHANNEL_SPECS for metric in SLOW_WAVE_METRICS
]
EEG_FEATURE_NAMES = tuple((*EEG_BACKGROUND_AGGREGATED_FEATURE_NAMES, *EEG_SLOW_WAVE_FEATURE_NAMES))
EEG_FEATURE_LENGTH = len(EEG_FEATURE_NAMES)
EEG_ALIASES_CACHE = {}


def _build_eeg_aliases(channels):
    alias_lookup = {}
    for _, row in channels.iterrows():
        aliases = split_channel_aliases(row['Channel_Names'])
        if not aliases:
            continue
        canonical_name = normalize_channel_label(str(row['Channel_Names']).split(';')[0])
        alias_lookup[canonical_name] = aliases
    return alias_lookup


def _get_eeg_aliases(csv_path):
    channels, normalized_csv_path = get_cached_channel_table(csv_path)
    eeg_aliases = EEG_ALIASES_CACHE.get(normalized_csv_path)
    if eeg_aliases is None:
        eeg_aliases = _build_eeg_aliases(channels)
        EEG_ALIASES_CACHE[normalized_csv_path] = eeg_aliases
    return eeg_aliases


def get_eeg_channel_source_labels(channel_name, available_labels, physiological_fs, eeg_aliases):
    """Return the EDF labels selected by the production channel rules."""
    channel_spec = EEG_CHANNEL_SPECS[channel_name]
    label_lookup = dict.fromkeys(available_labels)
    direct_aliases = eeg_aliases.get(normalize_channel_label(channel_spec['direct']), set())
    direct_label = find_matching_label(label_lookup, direct_aliases)
    if direct_label is not None and direct_label in physiological_fs:
        return (direct_label,)

    positive_aliases = eeg_aliases.get(normalize_channel_label(channel_spec['positive']), set())
    reference_aliases = eeg_aliases.get(normalize_channel_label(channel_spec['reference']), set())
    positive_label = find_matching_label(label_lookup, positive_aliases)
    reference_label = find_matching_label(label_lookup, reference_aliases)
    if positive_label is None or reference_label is None:
        return None
    if positive_label not in physiological_fs or reference_label not in physiological_fs:
        return None

    positive_fs = physiological_fs[positive_label]
    reference_fs = physiological_fs[reference_label]
    if positive_fs != reference_fs:
        return None
    return positive_label, reference_label


def _get_channel_signal(channel_name, physiological_data, physiological_fs, eeg_aliases):
    source_labels = get_eeg_channel_source_labels(
        channel_name, physiological_data.keys(), physiological_fs, eeg_aliases)
    if source_labels is None:
        return None, None
    if len(source_labels) == 1:
        label = source_labels[0]
        return np.asarray(physiological_data[label], dtype=float), physiological_fs[label]

    positive_label, reference_label = source_labels
    return (
        np.asarray(physiological_data[positive_label], dtype=float)
        - np.asarray(physiological_data[reference_label], dtype=float),
        physiological_fs[positive_label],
    )


def prepare_slow_wave_detector_input(signal, fs):
    """Apply the exact sanitizing and resampling used before SW detection."""
    signal = np.nan_to_num(np.asarray(signal, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if signal.size < max(int(fs * 30), 2):
        return None

    if fs != 200:
        signal, fs = resample_signal(signal, fs, 200)

    return signal, fs


def _extract_channel_metrics(signal, fs):
    prepared = prepare_slow_wave_detector_input(signal, fs)
    if prepared is None:
        return None
    detector_signal, fs = prepared

    filtered = eeg_features.butter_bandpass_filter(detector_signal, lowcut=0.3, highcut=35, fs=fs, order=4)
    signal_std = np.std(filtered)
    if signal_std == 0 or not np.isfinite(signal_std):
        return None

    normalized = (filtered - np.mean(filtered)) / signal_std
    epochs = eeg_features.create_epochs(normalized, fs, epoch_duration=30)
    if epochs.size == 0:
        return None

    band_powers, complexities = eeg_features.extract_band_powers(epochs, fs, win_len=30)
    if len(band_powers) > 60:
        band_powers = band_powers.iloc[60:]
        complexities = complexities.iloc[60:]
    if band_powers.empty:
        return None

    patient_profile = eeg_features.get_patient_profile(band_powers)
    metrics = {
        str(name): float(value)
        for name, value in patient_profile.replace([np.inf, -np.inf], np.nan).items()
    }

    for complexity_name in ('Hjorth_Mobility', 'Hjorth_Complexity'):
        if complexity_name in complexities:
            value = complexities[complexity_name].replace([np.inf, -np.inf], np.nan).std()
            metrics[complexity_name] = float(np.nan if pd.isna(value) else value)
        else:
            metrics[complexity_name] = np.nan
            
    return metrics


def processEEG(physiological_data, physiological_fs, csv_path):
    eeg_aliases = _get_eeg_aliases(csv_path)
    channel_profiles = {}

    for channel_name in EEG_CHANNEL_SPECS:
        signal, fs = _get_channel_signal(channel_name, physiological_data, physiological_fs, eeg_aliases)
        if signal is None or fs is None:
            continue

        metrics = _extract_channel_metrics(signal, fs)
        if metrics is not None:
            channel_profiles[channel_name] = metrics

    if not channel_profiles:
        return np.full(EEG_SEGMENT_FEATURE_LENGTH, np.nan, dtype=np.float32)

    values = []
    for channel_name, metric_name in EEG_BACKGROUND_FEATURE_SPECS:
        channel_metrics = channel_profiles.get(channel_name)
        if channel_metrics is None:
            values.append(np.nan)
            continue
        values.append(float(channel_metrics.get(metric_name, np.nan)))

    return np.asarray(values, dtype=np.float32)


def _weighted_quantile(values, weights, quantile):
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    values, weights = values[valid], weights[valid]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), quantile * np.sum(weights), side='left')])


def extract_record_slow_wave_features(physiological_data, physiological_fs, csv_path, annotation):
    """Extract NREM-aware record-level slow-wave features for four EEG channels."""
    from src.pipeline.config import SEGMENT_DURATION_SECONDS, SEGMENT_STRIDE_SECONDS
    if not annotation.get('available'):
        return np.full(len(EEG_SLOW_WAVE_FEATURE_NAMES), np.nan, dtype=np.float32)
    aliases = _get_eeg_aliases(csv_path)
    result = []
    for channel_name in EEG_CHANNEL_SPECS:
        signal, fs = _get_channel_signal(channel_name, physiological_data, physiological_fs, aliases)
        if signal is None or fs is None:
            result.extend([np.nan] * len(SLOW_WAVE_METRICS)); continue
        event_weights, p2p, slopes, durations = [], [], [], []
        exposure = 0.0
        duration = len(signal) / float(fs)
        for start in np.arange(0.0, duration - SEGMENT_DURATION_SECONDS + 1e-9, SEGMENT_STRIDE_SECONDS):
            end = start + SEGMENT_DURATION_SECONDS
            raw = signal[int(round(start * fs)):int(round(end * fs))]
            prepared = prepare_slow_wave_detector_input(raw, fs)
            if prepared is None: continue
            detector_signal, detector_fs = prepared
            try:
                detection = eeg_features.detect_slow_waves(
                    detector_signal, detector_fs,
                    sleep_stages=expand_stages_to_samples(annotation, start, len(detector_signal), detector_fs),
                    allowed_stages=(1, 2))
            except Exception:
                continue
            exposure += weighted_nrem_minutes(annotation, start, end)
            for event in detection['events']:
                trough = start + float(np.asarray(event['Ref_PeakInd']).squeeze()) / detector_fs
                weight = p_nrem_at_time(annotation, trough)
                event_weights.append(weight)
                p2p.append(float(np.asarray(event['Ref_P2PAmp']).squeeze()))
                slopes.append(float(np.asarray(event['Ref_NegSlope']).squeeze()))
                durations.append((float(np.asarray(event['Ref_UpInd']).squeeze()) -
                                  float(np.asarray(event['Ref_DownInd']).squeeze())) / detector_fs)
        valid_weights = np.asarray(event_weights, dtype=float)
        count = float(np.sum(valid_weights[np.isfinite(valid_weights)]))
        density = count / exposure if exposure > 0 else np.nan
        q = lambda values, level: _weighted_quantile(values, valid_weights, level)
        p25, p50, p75 = q(p2p, .25), q(p2p, .5), q(p2p, .75)
        s25, s50, s75 = q(slopes, .25), q(slopes, .5), q(slopes, .75)
        d25, d50, d75 = q(durations, .25), q(durations, .5), q(durations, .75)
        result.extend([density, p50, p75 - p25, s50, s75 - s25, d50, d75 - d25])
    return np.asarray(result, dtype=np.float32)


_normalize_label = normalize_channel_label
